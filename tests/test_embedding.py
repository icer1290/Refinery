"""Tests for embedding service."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.exceptions import EmbeddingError
from app.services.embedding import EmbeddingService
from app.workflow.nodes import storage_node


@pytest.fixture
def mock_settings():
    with patch("app.services.embedding.settings") as mock:
        mock.openai_embedding_model = "text-embedding-3-small"
        mock.openai_api_key = "test-key"
        yield mock


@pytest.mark.asyncio
async def test_truncate_text(mock_settings):
    """Test text truncation."""
    service = EmbeddingService(api_key="test-key")

    short_text = "Short text"
    long_text = "x" * 40000

    truncated = service._truncate_text(long_text, max_tokens=8000)
    assert len(truncated) < len(long_text)
    assert truncated.endswith("...")

    not_truncated = service._truncate_text(short_text)
    assert not_truncated == short_text


@pytest.mark.asyncio
async def test_compute_similarity(mock_settings):
    """Test cosine similarity computation."""
    service = EmbeddingService(api_key="test-key")

    embedding1 = [1.0, 0.0, 0.0]
    embedding2 = [1.0, 0.0, 0.0]
    embedding3 = [0.0, 1.0, 0.0]

    # Same vector
    similarity = await service.compute_similarity(embedding1, embedding2)
    assert similarity == pytest.approx(1.0, rel=0.01)

    # Orthogonal vector
    similarity = await service.compute_similarity(embedding1, embedding3)
    assert similarity == pytest.approx(0.0, abs=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_count", "expected_batch_sizes"),
    [
        (1, [1]),
        (10, [10]),
        (11, [10, 1]),
        (21, [10, 10, 1]),
    ],
)
async def test_embed_batch_dashscope_respects_max_batch_size(
    mock_settings, input_count, expected_batch_sizes
):
    """DashScope batch embedding requests should be split at 10 items."""
    service = EmbeddingService(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    texts = [f"text-{i}" for i in range(input_count)]
    mock_batches = [
        [[batch_index, item_index] for item_index in range(batch_size)]
        for batch_index, batch_size in enumerate(expected_batch_sizes)
    ]
    service._embed_batch_with_dashscope = AsyncMock(side_effect=mock_batches)

    embeddings = await service.embed_batch(texts)

    assert service._embed_batch_with_dashscope.await_count == len(expected_batch_sizes)
    assert [
        len(call.args[0]) for call in service._embed_batch_with_dashscope.await_args_list
    ] == expected_batch_sizes
    assert len(embeddings) == input_count


@pytest.mark.asyncio
async def test_storage_node_surfaces_storage_errors(mock_settings):
    """Storage failures should be returned as workflow errors."""

    @asynccontextmanager
    async def begin_nested():
        yield

    session = MagicMock()
    session.begin_nested = begin_nested
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))
    session.add = MagicMock()
    session.flush = AsyncMock()

    runtime = SimpleNamespace(context=SimpleNamespace(session=session))
    state = {
        "run_id": "test-run",
        "final_articles": [{
            "source_name": "test",
            "source_url": "https://example.com/article",
            "original_title": "Example",
            "original_description": "Description",
            "full_content": "x" * 200,
            "chinese_title": "[测试] 标题",
            "chinese_summary": "摘要",
        }],
    }

    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_text = AsyncMock(
        side_effect=EmbeddingError("boom", {"reason": "test"})
    )
    mock_chunking_service = MagicMock()

    with patch("app.workflow.nodes.get_embedding_service", return_value=mock_embedding_service), patch(
        "app.workflow.nodes.get_chunking_service", return_value=mock_chunking_service
    ):
        result = await storage_node(state, runtime)

    assert result["total_articles_stored"] == 0
    assert result["stored_article_ids"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0]["phase"] == "storage"
    assert result["errors"][0]["details"]["source_url"] == "https://example.com/article"
