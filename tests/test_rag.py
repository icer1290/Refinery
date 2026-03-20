"""Tests for RAG system components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chunking import ChunkingService, TextChunk


class TestChunkingService:
    """Tests for the ChunkingService."""

    def test_chunk_short_text(self):
        """Test chunking a short text that fits in one chunk."""
        service = ChunkingService(chunk_size=500, chunk_overlap=50)
        text = "This is a short text."

        chunks = service.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_number == 0

    def test_chunk_long_text(self):
        """Test chunking a longer text into multiple chunks."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        # Create text that will need multiple chunks
        text = "\n\n".join([
            "Paragraph 1: " + "word " * 30,
            "Paragraph 2: " + "word " * 30,
            "Paragraph 3: " + "word " * 30,
        ])

        chunks = service.chunk_text(text)

        assert len(chunks) > 1
        # Check chunk numbers are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_number == i
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        service = ChunkingService()

        chunks = service.chunk_text("")

        assert chunks == []

    def test_chunk_with_metadata(self):
        """Test that chunk metadata is correct."""
        service = ChunkingService(chunk_size=200, chunk_overlap=50)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = service.chunk_text(text)

        for chunk in chunks:
            # Verify the chunk text is in the original text
            assert chunk.text in text or text[chunk.start_char:chunk.end_char].strip() == chunk.text.strip()

    def test_chunk_with_summary_first(self):
        """Test chunking with summary prepended."""
        service = ChunkingService(chunk_size=500, chunk_overlap=50)
        text = "This is the main content. " * 50
        summary = "This is a summary."

        chunks = service.chunk_text_with_summary_first(text, summary)

        assert len(chunks) >= 1
        # First chunk should contain the summary
        assert "摘要" in chunks[0].text or summary in chunks[0].text

    def test_chunk_preserves_paragraphs(self):
        """Test that chunking tries to preserve paragraph boundaries."""
        service = ChunkingService(chunk_size=200, chunk_overlap=50)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = service.chunk_text(text)

        # All chunks should be valid strings
        for chunk in chunks:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_chunk_chinese_text(self):
        """Test chunking Chinese text."""
        service = ChunkingService(chunk_size=200, chunk_overlap=50)
        text = "这是第一段内容。" * 20 + "\n\n" + "这是第二段内容。" * 20

        chunks = service.chunk_text(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "内容" in chunk.text

    def test_chunk_mixed_language_text(self):
        """Test chunking mixed Chinese and English text."""
        service = ChunkingService(chunk_size=300, chunk_overlap=50)
        text = """
        This is English content about AI and machine learning.

        这是一段中文内容，讨论人工智能和机器学习。

        More English content here about neural networks.

        更多关于神经网络的内容。
        """

        chunks = service.chunk_text(text)

        assert len(chunks) >= 1