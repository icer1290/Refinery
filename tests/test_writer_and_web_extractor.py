"""Tests for writer agent and web extractor error handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.agents.writer_agent import WriterAgent
from app.services.llm_service import LLMService
from app.services.web_extractor import WebExtractor


@pytest.fixture
def mock_settings():
    with patch("app.services.llm_service.settings") as mock:
        mock.openai_chat_model = "gpt-4o-mini"
        mock.openai_api_key = "test-key"
        mock.openai_base_url = None
        yield mock


@pytest.mark.asyncio
async def test_writer_accepts_malformed_structured_translation(mock_settings):
    """Writer should succeed when translation output is malformed but recoverable."""
    llm_service = LLMService(api_key="test-key")
    llm_service.llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content="""{
                "chinese_title": "[网络安全] 苹果首推后台安全更新修复Safari跨站漏洞",
                "第一段 (核心内容)": 苹果向运行iOS、iPadOS及macOS 26.1及以上版本的设备推送首个后台安全更新。
                "第二段 (关键事实)": · 漏洞存在于WebKit引擎。· 更新仅需快速重启设备。
                "第三段 (专家点评)": 主编洞察：苹果正在把高频轻量级补丁纳入常态化防御体系。
                }"""
            )
        )
    )

    writer = WriterAgent(max_concurrent=1)
    writer.llm_service = llm_service

    article = {
        "source_url": "https://example.com/apple-security",
        "original_title": "Apple rolls out first background security update",
        "original_description": "desc",
        "entities_preserved": ["WebKit"],
    }

    with patch.object(writer, "_extract_content", AsyncMock(return_value="full content")):
        processed, failed = await writer.execute([article])

    assert len(processed) == 1
    assert failed == []
    assert processed[0]["chinese_title"].startswith("[网络安全]")
    assert "主编洞察：" in processed[0]["chinese_summary"]


class _MockResponse:
    """Minimal response object for extractor tests."""

    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("GET", "https://example.com/article")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                    text=self.text,
                ),
            )


class _MockAsyncClient:
    """Minimal async client wrapper used to intercept fetch requests."""

    def __init__(self, responses: list[_MockResponse], captured_headers: list[dict[str, str]]):
        self._responses = responses
        self._captured_headers = captured_headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict[str, str]):
        self._captured_headers.append(headers)
        response = self._responses.pop(0)
        response.request = httpx.Request("GET", url, headers=headers)
        return response


@pytest.mark.asyncio
async def test_web_extractor_retries_429_and_sends_browser_headers():
    """429 responses should be retried with browser-like headers."""
    extractor = WebExtractor(max_retries=2, retry_delay=0)
    responses = [
        _MockResponse(429),
        _MockResponse(200, "<html><body>ok</body></html>"),
    ]
    captured_headers: list[dict[str, str]] = []

    with patch(
        "app.services.web_extractor.httpx.AsyncClient",
        side_effect=lambda *args, **kwargs: _MockAsyncClient(responses, captured_headers),
    ), patch("app.services.web_extractor.asyncio.sleep", AsyncMock()):
        html = await extractor._fetch_page("https://venturebeat.com/article")

    assert html == "<html><body>ok</body></html>"
    assert len(captured_headers) == 2
    assert "Mozilla/5.0" in captured_headers[0]["User-Agent"]
    assert captured_headers[0]["Referer"] == "https://venturebeat.com"


@pytest.mark.asyncio
async def test_web_extractor_surfaces_status_code_after_retry_exhaustion():
    """Retry exhaustion should keep structured status details."""
    extractor = WebExtractor(max_retries=1, retry_delay=0)
    responses = [_MockResponse(429), _MockResponse(429)]
    captured_headers: list[dict[str, str]] = []

    with patch(
        "app.services.web_extractor.httpx.AsyncClient",
        side_effect=lambda *args, **kwargs: _MockAsyncClient(responses, captured_headers),
    ), patch("app.services.web_extractor.asyncio.sleep", AsyncMock()):
        with pytest.raises(Exception) as exc_info:
            await extractor._fetch_page("https://venturebeat.com/article")

    error = exc_info.value
    assert getattr(error, "details", {})["status_code"] == 429
    assert getattr(error, "details", {})["host"] == "venturebeat.com"
