"""Tests for RSS parser service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.rss_parser import RSSParser


@pytest.fixture
def rss_parser():
    return RSSParser()


@pytest.mark.asyncio
async def test_generate_hash(rss_parser):
    """Test content hash generation."""
    hash1 = rss_parser._generate_hash("Test content")
    hash2 = rss_parser._generate_hash("Test content")
    hash3 = rss_parser._generate_hash("Different content")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA256 produces 64 hex chars


@pytest.mark.asyncio
async def test_parse_entry(rss_parser):
    """Test parsing a single RSS entry."""
    entry = MagicMock()
    entry.title = "Test Article"
    entry.link = "https://example.com/article"
    entry.description = "Test description"
    entry.published_parsed = (2024, 1, 1, 12, 0, 0, 0, 0, 0)

    result = rss_parser._parse_entry(entry, "Test Source")

    assert result["source_name"] == "Test Source"
    assert result["source_url"] == "https://example.com/article"
    assert result["original_title"] == "Test Article"
    assert result["original_description"] == "Test description"
    assert "content_hash" in result


@pytest.mark.asyncio
async def test_parse_entry_missing_fields(rss_parser):
    """Test parsing entry with missing required fields."""
    entry = MagicMock()
    entry.title = ""
    entry.link = ""

    with pytest.raises(ValueError):
        rss_parser._parse_entry(entry, "Test Source")