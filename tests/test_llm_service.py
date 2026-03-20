"""Tests for LLM service."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.llm_service import LLMService
from app.models.schemas import ScoringResult, TranslationResult, ReflectionResult


@pytest.fixture
def mock_settings():
    with patch("app.services.llm_service.settings") as mock:
        mock.openai_chat_model = "gpt-4o-mini"
        mock.openai_api_key = "test-key"
        mock.openai_base_url = None
        yield mock


@pytest.mark.asyncio
async def test_extract_json():
    """Test JSON extraction from LLM response."""
    service = LLMService(api_key="test-key")

    # Plain JSON
    json_text = '{"key": "value"}'
    extracted = service._extract_json(json_text)
    assert extracted == '{"key": "value"}'

    # JSON in code block
    code_block = '```json\n{"key": "value"}\n```'
    extracted = service._extract_json(code_block)
    assert extracted == '{"key": "value"}'

    # JSON with surrounding text
    mixed_text = 'Some text {"key": "value"} more text'
    extracted = service._extract_json(mixed_text)
    assert extracted == '{"key": "value"}'


@pytest.mark.asyncio
async def test_parse_scoring_response(mock_settings):
    """Test parsing scoring response."""
    service = LLMService(api_key="test-key")

    response = '{"industry_impact_score": 8, "milestone_score": 7, "attention_score": 9, "reasoning": "Test"}'
    result = service._parse_scoring_response(response)

    assert isinstance(result, ScoringResult)
    assert result.industry_impact_score == 8
    assert result.milestone_score == 7
    assert result.attention_score == 9
    # Total = 0.4 * 8 + 0.35 * 7 + 0.25 * 9 = 3.2 + 2.45 + 2.25 = 7.9
    assert result.total_score == pytest.approx(7.9, rel=0.01)


@pytest.mark.asyncio
async def test_parse_translation_response(mock_settings):
    """Test parsing translation response."""
    service = LLMService(api_key="test-key")

    response = '{"chinese_title": "中文标题", "chinese_summary": "中文摘要", "entities_preserved": ["OpenAI"]}'
    result = service._parse_translation_response(response)

    assert isinstance(result, TranslationResult)
    assert result.chinese_title == "中文标题"
    assert result.chinese_summary == "中文摘要"
    assert "OpenAI" in result.entities_preserved


@pytest.mark.asyncio
async def test_parse_translation_response_with_structured_fallback(mock_settings):
    """Malformed JSON-like translation output should be salvaged."""
    service = LLMService(api_key="test-key")

    response = """{
    "chinese_title": "[网络安全] 苹果首推后台安全更新修复Safari跨站漏洞",
    "第一段 (核心内容)": 苹果向运行iOS、iPadOS及macOS 26.1及以上版本的设备推送首个“后台安全改进”更新，成功修补WebKit引擎中允许恶意网站窃取同会话数据的漏洞。
    "第二段 (关键事实)": · 漏洞源于WebKit引擎，可被利用以访问同一浏览器会话中的其他网站数据。
    "第三段 (专家点评)": 主编洞察：此更新不仅是技术迭代，更是防御范式的根本转变。
    }"""
    result = service._parse_translation_response(response)

    assert isinstance(result, TranslationResult)
    assert result.chinese_title == "[网络安全] 苹果首推后台安全更新修复Safari跨站漏洞"
    assert "苹果向运行iOS" in result.chinese_summary
    assert "漏洞源于WebKit引擎" in result.chinese_summary
    assert "主编洞察：" in result.chinese_summary


@pytest.mark.asyncio
async def test_parse_translation_response_with_chinese_punctuation_keys(mock_settings):
    """Structured fallback should tolerate Chinese punctuation and markdown numbering."""
    service = LLMService(api_key="test-key")

    response = """{
    "chinese_title": "[智能家居] Aqara 推出首款 Matter 认证摄像头，重构跨品牌互联逻辑",
    "1. **第一段 (核心内容)**：Aqara 发布 G350 与 G400 两款新品，确立其作为首个 Matter 认证摄像头的行业地位。",
    "2. **第二段 (关键事实)**：· G350 支持 4K 广角与 2.5K 长焦双摄；· G400 门铃支持以太网或 Wi-Fi 6 连接。",
    "3. **第三段 (专家点评)**：主编洞察：Matter 认证不再是营销噱头，而是硬件厂商的入场券。",
    "entities_preserved": ["Matter", "Aqara"]
    }"""
    result = service._parse_translation_response(response)

    assert isinstance(result, TranslationResult)
    assert result.chinese_title.startswith("[智能家居]")
    assert "Aqara 发布 G350" in result.chinese_summary
    assert "G400 门铃支持以太网" in result.chinese_summary
    assert result.entities_preserved == ["Matter", "Aqara"]


@pytest.mark.asyncio
async def test_parse_translation_response_with_plaintext_fallback(mock_settings):
    """Plaintext fallback should still work after structured parser changes."""
    service = LLMService(api_key="test-key")

    response = """[图形技术] 玩家集体抵制英伟达DLSS 5
英伟达发布DLSS 5后，玩家对画质与延迟成本的反弹迅速扩大。

· 新版本进一步依赖AI插帧。
· 玩家批评其掩盖原生性能不足。

主编洞察：图形技术路线正在从绝对性能转向感知性能管理。"""
    result = service._parse_translation_response(response)

    assert isinstance(result, TranslationResult)
    assert result.chinese_title == "[图形技术] 玩家集体抵制英伟达DLSS 5"
    assert "玩家对画质与延迟成本的反弹迅速扩大" in result.chinese_summary


@pytest.mark.asyncio
async def test_parse_translation_response_with_summary_embedded_in_title(mock_settings):
    """Structured fallback should recover when summary is accidentally appended to chinese_title."""
    service = LLMService(api_key="test-key")

    response = """{
    "chinese_title": "[搜索] Kagi 移动端上线“小网”人工内容库
Kagi 正式推出 iOS 与 Android 应用，将其筛选的 30,000+ 个非商业、纯人工创作网站集成至移动端，直面 AI 生成内容泛滥的搜索困境。

· 索引规模：收录超过 30,000 个符合“小网”定义的个人博客、漫画及独立视频站点。
· 功能特性：支持按类别（如代码库、视频）筛选，提供无干扰阅读模式及收藏功能。

主编洞察：Kagi 试图通过“人工策展”构建差异化护城河。", 
    "entities_preserved": ["Kagi", "iOS", "Android"]
    }"""
    result = service._parse_translation_response(response)

    assert result.chinese_title == "[搜索] Kagi 移动端上线“小网”人工内容库"
    assert "Kagi 正式推出 iOS 与 Android 应用" in result.chinese_summary
    assert "主编洞察：" in result.chinese_summary
    assert result.entities_preserved == ["Kagi", "iOS", "Android"]


@pytest.mark.asyncio
async def test_parse_translation_response_with_unescaped_quotes_in_summary(mock_settings):
    """Structured fallback should recover summary text containing unescaped quotes."""
    service = LLMService(api_key="test-key")

    response = """{
    "chinese_title": "[AI 工程] YC CEO Garry Tan 开源\"god mode\"引发两极争议",
    "chinese_summary": "YC 创始人 Garry Tan 公开其基于 Claude Code 的"gstack"自动化开发配置，GitHub 获近 20,000 星标。

· Tan 于3月12日开源包含6项初始技能的"gstack"系统。

主编洞察：这场争论本质是技术信仰者与务实派的碰撞。",
    "entities_preserved": ["Claude Code", "gstack", "GitHub"]
    }"""
    result = service._parse_translation_response(response)

    assert result.chinese_title.startswith("[AI 工程]")
    assert '"gstack"自动化开发配置' in result.chinese_summary
    assert "主编洞察：" in result.chinese_summary
    assert result.entities_preserved == ["Claude Code", "gstack", "GitHub"]


@pytest.mark.asyncio
async def test_parse_reflection_response(mock_settings):
    """Test parsing reflection response."""
    service = LLMService(api_key="test-key")

    response = '{"passed": true, "issues": [], "feedback": null}'
    result = service._parse_reflection_response(response)

    assert isinstance(result, ReflectionResult)
    assert result.passed is True
    assert len(result.issues) == 0

    # Failed reflection
    response = '{"passed": false, "issues": ["问题1"], "feedback": "改进建议"}'
    result = service._parse_reflection_response(response)

    assert result.passed is False
    assert len(result.issues) == 1
    assert result.feedback == "改进建议"
