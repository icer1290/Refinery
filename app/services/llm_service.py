"""LLM service for text generation and scoring."""

import json
import re
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core import get_logger
from app.core.exceptions import LLMError
from app.models.schemas import ReflectionResult, ScoringResult, TranslationResult
from app.prompts import get_prompt
from app.prompts.formatters import (
    build_content_section,
    build_entities_section,
    build_feedback_section,
)

logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """Service for LLM operations using LangChain and OpenAI-compatible APIs."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or settings.openai_chat_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        if not self.api_key:
            raise LLMError("API key not configured")

        # Build ChatOpenAI with optional custom base_url
        llm_kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": settings.llm_temperature,
            "extra_body": {"enable_thinking": settings.llm_enable_thinking},
        }

        # Add base_url if provided (for DashScope, etc.)
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        self.llm = ChatOpenAI(**llm_kwargs)

        logger.info(
            "LLM service initialized",
            model=self.model,
            base_url=self.base_url or "default",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def score_article(
        self,
        title: str,
        description: str,
        content: Optional[str] = None,
    ) -> ScoringResult:
        """Score an article on multiple dimensions.

        Args:
            title: Article title
            description: Article description
            content: Full article content (optional)

        Returns:
            ScoringResult with scores and reasoning
        """
        prompt = self._build_scoring_prompt(title, description, content)

        try:
            response = await self.llm.ainvoke(prompt)
            result = self._parse_scoring_response(response.content)

            logger.debug(
                "Scored article",
                title=title[:50],
                total_score=result.total_score,
            )

            return result

        except Exception as e:
            logger.warning(
                "Score article attempt failed",
                title=title[:80],
                error_type=type(e).__name__,
                error=str(e),
            )
            raise LLMError(
                f"Failed to score article: {str(e)}",
                {"title": title, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def translate_and_summarize(
        self,
        title: str,
        content: str,
        entities_to_preserve: Optional[list[str]] = None,
        feedback: Optional[str] = None,
    ) -> TranslationResult:
        """Translate and summarize article to Chinese.

        Args:
            title: Original article title
            content: Full article content
            entities_to_preserve: List of entities to keep in original language
            feedback: Feedback from previous reflection for retry (optional)

        Returns:
            TranslationResult with Chinese title and summary
        """
        prompt = self._build_translation_prompt(title, content, entities_to_preserve, feedback)

        try:
            response = await self.llm.ainvoke(prompt)
            result = self._parse_translation_response(response.content)

            logger.info(
                "Translated article",
                original_title=title[:50],
                chinese_title=result.chinese_title[:50],
            )

            return result

        except Exception as e:
            logger.warning(
                "Translate article attempt failed",
                title=title[:80],
                error_type=type(e).__name__,
                error=str(e),
            )
            raise LLMError(
                f"Failed to translate article: {str(e)}",
                {"title": title, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def reflect(
        self,
        chinese_title: str,
        chinese_summary: str,
        original_title: str,
        original_content: str,
    ) -> ReflectionResult:
        """Reflect on translation quality.

        Args:
            chinese_title: Translated Chinese title
            chinese_summary: Translated Chinese summary
            original_title: Original article title
            original_content: Original article content

        Returns:
            ReflectionResult with pass/fail and feedback
        """
        prompt = self._build_reflection_prompt(
            chinese_title,
            chinese_summary,
            original_title,
            original_content,
        )

        try:
            response = await self.llm.ainvoke(prompt)
            result = self._parse_reflection_response(response.content)

            logger.debug(
                "Reflected on translation",
                passed=result.passed,
                issues_count=len(result.issues),
            )

            return result

        except Exception as e:
            logger.warning(
                "Reflection attempt failed",
                original_title=original_title[:80],
                error_type=type(e).__name__,
                error=str(e),
            )
            raise LLMError(
                f"Failed to reflect on translation: {str(e)}",
                {"chinese_title": chinese_title, "error": str(e)},
            )

    def _build_scoring_prompt(
        self, title: str, description: str, content: Optional[str]
    ) -> str:
        """Build prompt for article scoring."""
        content_section = build_content_section(content)
        return get_prompt("scoring.article").format(
            title=title,
            description=description,
            content_section=content_section,
        )

    def _build_translation_prompt(
        self,
        title: str,
        content: str,
        entities_to_preserve: Optional[list[str]] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Build prompt for translation and summarization."""
        entities_section = build_entities_section(entities_to_preserve)
        feedback_section = build_feedback_section(feedback)
        return get_prompt("translation.article").format(
            title=title,
            content=content,
            entities_section=entities_section,
            feedback_section=feedback_section,
        )

    def _build_reflection_prompt(
        self,
        chinese_title: str,
        chinese_summary: str,
        original_title: str,
        original_content: str,
    ) -> str:
        """Build prompt for reflection check."""
        content_preview = original_content[:500] if original_content else "（无内容）"
        return get_prompt("reflection.check").format(
            original_title=original_title,
            original_content_preview=content_preview,
            chinese_title=chinese_title,
            chinese_summary=chinese_summary,
        )

    def _parse_scoring_response(self, response: str) -> ScoringResult:
        """Parse LLM response for scoring."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            industry_impact = float(data["industry_impact_score"])
            milestone = float(data["milestone_score"])
            attention = float(data["attention_score"])

            # Calculate weighted total score using config weights
            total = (
                settings.scoring_weight_industry_impact * industry_impact
                + settings.scoring_weight_milestone * milestone
                + settings.scoring_weight_attention * attention
            )

            return ScoringResult(
                industry_impact_score=industry_impact,
                milestone_score=milestone,
                attention_score=attention,
                total_score=total,
                reasoning=data.get("reasoning"),
            )
        except Exception as e:
            raise LLMError(f"Failed to parse scoring response: {str(e)}")

    def _parse_translation_response(self, response: str) -> TranslationResult:
        """Parse LLM response for translation."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            result = TranslationResult(
                chinese_title=data["chinese_title"],
                chinese_summary=data["chinese_summary"],
                entities_preserved=data.get("entities_preserved", []),
            )
            logger.info(
                "Parsed translation response",
                parser_mode="json",
                chinese_title=result.chinese_title[:80],
            )
            return result
        except Exception as e:
            structured_fallback = self._parse_structured_translation(response)
            if structured_fallback is not None:
                logger.info(
                    "Parsed translation response",
                    parser_mode="structured_fallback",
                    chinese_title=structured_fallback.chinese_title[:80],
                )
                return structured_fallback

            fallback = self._parse_plaintext_translation(response)
            if fallback is not None:
                logger.info(
                    "Parsed translation response",
                    parser_mode="plaintext_fallback",
                    chinese_title=fallback.chinese_title[:80],
                )
                return fallback

            preview = str(response).replace("\n", "\\n")[:500]
            raise LLMError(
                f"Failed to parse translation response: {str(e)} | response_preview={preview}"
            )

    def _parse_structured_translation(self, response: str) -> TranslationResult | None:
        """Recover translation output from malformed JSON-like structured text."""
        text = str(response).strip()
        if not text:
            return None
        if '"chinese_title"' not in text and '"chinese_summary"' not in text:
            return None

        title_payload = self._extract_raw_json_field(
            text,
            "chinese_title",
            next_keys=[
                "chinese_summary",
                "entities_preserved",
                "第一段 (核心内容)",
                "第二段 (关键事实)",
                "第三段 (专家点评)",
            ],
        )
        summary_payload = self._extract_raw_json_field(
            text,
            "chinese_summary",
            next_keys=["entities_preserved"],
        )

        overflow_summary = ""
        title = self._clean_extracted_json_value(title_payload)
        if title:
            title, overflow_summary = self._split_title_and_overflow(title)

        if not title.startswith("[") or "]" not in title:
            title = self._extract_title_from_lines(text)
        if not title:
            return None

        if summary_payload:
            summary = self._clean_extracted_json_value(summary_payload)
        else:
            summary = ""

        if overflow_summary:
            summary = f"{overflow_summary}\n\n{summary}".strip() if summary else overflow_summary

        sections = self._extract_structured_sections(text)
        if not summary and len(sections) >= 3:
            summary = "\n\n".join(section for section in sections if section).strip()
        elif sections:
            summary = self._merge_summary_sections(summary, sections)

        if not summary:
            return None

        entities = self._extract_entities_preserved(text)
        return TranslationResult(
            chinese_title=title,
            chinese_summary=summary,
            entities_preserved=entities,
        )

    def _parse_plaintext_translation(self, response: str) -> TranslationResult | None:
        """Fallback parser for models that return formatted plain text instead of JSON."""
        text = str(response).strip()
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        title = lines[0]
        if not title.startswith("[") or "]" not in title:
            return None

        summary = "\n".join(lines[1:]).strip()
        if not summary:
            return None

        return TranslationResult(
            chinese_title=title,
            chinese_summary=summary,
            entities_preserved=[],
        )

    def _extract_title_from_lines(self, text: str) -> str:
        """Extract a bracketed title from the first non-empty lines."""
        lines = [line.strip().strip('",') for line in text.splitlines() if line.strip()]
        for line in lines[:5]:
            if line.startswith("[") and "]" in line:
                return line
        return ""

    def _extract_structured_sections(self, text: str) -> list[str]:
        """Extract the three expected summary sections from malformed JSON-like text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: list[str] = []
        current: str | None = None

        for raw_line in lines:
            line = raw_line.strip().strip(",")
            if line.startswith("{") or line.startswith("}") or line.startswith('"chinese_title"'):
                continue
            if line.startswith('"entities_preserved"'):
                continue

            normalized = self._normalize_structured_line(line)
            if not normalized:
                continue

            if self._is_section_start(normalized):
                if current:
                    sections.append(current.strip())
                current = normalized
                continue

            if current:
                current = f"{current}\n{normalized}".strip()

        if current:
            sections.append(current.strip())

        cleaned_sections = [self._clean_section_text(section) for section in sections]
        return [section for section in cleaned_sections if section]

    def _normalize_structured_line(self, line: str) -> str:
        """Normalize line wrappers from malformed structured output."""
        normalized = line.strip().strip('"').strip()
        normalized = re.sub(r"^\d+\.\s*\*+\s*", "", normalized)
        normalized = re.sub(r"\*+\s*$", "", normalized)
        normalized = normalized.replace('\\"', '"')
        normalized = normalized.replace("\\n", "\n")
        normalized = normalized.replace("**", "")
        return normalized.strip()

    def _is_section_start(self, line: str) -> bool:
        """Check whether a line starts one of the expected summary sections."""
        patterns = (
            "第一段",
            "第二段",
            "第三段",
            "主编洞察",
        )
        return any(line.startswith(pattern) for pattern in patterns)

    def _clean_section_text(self, section: str) -> str:
        """Remove malformed key wrappers while keeping the summary content intact."""
        cleaned = section.strip().strip('"').strip(",").strip()
        cleaned = re.sub(r"^(第一段[^：:]*[：:])\s*", "", cleaned)
        cleaned = re.sub(r"^(第二段[^：:]*[：:])\s*", "", cleaned)
        cleaned = re.sub(r"^(第三段[^：:]*[：:])\s*", "", cleaned)

        if cleaned.startswith("主编洞察"):
            return cleaned

        return cleaned

    def _extract_raw_json_field(
        self,
        text: str,
        field_name: str,
        next_keys: list[str],
    ) -> str:
        """Extract a raw JSON field value without requiring valid JSON escaping."""
        key_pattern = f'"{field_name}"'
        key_index = text.find(key_pattern)
        if key_index == -1:
            return ""

        colon_index = text.find(":", key_index + len(key_pattern))
        if colon_index == -1:
            return ""

        value_start = colon_index + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1

        if value_start >= len(text):
            return ""

        if text[value_start] == '"':
            value_start += 1

        candidate_markers = []
        for next_key in next_keys:
            marker = f'",\n    "{next_key}"'
            idx = text.find(marker, value_start)
            if idx != -1:
                candidate_markers.append(idx)

            marker = f'",\n"{next_key}"'
            idx = text.find(marker, value_start)
            if idx != -1:
                candidate_markers.append(idx)

            marker = f'", "{next_key}"'
            idx = text.find(marker, value_start)
            if idx != -1:
                candidate_markers.append(idx)

        end_marker = text.find("\n}", value_start)
        if end_marker != -1:
            candidate_markers.append(end_marker)

        inline_end_marker = text.find('"}', value_start)
        if inline_end_marker != -1:
            candidate_markers.append(inline_end_marker)

        if candidate_markers:
            value_end = min(candidate_markers)
        else:
            value_end = len(text)

        return text[value_start:value_end]

    def _clean_extracted_json_value(self, value: str) -> str:
        """Normalize a loosely extracted JSON string value."""
        cleaned = value.strip()
        cleaned = cleaned.rstrip('", \n\r\t')
        cleaned = cleaned.lstrip('"')
        cleaned = cleaned.replace('\\"', '"')
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\\t", "\t")
        return cleaned.strip()

    def _split_title_and_overflow(self, title: str) -> tuple[str, str]:
        """Split title payload when the model accidentally appends summary content."""
        if "\n" not in title:
            return title.strip(), ""

        first_line, remainder = title.split("\n", 1)
        return first_line.strip(), remainder.strip()

    def _merge_summary_sections(self, summary: str, sections: list[str]) -> str:
        """Merge extracted sections into an existing summary without duplication."""
        merged = summary.strip()
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if section in merged:
                continue
            merged = f"{merged}\n\n{section}".strip() if merged else section
        return merged

    def _extract_entities_preserved(self, text: str) -> list[str]:
        """Extract entities_preserved from valid JSON fragments when available."""
        match = re.search(
            r'"entities_preserved"\s*:\s*\[(?P<entities>[^\]]*)\]',
            text,
            re.DOTALL,
        )
        if not match:
            return []

        entities = re.findall(r'"([^"]+)"', match.group("entities"))
        return [entity.strip() for entity in entities if entity.strip()]

    def _parse_reflection_response(self, response: str) -> ReflectionResult:
        """Parse LLM response for reflection."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return ReflectionResult(
                passed=data["passed"],
                feedback=data.get("feedback"),
                issues=data.get("issues", []),
            )
        except Exception as e:
            raise LLMError(f"Failed to parse reflection response: {str(e)}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown code blocks."""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        # Try to find JSON object directly
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
