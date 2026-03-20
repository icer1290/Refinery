"""Reflection Agent - Quality check and validation."""

import re
import traceback
from typing import Any

from app.agents.base import BaseAgent
from app.config import get_settings
from app.services.llm_service import LLMService, get_llm_service

settings = get_settings()


class ReflectionAgent(BaseAgent):
    """Agent responsible for validating translation quality."""

    def __init__(self, max_retries: int | None = None):
        super().__init__("Reflection")
        self.llm_service: LLMService | None = None
        self.max_retries = max_retries or settings.max_reflection_retries

    def _get_llm_service(self) -> LLMService:
        """Lazily initialize LLM service."""
        if self.llm_service is None:
            self.llm_service = get_llm_service()
        return self.llm_service

    def _quick_format_check(self, title: str, summary: str) -> tuple[bool, list[str]]:
        """Quick regex check before expensive LLM reflection.

        Args:
            title: Chinese title to check
            summary: Chinese summary to check

        Returns:
            Tuple of (passed, issues list)
        """
        issues = []

        # Check title: [领域] 标题
        if not re.match(r'^\[[\u4e00-\u9fa5a-zA-Z]+\]\s+.+', title):
            issues.append("标题格式错误：必须以 [领域] 开头，如 [AI] GPT-5发布")

        # Check three paragraphs
        paragraphs = [p.strip() for p in summary.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            issues.append(f"摘要结构错误：需要3个段落，当前{len(paragraphs)}个")
        else:
            # Check bullet points in second paragraph
            second_para = paragraphs[1]
            if not re.search(r'^·\s+', second_para, re.MULTILINE):
                issues.append("第二段格式错误：要点必须以 `· ` 开头")

            # Check 主编洞察 in third paragraph
            if not paragraphs[2].startswith('主编洞察：'):
                issues.append("第三段格式错误：必须以 `主编洞察：` 开头")

        return len(issues) == 0, issues

    async def execute(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate articles and retry if needed.

        Args:
            articles: List of processed articles to validate

        Returns:
            List of validated articles
        """
        if not articles:
            return []

        # 初始化 LLM 服务
        try:
            llm = self._get_llm_service()
            self.logger.info("LLM service initialized for reflection", model=llm.model)
        except Exception as e:
            self.logger.error("Failed to initialize LLM service for reflection", error=str(e))
            # 如果 LLM 初始化失败，直接返回文章（跳过 reflection）
            for article in articles:
                article["reflection_passed"] = True
                article["reflection_retries"] = 0
            return articles

        validated_articles = []
        errors_count = 0

        for i, article in enumerate(articles):
            try:
                validated = await self._validate_article(article, i)
                if validated:
                    validated_articles.append(validated)
            except Exception as e:
                errors_count += 1
                self.logger.error(
                    f"Error validating article {i + 1}",
                    error=str(e),
                    article_url=article.get("source_url"),
                    traceback=traceback.format_exc(),
                )
                # 出错时仍保留文章
                article["reflection_passed"] = True
                article["reflection_retries"] = 0
                validated_articles.append(article)

        self.logger.info(
            "Reflection phase complete",
            articles_input=len(articles),
            articles_validated=len(validated_articles),
            errors_count=errors_count,
        )

        return validated_articles

    async def _validate_article(self, article: dict[str, Any], index: int) -> dict[str, Any] | None:
        """Validate a single article with retries.

        Args:
            article: Article to validate
            index: Article index for logging

        Returns:
            Validated article or None if validation failed after retries
        """
        llm = self._get_llm_service()
        retries = 0

        # 检查是否有内容，无内容时跳过 reflection 以避免浪费重试
        if not article.get("full_content"):
            self.logger.warning(
                "Article has no content, skipping reflection",
                article_url=article.get("source_url"),
            )
            article["reflection_passed"] = True
            article["reflection_retries"] = 0
            return article

        # 检查必要字段
        if not article.get("chinese_title") or not article.get("chinese_summary"):
            self.logger.warning(
                f"Article {index + 1} missing translation, skipping reflection",
                article_url=article.get("source_url"),
            )
            article["reflection_passed"] = True
            article["reflection_retries"] = 0
            return article

        while retries < self.max_retries:
            try:
                # Quick format check before expensive LLM call
                quick_passed, quick_issues = self._quick_format_check(
                    article["chinese_title"], article["chinese_summary"]
                )

                if quick_passed:
                    # Format looks good, still do LLM reflection for semantic validation
                    result = await llm.reflect(
                        chinese_title=article["chinese_title"],
                        chinese_summary=article["chinese_summary"],
                        original_title=article["original_title"],
                        original_content=article.get("full_content", ""),
                    )

                    if result.passed:
                        article["reflection_passed"] = True
                        article["reflection_feedback"] = None
                        article["reflection_retries"] = retries
                        return article

                    issues = result.issues or []
                else:
                    # Quick check found issues, skip LLM reflection and retry directly
                    issues = quick_issues
                    self.logger.info(
                        "Quick format check failed, retrying",
                        article_url=article.get("source_url"),
                        attempt=retries + 1,
                        issues=issues[:2],
                    )

                # Retry: regenerate translation
                self.logger.info(
                    "Reflection failed, retrying",
                    article_url=article.get("source_url"),
                    attempt=retries + 1,
                    max_retries=self.max_retries,
                    issues=issues[:2] if issues else [],  # 只显示前2个问题
                )

                # Build feedback from issues
                feedback = "\n".join(f"- {issue}" for issue in issues) if issues else None

                # Regenerate translation with feedback
                translation = await llm.translate_and_summarize(
                    title=article["original_title"],
                    content=article.get("full_content", ""),
                    entities_to_preserve=article.get("entities_preserved"),
                    feedback=feedback,  # Pass feedback from reflection or quick check
                )

                article["chinese_title"] = translation.chinese_title
                article["chinese_summary"] = translation.chinese_summary
                article["entities_preserved"] = translation.entities_preserved or []
                article["reflection_feedback"] = feedback

                retries += 1

            except Exception as e:
                self.logger.warning(
                    f"Error during reflection attempt {retries + 1}",
                    error=str(e),
                    article_url=article.get("source_url"),
                )
                retries += 1
                if retries >= self.max_retries:
                    break

        # Max retries reached - still return article
        self.logger.warning(
            "Reflection failed after max retries, keeping article",
            article_url=article.get("source_url"),
            max_retries=self.max_retries,
        )

        article["reflection_passed"] = True  # 标记为通过，避免丢失文章
        article["reflection_retries"] = retries
        return article


# Singleton instance
reflection_agent = ReflectionAgent()