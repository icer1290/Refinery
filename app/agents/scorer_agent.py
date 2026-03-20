"""Scorer Agent - Article scoring and evaluation."""

import asyncio
import traceback
from typing import Any

from app.agents.base import BaseAgent
from app.config import get_settings
from app.services.llm_service import LLMService, get_llm_service

settings = get_settings()


class ScorerAgent(BaseAgent):
    """Agent responsible for scoring articles."""

    def __init__(
        self,
        score_threshold: float | None = None,
        max_concurrent: int | None = None,
    ):
        super().__init__("Scorer")
        self.llm_service: LLMService | None = None
        self.score_threshold = score_threshold or settings.score_threshold
        self.max_concurrent = max_concurrent or settings.max_concurrent_scorers
        self._semaphore: asyncio.Semaphore | None = None

    def _get_llm_service(self) -> LLMService:
        """Lazily initialize LLM service."""
        if self.llm_service is None:
            self.llm_service = get_llm_service()
        return self.llm_service

    async def execute(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score articles and filter by threshold.

        Args:
            articles: List of articles to score

        Returns:
            List of articles with scores, filtered by threshold
        """
        if not articles:
            return []

        # 初始化 LLM 服务
        try:
            llm = self._get_llm_service()
            self.logger.info("LLM service initialized for scoring", model=llm.model)
        except Exception as e:
            self.logger.error("Failed to initialize LLM service", error=str(e))
            return []

        # Initialize semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        self.logger.info(
            "Starting to score articles",
            total_articles=len(articles),
            max_concurrent=self.max_concurrent,
        )

        # Score articles concurrently
        tasks = [self._score_article(article, i) for i, article in enumerate(articles)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and collect results
        scored_articles = []
        errors_count = 0
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                errors_count += 1
                self.logger.warning(
                    "Failed to score article",
                    article_url=article.get("source_url"),
                    error=str(result),
                )
                continue

            if result["total_score"] >= self.score_threshold:
                scored_articles.append(result)
            else:
                self.logger.debug(
                    "Article filtered by score threshold",
                    article_url=article.get("source_url"),
                    total_score=result["total_score"],
                    threshold=self.score_threshold,
                )

        self.logger.info(
            "Scorer phase complete",
            articles_input=len(articles),
            articles_passed=len(scored_articles),
            errors_count=errors_count,
            threshold=self.score_threshold,
        )

        return scored_articles

    async def _score_article(self, article: dict[str, Any], index: int) -> dict[str, Any]:
        """Score a single article.

        Args:
            article: Article to score
            index: Article index for logging

        Returns:
            Article with scoring fields added
        """
        async with self._semaphore:
            llm = self._get_llm_service()

            self.logger.debug(
                f"Scoring article {index + 1}",
                title=article["original_title"][:50],
            )

            try:
                result = await asyncio.wait_for(
                    llm.score_article(
                        title=article["original_title"],
                        description=article.get("original_description", ""),
                        content=article.get("full_content"),
                    ),
                    timeout=60.0,  # 60秒超时
                )

                # Update article with scores
                scored_article = article.copy()
                scored_article.update({
                    "industry_impact_score": result.industry_impact_score,
                    "milestone_score": result.milestone_score,
                    "attention_score": result.attention_score,
                    "total_score": result.total_score,
                    "scoring_reasoning": result.reasoning,
                })

                self.logger.debug(
                    f"Scored article {index + 1}",
                    total_score=result.total_score,
                )

                return scored_article

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout scoring article {index + 1}",
                    article_url=article.get("source_url"),
                )
                raise Exception(f"Timeout scoring article {index + 1}")
            except Exception as e:
                self.logger.error(
                    f"Error scoring article {index + 1}",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                raise


# Singleton instance
scorer_agent = ScorerAgent()