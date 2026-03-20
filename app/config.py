"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator"

    # LLM API Configuration
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None  # For custom API endpoints (DashScope, etc.)
    openai_embedding_model: str = ""
    openai_chat_model: str = ""

    # === LLM Settings ===
    llm_temperature: float = 0.3
    llm_enable_thinking: bool = False
    llm_max_tokens: int = 4096

    # === Scoring Weights ===
    scoring_weight_industry_impact: float = 0.4
    scoring_weight_milestone: float = 0.35
    scoring_weight_attention: float = 0.25

    # === Embedding Settings ===
    embedding_provider: str = "openai"  # "openai" | "dashscope" | "custom"
    embedding_max_batch_size: int = 10  # For DashScope batching

    # === Reranker Settings ===
    rerank_provider: str = "none"  # "none" | "dashscope" | "cohere" | "jina"
    rerank_model: str = "gte-rerank"
    rerank_api_url: Optional[str] = None  # Override default
    rerank_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY if not set

    # === DashScope-Specific ===
    dashscope_rerank_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    # Deduplication
    dedup_similarity_threshold: float = 0.85

    # Scoring
    score_threshold: float = 5.0

    # Reflection
    max_reflection_retries: int = 3  # Reduced from 7 after prompt optimization

    # Concurrency
    max_concurrent_scorers: int = 5
    max_concurrent_writers: int = 3
    max_concurrent_reflectors: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # RSS Feed Sources
    default_rss_feeds: List[str] = [
        # Tech News
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "https://www.techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.feedburner.com/oreilly/radar",
        "https://www.wired.com/feed/rss",
        "https://feeds.feedburner.com/TechCrunch/",
        "https://www.engadget.com/rss.xml",
        "https://gizmodo.com/rss",
        "https://venturebeat.com/feed/",
        # AI/ML News
        "https://www.artificialintelligence-news.com/feed/",
        "https://openai.com/blog/rss.xml",
        "https://deepmind.google/discover/blog/rss/",
        "https://huggingface.co/blog/feed.xml",
        # Developer News
        "https://github.blog/feed/",
        "https://stackoverflow.blog/feed/",
        "https://news.ycombinator.com/rss",
    ]

    # Deep Search Configuration
    deep_search_max_iterations: int = 5
    web_search_provider: str = "duckduckgo"  # or "tavily"
    web_search_api_key: Optional[str] = None

    # === Web Search Settings ===
    web_search_timeout: float = 30.0
    tavily_api_url: str = "https://api.tavily.com/search"
    web_search_min_interval: float = 2.0
    web_search_max_interval: float = 4.0

    # === Web Extractor Settings ===
    web_extractor_timeout: float = 30.0
    web_extractor_max_retries: int = 3
    web_extractor_retry_delay: float = 1.0
    web_extractor_host_limit: int = 2

    # RAG Configuration
    rag_chunk_size: int = 2000
    rag_chunk_overlap: int = 400
    rag_vector_weight: float = 0.6  # Weight for vector similarity in hybrid search
    rag_fts_weight: float = 0.4  # Weight for full-text search in hybrid search
    rag_rerank_model: str = "gte-rerank"  # DashScope rerank model
    rag_rerank_top_k: int = 10  # Number of candidates to retrieve before reranking
    rag_final_top_k: int = 5  # Number of final results after reranking

    # LangSmith Tracing
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None
    langsmith_project: str = "default"

    # DeepGraph Configuration
    deepgraph_enabled: bool = True
    deepgraph_builder_enabled: bool = True  # Run graph builder after article storage
    deepgraph_max_hops: int = 2
    deepgraph_expansion_limit: int = 50
    deepgraph_entity_similarity_threshold: float = 0.85

    # === CORS Settings (comma-separated strings) ===
    cors_origins: str = "http://localhost:3000"  # Safe default for local development
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_allow_methods_list(self) -> List[str]:
        """Parse CORS methods as a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]

    @property
    def cors_allow_headers_list(self) -> List[str]:
        """Parse CORS headers as a list."""
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []

        # Required
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.openai_chat_model:
            errors.append("OPENAI_CHAT_MODEL is required (e.g., gpt-4o-mini, qwen3.5-35b-a3b)")
        if not self.openai_embedding_model:
            errors.append("OPENAI_EMBEDDING_MODEL is required (e.g., text-embedding-3-small, text-embedding-v4)")

        # Validate weights sum to ~1.0
        weight_sum = self.scoring_weight_industry_impact + self.scoring_weight_milestone + self.scoring_weight_attention
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Scoring weights should sum to 1.0, got {weight_sum:.2f}")

        return errors


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()