"""Configuration manager for ai-engine CLI.

Handles multi-layer configuration with priority:
CLI args > env vars > local config > global config > defaults
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore


# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "api_key": "",
        "base_url": None,
        "chat_model": "",
        "embedding_model": "",
        "temperature": 0.3,
        "max_tokens": 4096,
        "enable_thinking": False,
    },
    "database": {
        "url": "postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator",
    },
    "services": {
        "redis_enabled": False,
        "redis_url": "redis://localhost:6379/0",
    },
    "web_search": {
        "provider": "duckduckgo",
        "api_key": None,
        "timeout": 30.0,
    },
    "rag": {
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "vector_weight": 0.6,
        "fts_weight": 0.4,
    },
    "scoring": {
        "weight_industry_impact": 0.4,
        "weight_milestone": 0.35,
        "weight_attention": 0.25,
        "score_threshold": 5.0,
    },
    "dedup": {
        "similarity_threshold": 0.85,
    },
}


class ConfigManager:
    """Manages ai-engine configuration with multi-layer priority.

    Configuration layers (highest to lowest priority):
    1. CLI arguments (passed at runtime)
    2. Environment variables
    3. Local config file (./.ai-engine.toml)
    4. Global config file (~/.ai-engine/config.toml)
    5. Default values
    """

    GLOBAL_CONFIG_DIR = Path.home() / ".ai-engine"
    GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.toml"
    LOCAL_CONFIG_FILE = Path(".ai-engine.toml")

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self._config: dict[str, Any] | None = None

    @property
    def global_config_path(self) -> Path:
        """Get the global config file path."""
        return self.GLOBAL_CONFIG_FILE

    @property
    def local_config_path(self) -> Path:
        """Get the local config file path."""
        return self.LOCAL_CONFIG_FILE

    def _read_toml_file(self, path: Path) -> dict[str, Any]:
        """Read a TOML file if it exists.

        Args:
            path: Path to the TOML file.

        Returns:
            Parsed TOML content or empty dict if file doesn't exist.
        """
        if not path.exists():
            return {}
        try:
            content = path.read_text(encoding="utf-8")
            return tomllib.loads(content)
        except Exception:
            return {}

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary to merge into.
            override: Dictionary with override values.

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _get_env_overrides(self) -> dict[str, Any]:
        """Get configuration overrides from environment variables.

        Returns:
            Dictionary of configuration from environment variables.
        """
        env_overrides: dict[str, Any] = {}

        # LLM configuration
        if api_key := os.environ.get("OPENAI_API_KEY"):
            env_overrides.setdefault("llm", {})["api_key"] = api_key
        if base_url := os.environ.get("OPENAI_BASE_URL"):
            env_overrides.setdefault("llm", {})["base_url"] = base_url
        if chat_model := os.environ.get("OPENAI_CHAT_MODEL"):
            env_overrides.setdefault("llm", {})["chat_model"] = chat_model
        if embedding_model := os.environ.get("OPENAI_EMBEDDING_MODEL"):
            env_overrides.setdefault("llm", {})["embedding_model"] = embedding_model
        if temp := os.environ.get("LLM_TEMPERATURE"):
            try:
                env_overrides.setdefault("llm", {})["temperature"] = float(temp)
            except ValueError:
                pass
        if max_tokens := os.environ.get("LLM_MAX_TOKENS"):
            try:
                env_overrides.setdefault("llm", {})["max_tokens"] = int(max_tokens)
            except ValueError:
                pass
        if thinking := os.environ.get("LLM_ENABLE_THINKING"):
            env_overrides.setdefault("llm", {})["enable_thinking"] = thinking.lower() in ("true", "1", "yes")

        # Database configuration
        if db_url := os.environ.get("DATABASE_URL"):
            env_overrides.setdefault("database", {})["url"] = db_url

        # Redis configuration
        if redis_enabled := os.environ.get("REDIS_ENABLED"):
            env_overrides.setdefault("services", {})["redis_enabled"] = redis_enabled.lower() in ("true", "1", "yes")
        if redis_url := os.environ.get("REDIS_URL"):
            env_overrides.setdefault("services", {})["redis_url"] = redis_url

        # Web search configuration
        if provider := os.environ.get("WEB_SEARCH_PROVIDER"):
            env_overrides.setdefault("web_search", {})["provider"] = provider
        if api_key := os.environ.get("WEB_SEARCH_API_KEY"):
            env_overrides.setdefault("web_search", {})["api_key"] = api_key

        # RAG configuration
        if chunk_size := os.environ.get("RAG_CHUNK_SIZE"):
            try:
                env_overrides.setdefault("rag", {})["chunk_size"] = int(chunk_size)
            except ValueError:
                pass
        if chunk_overlap := os.environ.get("RAG_CHUNK_OVERLAP"):
            try:
                env_overrides.setdefault("rag", {})["chunk_overlap"] = int(chunk_overlap)
            except ValueError:
                pass

        # Scoring configuration
        if threshold := os.environ.get("SCORE_THRESHOLD"):
            try:
                env_overrides.setdefault("scoring", {})["score_threshold"] = float(threshold)
            except ValueError:
                pass

        # Dedup configuration
        if threshold := os.environ.get("DEDUP_SIMILARITY_THRESHOLD"):
            try:
                env_overrides.setdefault("dedup", {})["similarity_threshold"] = float(threshold)
            except ValueError:
                pass

        return env_overrides

    def load(self, use_env: bool = True) -> dict[str, Any]:
        """Load configuration from all layers.

        Args:
            use_env: Whether to include environment variable overrides.

        Returns:
            Merged configuration dictionary.
        """
        if self._config is not None:
            return self._config

        # Start with defaults
        config = DEFAULT_CONFIG.copy()

        # Merge global config
        global_config = self._read_toml_file(self.GLOBAL_CONFIG_FILE)
        if global_config:
            config = self._deep_merge(config, global_config)

        # Merge local config (higher priority)
        local_config = self._read_toml_file(self.LOCAL_CONFIG_FILE)
        if local_config:
            config = self._deep_merge(config, local_config)

        # Merge environment variables (highest priority)
        if use_env:
            env_overrides = self._get_env_overrides()
            if env_overrides:
                config = self._deep_merge(config, env_overrides)

        self._config = config
        return config

    def save(
        self,
        config: dict[str, Any],
        global_config: bool = True,
        create_dir: bool = True,
    ) -> Path:
        """Save configuration to a TOML file.

        Args:
            config: Configuration dictionary to save.
            global_config: If True, save to global config; otherwise local config.
            create_dir: Create parent directory if it doesn't exist.

        Returns:
            Path to the saved config file.

        Raises:
            ImportError: If tomli_w is not installed.
            OSError: If the file cannot be written.
        """
        if tomli_w is None:
            raise ImportError(
                "tomli_w is required for writing TOML files. "
                "Install it with: pip install tomli-w"
            )

        target_path = self.GLOBAL_CONFIG_FILE if global_config else self.LOCAL_CONFIG_FILE

        if create_dir and global_config:
            self.GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        content = tomli_w.dumps(config)
        target_path.write_text(content, encoding="utf-8")

        # Clear cached config
        self._config = None

        return target_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Supports dot notation for nested keys (e.g., "llm.api_key").

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        config = self.load()

        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(
        self,
        key: str,
        value: Any,
        global_config: bool = True,
    ) -> None:
        """Set a configuration value.

        Supports dot notation for nested keys (e.g., "llm.api_key").
        Saves the entire config after setting the value.

        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
            global_config: If True, save to global config; otherwise local config.
        """
        config = self.load(use_env=False)

        keys = key.split(".")
        current = config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

        # Save the config
        self.save(config, global_config=global_config)

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation error messages. Empty if valid.
        """
        config = self.load()
        errors: list[str] = []

        # Required LLM settings
        llm = config.get("llm", {})
        if not llm.get("api_key"):
            errors.append("LLM API key is required (llm.api_key)")
        if not llm.get("chat_model"):
            errors.append("Chat model is required (llm.chat_model)")
        if not llm.get("embedding_model"):
            errors.append("Embedding model is required (llm.embedding_model)")

        # Validate scoring weights sum to ~1.0
        scoring = config.get("scoring", {})
        weight_keys = ["weight_industry_impact", "weight_milestone", "weight_attention"]
        if all(k in scoring for k in weight_keys):
            weight_sum = sum(scoring[k] for k in weight_keys)
            if abs(weight_sum - 1.0) > 0.01:
                errors.append(f"Scoring weights should sum to 1.0, got {weight_sum:.2f}")

        return errors

    def exists(self, global_config: bool = True) -> bool:
        """Check if a config file exists.

        Args:
            global_config: If True, check global config; otherwise local config.

        Returns:
            True if the config file exists.
        """
        path = self.GLOBAL_CONFIG_FILE if global_config else self.LOCAL_CONFIG_FILE
        return path.exists()

    def get_config_path(self, global_config: bool = True) -> Path:
        """Get the path to a config file.

        Args:
            global_config: If True, return global config path; otherwise local.

        Returns:
            Path to the config file.
        """
        return self.GLOBAL_CONFIG_FILE if global_config else self.LOCAL_CONFIG_FILE

    def clear_cache(self) -> None:
        """Clear the cached configuration."""
        self._config = None


# Global instance for convenience
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance.

    Returns:
        ConfigManager singleton instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager