"""Redis service for caching and session management."""

import json
from typing import Any, Optional

import redis.asyncio as redis

from app.config import get_settings
from app.core import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RedisService:
    """Redis service for caching and session management.

    Provides async Redis operations for caching chat sessions,
    memories, and article contexts.
    """

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        """Initialize Redis connection if configured."""
        if settings.redis_url and settings.redis_enabled:
            try:
                self.client = redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                self._connected = True
                logger.info("Redis connection initialized", redis_url=settings.redis_url)
            except Exception as e:
                logger.warning("Failed to initialize Redis connection", error=str(e))
                self._connected = False
        else:
            logger.debug("Redis not configured, using in-memory fallback")

    async def ping(self) -> bool:
        """Check if Redis is available."""
        if not self.client:
            return False
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Value string or None if not found
        """
        if not self.client:
            return None
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.warning("Redis get failed", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: int = 3600,
    ) -> bool:
        """Set value with TTL.

        Args:
            key: Redis key
            value: Value string
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        try:
            return await self.client.setex(key, ttl, value)
        except Exception as e:
            logger.warning("Redis set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis.

        Args:
            key: Redis key

        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            return False
        try:
            return await self.client.delete(key) > 0
        except Exception as e:
            logger.warning("Redis delete failed", key=key, error=str(e))
            return False

    async def get_json(self, key: str) -> Optional[dict[str, Any]]:
        """Get JSON value from Redis.

        Args:
            key: Redis key

        Returns:
            Parsed JSON dict or None
        """
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.warning("Failed to decode JSON from Redis", key=key, error=str(e))
                return None
        return None

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        """Set JSON value in Redis.

        Args:
            key: Redis key
            value: Dict to store as JSON
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        return await self.set(key, json.dumps(value), ttl)

    async def incr(self, key: str) -> int:
        """Increment counter in Redis.

        Args:
            key: Redis key

        Returns:
            New value after increment
        """
        if not self.client:
            return 0
        try:
            return await self.client.incr(key)
        except Exception as e:
            logger.warning("Redis incr failed", key=key, error=str(e))
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key.

        Args:
            key: Redis key
            ttl: New TTL in seconds

        Returns:
            True if successful
        """
        if not self.client:
            return False
        try:
            return await self.client.expire(key, ttl)
        except Exception as e:
            logger.warning("Redis expire failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Redis key

        Returns:
            True if key exists
        """
        if not self.client:
            return False
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.warning("Redis exists failed", key=key, error=str(e))
            return False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected


# Singleton instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Get Redis service instance.

    Returns:
        RedisService singleton
    """
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service