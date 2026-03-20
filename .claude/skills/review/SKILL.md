---
name: review
description: Code review for async patterns, error handling, type hints, and config consistency
invocable: true
---

# Code Review

Review code for the following issues:

## 1. Async/Await Patterns

- **Missing awaits**: Functions called without `await` will run synchronously and may cause issues
- **Blocking calls in async functions**: Avoid `time.sleep()`, use `asyncio.sleep()` instead
- **Mixed sync/async**: Don't call async functions from sync context without proper handling

```python
# Bad
async def fetch_data():
    result = some_async_function()  # Missing await!

# Good
async def fetch_data():
    result = await some_async_function()
```

## 2. Error Handling Completeness

- **Try/except coverage**: Ensure async operations that can fail have proper exception handling
- **Specific exceptions**: Catch specific exceptions rather than bare `except:`
- **Logging errors**: Log errors with context for debugging
- **Graceful degradation**: Handle failures without crashing the entire workflow

```python
# Bad
try:
    await fetch_url(url)
except:
    pass  # Swallowing all errors silently

# Good
try:
    await fetch_url(url)
except httpx.TimeoutException:
    logger.warning(f"Timeout fetching {url}")
except httpx.HTTPStatusError as e:
    logger.error(f"HTTP error {e.response.status_code} for {url}")
```

## 3. Type Hints in Python

- **Function signatures**: All functions should have parameter and return type hints
- **Optional types**: Use `Optional[T]` or `T | None` for nullable values
- **Collection types**: Use `list[T]`, `dict[str, T]` instead of bare `list`, `dict`
- **Avoid `Any`**: Use specific types or `Union` when possible

```python
# Bad
def process_items(items, config):
    ...

# Good
def process_items(items: list[dict[str, Any]], config: Config) -> list[Article]:
    ...
```

## 4. Configuration Consistency

- **Docker Compose**: Environment variables should match `docker-compose.yml`
- **Env files**: Check `.env.example` has all required variables
- **Config class**: `app/config.py` should reflect all environment variables
- **Default values**: Ensure defaults are consistent across all config sources
- **Required vs Optional**: Mark truly required variables as such in config

Check alignment between:
- `docker-compose.yml` environment section
- `.env.example` file
- `app/config.py` Pydantic settings
- Documentation in `CLAUDE.md` or `README.md`

## Review Checklist

When reviewing PRs or code changes:

- [ ] All async functions properly use `await` for coroutines
- [ ] Error handling covers expected failure modes
- [ ] Type hints present on all public functions
- [ ] Config changes reflected in all relevant files