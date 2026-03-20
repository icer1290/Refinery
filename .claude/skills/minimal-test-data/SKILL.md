---
name: minimal-test-data
description: When testing with real LLM calls, always use minimal data to reduce token usage
invocable: false
---

# Minimal Test Data Skill

When running tests or experiments that involve real LLM API calls, always minimize token usage by using the smallest possible test data:

## Rules

1. **RSS Feeds**: Test with a single RSS feed, not the full list of 17+ sources
2. **Articles**: Process only 1-2 articles at a time for testing, not the full batch
3. **Content**: Use truncated or minimal article content when testing extraction/translation
4. **GraphRAG**: Test with single articles before running on larger datasets

## Example Usage

When testing workflow:
```python
# Instead of testing all feeds
feeds = ["https://example.com/single-feed.xml"]

# Instead of processing all articles
articles = articles[:1]  # Just one article
```

## Why

- Reduces API costs during development
- Speeds up iteration cycles
- Makes debugging easier with smaller data scope
- Prevents runaway token usage in experimental code

## When to Apply

- Manual testing and debugging
- Development of new features
- Prototyping new agents or workflows
- Any code that calls LLM APIs for testing purposes

**Note**: For production CI/CD tests, use mocks/stubs instead of real LLM calls.