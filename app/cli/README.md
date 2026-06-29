# AI-Engine CLI User Guide

AI-powered tech news aggregation service with LangGraph.

## Installation

```bash
pip install refinery
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Initialize configuration
refinery init

# 2. Start Docker services (PostgreSQL, Redis)
refinery services start

# 3. Run news aggregation workflow
refinery workflow run

# 4. View collected articles
refinery article list
```

## Configuration

### Setup Wizard

Run the interactive setup wizard:

```bash
refinery init
```

This will guide you through:

- LLM API configuration (API key, models, base URL)
- Database connection
- Redis settings
- Web search provider
- Advanced RAG and scoring parameters

### Configuration Files

Configuration is stored in TOML format with multi-layer priority:

| Priority    | Location              | Description                |
| ----------- | --------------------- | -------------------------- |
| 1 (highest) | CLI arguments         | `--option` flags           |
| 2           | Environment variables | `OPENAI_API_KEY`, etc.     |
| 3           | Local config          | `./.refinery.toml`        |
| 4           | Global config         | `~/.refinery/config.toml` |
| 5 (lowest)  | Defaults              | Built-in values            |

### Global Config Location

```
~/.refinery/config.toml
```

### Example Config

```toml
[llm]
api_key = "sk-xxx"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen3.6-35b-a3b"
embedding_model = "text-embedding-v4"
temperature = 0.3

[database]
url = "postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator"

[services]
redis_enabled = true
redis_url = "redis://localhost:6379/0"

[web_search]
provider = "duckduckgo"

[rag]
chunk_size = 2000
chunk_overlap = 400

[scoring]
score_threshold = 5.0
weight_industry_impact = 0.4
weight_milestone = 0.35
weight_attention = 0.25

[dedup]
similarity_threshold = 0.85
```

### Environment Variables

| Variable                 | Description                               |
| ------------------------ | ----------------------------------------- |
| `OPENAI_API_KEY`         | LLM API key                               |
| `OPENAI_BASE_URL`        | API base URL (for DashScope, Azure, etc.) |
| `OPENAI_CHAT_MODEL`      | Chat model name                           |
| `OPENAI_EMBEDDING_MODEL` | Embedding model name                      |
| `DATABASE_URL`           | PostgreSQL connection URL                 |
| `REDIS_URL`              | Redis connection URL                      |
| `REDIS_ENABLED`          | Enable Redis caching                      |
| `WEB_SEARCH_PROVIDER`    | `duckduckgo` or `tavily`                  |

## Commands

### Services Management

Manage Docker services (PostgreSQL, Redis, refinery server).

```bash
# Start all services
refinery services start

# Start specific services
refinery services start postgres redis

# Start without waiting for health checks
refinery services start --no-wait

# View service status
refinery services status

# View logs
refinery services logs
refinery services logs postgres -f        # Follow logs
refinery services logs refinery --tail=50

# Stop services
refinery services stop

# Stop and remove volumes (deletes data)
refinery services stop -v

# Check if running (useful for scripts)
refinery services is-running
refinery services is-running postgres
```

### Workflow Commands

Run and manage news aggregation workflows.

```bash
# Run workflow (default: last 24 hours)
refinery workflow run

# Run with specific feeds
refinery workflow run -f https://feeds.arstechnica.com/arstechnica/technology-lab

# Run with score threshold
refinery workflow run --threshold 6.0

# Force reprocess existing articles
refinery workflow run --force

# Look back more hours
refinery workflow run --hours 48

# JSON output (agent-friendly)
refinery workflow run --json

# List workflow run history
refinery workflow list
refinery workflow list --hours 72 --limit 10

# Show specific run details
refinery workflow show <run-id>
```

### Article Commands

Browse and view collected articles.

```bash
# List articles
refinery article list

# Pagination
refinery article list --page 2 --size 50

# Filter by minimum score
refinery article list --min-score 7.0

# Filter by source
refinery article list --source "TechCrunch"

# JSON output
refinery article list --json

# Show article details
refinery article show <article-id>
refinery article show <article-id> --json
```

### Deep Search Commands

Run comprehensive research on specific articles.

```bash
# Run deep search
refinery search run <article-id>

# Set max iterations
refinery search run <article-id> --iterations 10

# JSON output
refinery search run <article-id> --json

# Check if deep search was performed
refinery search status <article-id>
```

### Graph Commands

Build and analyze knowledge graphs (GraphRAG).

```bash
# Build knowledge graph from articles
refinery graph build <article-id-1> <article-id-2>

# Analyze graph
refinery graph analyze <article-id>

# Custom expansion settings
refinery graph analyze <article-id> --hops 3 --expansion 100

# JSON output
refinery graph build <article-id> --json
refinery graph analyze <article-id> --json
```

### Chat Commands

Interactive conversation about articles.

```bash
# Start interactive chat
refinery chat <article-id>

# Specify user ID
refinery chat <article-id> --user 42

# Exit commands: quit, exit, q
```

Example session:

```
$ refinery chat abc123

╭─────────────────────────────────────╮
│ Article Context                     │
│ Title: AI Breakthrough in NLP       │
│ Source: TechCrunch                  │
│ DeepSearch: Available               │
╰─────────────────────────────────────╯

Starting interactive chat...
Type 'quit', 'exit', or 'q' to end the session

You: What are the key findings?

╭─────────────────────────────────────╮
│ AI Response                         │
│ The article discusses three major   │
│ breakthroughs...                    │
╰─────────────────────────────────────╯

You: quit
Ending chat session...
```

## JSON Output

All commands support `--json` flag for structured output, useful for:

- Scripting and automation
- Agent/LLM integration
- API-like usage

Example:

```bash
refinery article list --json
```

Output:

```json
{
  "articles": [
    {
      "id": "abc123",
      "source_name": "TechCrunch",
      "chinese_title": "AI技术突破",
      "total_score": 8.5,
      "published_at": "2024-01-15 10:30"
    }
  ],
  "total": 42,
  "page": 1,
  "size": 20
}
```

## Exit Codes

| Code | Meaning          |
| ---- | ---------------- |
| 0    | Success          |
| 1    | Error / Failure  |
| 2    | Validation error |

## Troubleshooting

### Docker Not Available

```
Error: Docker is not installed or not in PATH
```

Solution: Install Docker and docker-compose.

### Database Connection Failed

```
Error: Could not connect to database
```

Solution:

1. Ensure PostgreSQL is running: `refinery services start postgres`
2. Check DATABASE\_URL in config

### Configuration Errors

```
Error: LLM API key is required
```

Solution: Run `refinery init` to configure, or set `OPENAI_API_KEY` environment variable.

### No Articles Found

```
No articles found.
```

Solution: Run workflow first: `refinery workflow run`

## Advanced Usage

### Custom RSS Feeds

Add feeds to config or use `-f` option:

```bash
refinery workflow run \
  -f https://feeds.arstechnica.com/arstechnica/technology-lab \
  -f https://www.techcrunch.com/feed/
```

### Scoring Weights

Adjust in config:

```toml
[scoring]
weight_industry_impact = 0.5   # Industry relevance
weight_milestone = 0.3         # Major breakthroughs
weight_attention = 0.2         # Public attention
```

### Web Search Provider

DuckDuckGo (free):

```toml
[web_search]
provider = "duckduckgo"
```

Tavily (better results, requires API key):

```toml
[web_search]
provider = "tavily"
api_key = "tvly-xxx"
```

### Redis Caching

Enable for session caching:

```toml
[services]
redis_enabled = true
redis_url = "redis://localhost:6379/0"
```

## Integration Examples

### Cron Job

Run workflow every hour:

```bash
# Add to crontab
0 * * * * /usr/local/bin/refinery workflow run --json >> /var/log/refinery.log
```

### Script Integration

```bash
#!/bin/bash

# Start services if not running
refinery services is-running || refinery services start

# Run workflow
RESULT=$(refinery workflow run --json)

# Extract article count
COUNT=$(echo "$RESULT" | jq '.total_articles_stored')

echo "Collected $COUNT articles"
```

### Python Integration

```python
import subprocess
import json

# Run workflow
result = subprocess.run(
    ["refinery", "workflow", "run", "--json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
print(f"Stored {data['total_articles_stored']} articles")
```

## Help

View command help:

```bash
refinery --help
refinery workflow --help
refinery workflow run --help
```

