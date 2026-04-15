# AI-Engine CLI User Guide

AI-powered tech news aggregation service with LangGraph.

## Installation

```bash
pip install ai-engine
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Initialize configuration
ai-engine init

# 2. Start Docker services (PostgreSQL, Redis)
ai-engine services start

# 3. Run news aggregation workflow
ai-engine workflow run

# 4. View collected articles
ai-engine article list
```

## Configuration

### Setup Wizard

Run the interactive setup wizard:

```bash
ai-engine init
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
| 3           | Local config          | `./.ai-engine.toml`        |
| 4           | Global config         | `~/.ai-engine/config.toml` |
| 5 (lowest)  | Defaults              | Built-in values            |

### Global Config Location

```
~/.ai-engine/config.toml
```

### Example Config

```toml
[llm]
api_key = "sk-xxx"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen3.5-35b-a3b"
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

Manage Docker services (PostgreSQL, Redis, ai-engine server).

```bash
# Start all services
ai-engine services start

# Start specific services
ai-engine services start postgres redis

# Start without waiting for health checks
ai-engine services start --no-wait

# View service status
ai-engine services status

# View logs
ai-engine services logs
ai-engine services logs postgres -f        # Follow logs
ai-engine services logs ai-engine --tail=50

# Stop services
ai-engine services stop

# Stop and remove volumes (deletes data)
ai-engine services stop -v

# Check if running (useful for scripts)
ai-engine services is-running
ai-engine services is-running postgres
```

### Workflow Commands

Run and manage news aggregation workflows.

```bash
# Run workflow (default: last 24 hours)
ai-engine workflow run

# Run with specific feeds
ai-engine workflow run -f https://feeds.arstechnica.com/arstechnica/technology-lab

# Run with score threshold
ai-engine workflow run --threshold 6.0

# Force reprocess existing articles
ai-engine workflow run --force

# Look back more hours
ai-engine workflow run --hours 48

# JSON output (agent-friendly)
ai-engine workflow run --json

# List workflow run history
ai-engine workflow list
ai-engine workflow list --hours 72 --limit 10

# Show specific run details
ai-engine workflow show <run-id>
```

### Article Commands

Browse and view collected articles.

```bash
# List articles
ai-engine article list

# Pagination
ai-engine article list --page 2 --size 50

# Filter by minimum score
ai-engine article list --min-score 7.0

# Filter by source
ai-engine article list --source "TechCrunch"

# JSON output
ai-engine article list --json

# Show article details
ai-engine article show <article-id>
ai-engine article show <article-id> --json
```

### Deep Search Commands

Run comprehensive research on specific articles.

```bash
# Run deep search
ai-engine search run <article-id>

# Set max iterations
ai-engine search run <article-id> --iterations 10

# JSON output
ai-engine search run <article-id> --json

# Check if deep search was performed
ai-engine search status <article-id>
```

### Graph Commands

Build and analyze knowledge graphs (GraphRAG).

```bash
# Build knowledge graph from articles
ai-engine graph build <article-id-1> <article-id-2>

# Analyze graph
ai-engine graph analyze <article-id>

# Custom expansion settings
ai-engine graph analyze <article-id> --hops 3 --expansion 100

# JSON output
ai-engine graph build <article-id> --json
ai-engine graph analyze <article-id> --json
```

### Chat Commands

Interactive conversation about articles.

```bash
# Start interactive chat
ai-engine chat <article-id>

# Specify user ID
ai-engine chat <article-id> --user 42

# Exit commands: quit, exit, q
```

Example session:

```
$ ai-engine chat abc123

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
ai-engine article list --json
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

1. Ensure PostgreSQL is running: `ai-engine services start postgres`
2. Check DATABASE\_URL in config

### Configuration Errors

```
Error: LLM API key is required
```

Solution: Run `ai-engine init` to configure, or set `OPENAI_API_KEY` environment variable.

### No Articles Found

```
No articles found.
```

Solution: Run workflow first: `ai-engine workflow run`

## Advanced Usage

### Custom RSS Feeds

Add feeds to config or use `-f` option:

```bash
ai-engine workflow run \
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
0 * * * * /usr/local/bin/ai-engine workflow run --json >> /var/log/ai-engine.log
```

### Script Integration

```bash
#!/bin/bash

# Start services if not running
ai-engine services is-running || ai-engine services start

# Run workflow
RESULT=$(ai-engine workflow run --json)

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
    ["ai-engine", "workflow", "run", "--json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
print(f"Stored {data['total_articles_stored']} articles")
```

## Help

View command help:

```bash
ai-engine --help
ai-engine workflow --help
ai-engine workflow run --help
```

