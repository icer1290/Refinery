[English](README.md) | [中文](README.zh-CN.md)

# AI-Engine

[![PyPI version](https://badge.fury.io/py/ai-engine.svg)](https://badge.fury.io/py/ai-engine)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered tech news aggregation system built with LangGraph, featuring CLI-first design for seamless agent integration.

## Features

- **RSS Feed Aggregation**: Automatically fetches news from 17+ tech news sources
- **Semantic Deduplication**: Uses vector embeddings to identify and remove duplicate articles
- **Multi-dimensional Scoring**: AI-powered scoring based on industry impact, milestone significance, and attention value
- **Content Extraction**: Extracts full article content using trafilatura
- **Chinese Translation**: Generates Chinese titles and summaries with entity preservation
- **Self-reflection**: Validates translation quality with automatic retry mechanism
- **Deep Search**: On-demand article research via ReAct loop with web search (DuckDuckGo/Tavily)
- **GraphRAG**: Knowledge graph construction with community detection for contextual analysis
- **Multi-turn Chat**: Conversational AI with multi-agent architecture, ReAct loop, and multi-layer memory
- **Vector Storage**: PostgreSQL with pgvector extension for vector storage and graph data
- **CLI Interface**: Full command-line interface with JSON output for agent automation

## Installation

### From PyPI (Recommended)

```bash
pip install ai-engine
```

### From GitHub

```bash
# Latest version
pip install git+https://github.com/your-org/ai-engine.git

# Specific version/tag
pip install git+https://github.com/your-org/ai-engine.git@v1.0.0
```

### From Source (Development)

```bash
git clone https://github.com/your-org/ai-engine.git
cd ai-engine

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for PostgreSQL and Redis)
- LLM API key (OpenAI, DashScope, Azure, or compatible)

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

## CLI Commands

### Services Management

```bash
# Start all services
ai-engine services start

# Start specific services
ai-engine services start postgres redis

# View service status
ai-engine services status

# View logs
ai-engine services logs postgres -f

# Stop services
ai-engine services stop

# Check if running (useful for scripts)
ai-engine services is-running
```

### Workflow Commands

```bash
# Run workflow (default: last 24 hours)
ai-engine workflow run

# Run with specific feeds
ai-engine workflow run -f https://feeds.arstechnica.com/arstechnica/technology-lab

# Run with score threshold
ai-engine workflow run --threshold 6.0

# Force reprocess existing articles
ai-engine workflow run --force

# JSON output (agent-friendly)
ai-engine workflow run --json

# List workflow run history
ai-engine workflow list

# Show specific run details
ai-engine workflow show <run-id>
```

### Article Commands

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
```

### Deep Search Commands

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

```bash
# Build knowledge graph
ai-engine graph build <article-id-1> <article-id-2>

# Analyze graph
ai-engine graph analyze <article-id>

# Custom expansion settings
ai-engine graph analyze <article-id> --hops 3 --expansion 100
```

### Chat Commands

```bash
# Start interactive chat
ai-engine chat <article-id>

# Specify user ID
ai-engine chat <article-id> --user 42
```

## JSON Output

All commands support `--json` flag for structured output, useful for scripting and agent integration:

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

## Configuration

### Configuration Priority

| Priority | Location | Description |
|----------|----------|-------------|
| 1 (highest) | CLI arguments | `--option` flags |
| 2 | Environment variables | `OPENAI_API_KEY`, etc. |
| 3 | Local config | `./.ai-engine.toml` |
| 4 | Global config | `~/.ai-engine/config.toml` |
| 5 (lowest) | Defaults | Built-in values |

### Setup Wizard

Run the interactive setup wizard:

```bash
ai-engine init
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `OPENAI_API_KEY` | Yes | OpenAI or compatible API key |
| `OPENAI_CHAT_MODEL` | Yes | Chat model (e.g., gpt-4o-mini, qwen3.5-35b-a3b) |
| `OPENAI_EMBEDDING_MODEL` | Yes | Embedding model |
| `OPENAI_BASE_URL` | No | Override API base URL (for DashScope, Azure, Ollama) |
| `WEB_SEARCH_PROVIDER` | No | `duckduckgo` or `tavily` (default: duckduckgo) |
| `REDIS_URL` | No | Redis connection URL |
| `REDIS_ENABLED` | No | Enable Redis caching (default: true) |

See `.env.example` for complete configuration options.

## Architecture

### Main Workflow Pipeline

```
[Entry] → [Scout] → [Dedup] → [Scoring] → [Writing] → [Reflection] → [Storage] → [End]
```

- **Scout**: Fetch RSS feeds, extract articles
- **Dedup**: Vector similarity-based deduplication
- **Scoring**: Multi-dimensional AI scoring (industry impact, milestone, attention)
- **Writing**: Content extraction + Chinese translation
- **Reflection**: Translation quality validation with retry
- **Storage**: Persist to PostgreSQL with vectors

### Deep Search (ReAct Loop)

On-demand deep research for articles:
1. Fetch article content
2. ReAct loop with web search tools
3. Generate comprehensive report stored in `deepsearch_report` field

### GraphRAG (DeepGraph)

Two-phase knowledge graph system:
- **Background GraphBuilder**: Extract entities/relationships, detect communities (Leiden algorithm)
- **On-demand GraphAnalyst**: Fetch subgraph, expand via traversal, generate analysis

### Multi-turn Chat (Hub-and-Spoke Architecture)

Conversational AI with multi-agent coordination:
- **Supervisor**: Central hub that evaluates intent and routes to specialist agents
- **Researcher**: ReAct loop agent for deep information gathering
- **Explainer**: Provides article explanations and context
- **Fact Checker**: Validates claims against knowledge graph and web sources
- **Multi-layer Memory**: Short-term, mid-term, and long-term memory

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Database**: PostgreSQL + pgvector
- **Cache**: Redis (optional)
- **AI**: OpenAI or compatible APIs
- **RSS/Web**: feedparser, trafilatura, duckduckgo-search
- **CLI**: Typer, Rich

## API Endpoints

All endpoints under `/api/v1/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/workflow/run` | POST | Trigger news aggregation workflow |
| `/workflow/runs` | GET | List workflow run history |
| `/workflow/articles` | GET | List articles |
| `/workflow/articles/{id}` | GET | Get article details |
| `/deep-search/run` | POST | Run deep search for an article |
| `/deep-graph/analyze` | POST | Generate knowledge graph analysis |
| `/chat/chat` | POST | Send message and get AI response |

Access API documentation at: http://localhost:8000/docs

## Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ai-engine

# Stop services
docker-compose down
```

## Testing

```bash
pytest
pytest --cov=app tests/
pytest tests/test_rag.py -v
```

## Project Structure

```
ai-engine/
├── app/
│   ├── cli/           # CLI commands and utilities
│   ├── api/           # FastAPI routes
│   ├── models/        # Database models
│   ├── agents/        # LangGraph agents
│   ├── workflow/      # Main workflow graph
│   ├── deep_search/   # ReAct loop for deep research
│   ├── deep_graph/    # GraphRAG builder and analyst
│   ├── chat/          # Multi-turn chat system
│   ├── services/      # RAG services
│   └── prompts/       # Centralized prompts
├── alembic/           # Database migrations
├── tests/             # Test files
├── pyproject.toml
└── docker-compose.yml
```

## Integration Examples

### Cron Job

```bash
# Run workflow every hour
0 * * * * /usr/local/bin/ai-engine workflow run --json >> /var/log/ai-engine.log
```

### Python Integration

```python
import subprocess
import json

result = subprocess.run(
    ["ai-engine", "workflow", "run", "--json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
print(f"Stored {data['total_articles_stored']} articles")
```

## Help

```bash
ai-engine --help
ai-engine workflow --help
ai-engine article --help
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Issues

If you encounter any problems, please file an issue at: https://github.com/your-org/ai-engine/issues