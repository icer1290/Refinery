# Tech News Aggregator

An AI-powered tech news aggregation system built with LangGraph, FastAPI, and PostgreSQL.

## Features

- **RSS Feed Aggregation**: Automatically fetches news from 17+ tech news sources
- **Semantic Deduplication**: Uses vector embeddings to identify and remove duplicate articles
- **Multi-dimensional Scoring**: AI-powered scoring based on industry impact, milestone significance, and attention value
- **Content Extraction**: Extracts full article content using trafilatura
- **Chinese Translation**: Generates Chinese titles and summaries with entity preservation
- **Self-reflection**: Validates translation quality with automatic retry mechanism
- **Deep Search**: On-demand article research via ReAct loop with web search (DuckDuckGo/Tavily)
- **GraphRAG**: Knowledge graph construction with community detection for contextual analysis
- **Vector Storage**: PostgreSQL with pgvector extension for vector storage and graph data

## Architecture

### Main Workflow Pipeline

```
[Entry] → [Scout] → [Dedup] → [Scoring] → [Writing] → [Reflection] → [Storage] → [End]
```

### Deep Search (ReAct Loop)

On-demand deep research for articles:
1. Fetch article content
2. ReAct loop with web search tools (DuckDuckGo or Tavily)
3. Generate comprehensive report

### GraphRAG (DeepGraph)

Two-phase knowledge graph system:
- **Background GraphBuilder**: Extract entities/relationships, detect communities (Leiden algorithm)
- **On-demand GraphAnalyst**: Fetch subgraph, expand via traversal, generate analysis

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Database**: PostgreSQL + pgvector
- **AI**: OpenAI or compatible APIs (DashScope, Azure, Ollama)
- **RSS/Web**: feedparser, trafilatura, httpx, duckduckgo-search

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- OpenAI API key (or compatible)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-engine
```

2. Install dependencies (uv preferred):
```bash
uv sync
# Or use pip
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Set up database:
```bash
# Create PostgreSQL database
createdb news_aggregator

# Run migrations
alembic upgrade head
```

### Running

Start the server:
```bash
uvicorn app.main:app --reload
```

Access the API documentation at: http://localhost:8000/docs

### Docker

```bash
docker-compose up -d
docker-compose logs -f ai-engine
docker-compose down
```

## API Endpoints

All endpoints under `/api/v1/`:

### Workflow

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/workflow/run` | POST | Trigger news aggregation workflow |
| `/workflow/runs` | GET | List workflow run history |
| `/workflow/runs/{id}` | GET | Get workflow run details |
| `/workflow/articles` | GET | List articles |
| `/workflow/articles/{id}` | GET | Get article details |
| `/workflow/feeds` | GET/POST | List or add RSS feed sources |
| `/workflow/feeds/{id}` | DELETE | Delete RSS feed |
| `/workflow/feeds/{id}/toggle` | PATCH | Toggle feed active status |

### Deep Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/deep-search/run` | POST | Run deep search for an article |

### Graph Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/deep-graph/analyze` | POST | Generate knowledge graph analysis |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |
| `/health/live` | GET | Liveness check |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `OPENAI_API_KEY` | Yes | OpenAI or compatible API key |
| `OPENAI_CHAT_MODEL` | Yes | Chat model (e.g., gpt-4o-mini, qwen3.5-35b-a3b) |
| `OPENAI_EMBEDDING_MODEL` | Yes | Embedding model (e.g., text-embedding-3-small) |
| `OPENAI_BASE_URL` | No | Override API base URL (for DashScope, Azure, Ollama) |
| `WEB_SEARCH_PROVIDER` | No | Provider: "duckduckgo" or "tavily" (default: duckduckgo) |
| `RERANK_PROVIDER` | No | Provider: "none", "dashscope", "cohere", "jina" |

See `.env.example` for complete configuration options.

## Testing

```bash
pytest
pytest --cov=app tests/
pytest tests/test_rag.py -v  # Run single test file
```

## Project Structure

```
ai-engine/
├── app/
│   ├── api/           # FastAPI routes
│   ├── models/        # Database models
│   ├── agents/        # LangGraph agents (Scout, Scorer, Writer, Reflection)
│   ├── workflow/      # Main workflow graph and nodes
│   ├── deep_search/   # ReAct loop for deep research
│   ├── deep_graph/    # GraphRAG builder and analyst
│   ├── services/      # RAG services (embedding, vector_store, reranker, etc.)
│   ├── prompts/       # Centralized prompt templates
│   ├── core/          # Exceptions, logging
│   └── utils/         # Helpers, constants
├── alembic/           # Database migrations
├── tests/             # Test files
├── pyproject.toml
└── docker-compose.yml
```

## License

MIT