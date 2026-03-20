# Tech News Aggregator

An AI-powered tech news aggregation system built with LangGraph, FastAPI, and PostgreSQL.

## Features

- **RSS Feed Aggregation**: Automatically fetches news from 17+ tech news sources
- **Semantic Deduplication**: Uses vector embeddings to identify and remove duplicate articles
- **Multi-dimensional Scoring**: AI-powered scoring based on industry impact, milestone significance, and attention value
- **Content Extraction**: Extracts full article content using trafilatura
- **Chinese Translation**: Generates Chinese titles and summaries with entity preservation
- **Self-reflection**: Validates translation quality with automatic retry mechanism
- **Vector Storage**: Stores embeddings in PostgreSQL with pgvector extension

## Architecture

```
[Entry]
   ↓
[Scout Phase] → Parallel RSS feed fetching
   ↓
[Deduplication] → Vector similarity deduplication
   ↓
[Scoring Phase] → Multi-dimensional scoring, filter < 6.0
   ↓
[Writing Phase] → Extract content + Generate Chinese summary
   ↓
[Reflection] → Validate (retry max=3)
   ↓
[Storage Phase] → PostgreSQL + pgvector
   ↓
[End]
```

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Database**: PostgreSQL + pgvector
- **AI**: OpenAI GPT-4o-mini, text-embedding-3-small
- **RSS/Web**: feedparser, trafilatura, httpx

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tech-news-aggregator
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Set up database:
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

## API Endpoints

### Workflow

- `POST /api/v1/workflow/run` - Trigger news aggregation workflow
- `GET /api/v1/workflow/runs` - List workflow run history
- `GET /api/v1/workflow/runs/{id}` - Get workflow run details

### Articles

- `GET /api/v1/articles` - List articles
- `GET /api/v1/articles/{id}` - Get article details

### Feeds

- `GET /api/v1/feeds` - List RSS feed sources
- `POST /api/v1/feeds` - Add new RSS feed
- `DELETE /api/v1/feeds/{id}` - Delete RSS feed
- `PATCH /api/v1/feeds/{id}/toggle` - Toggle feed active status

### Health

- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Readiness check
- `GET /api/v1/health/live` - Liveness check

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DEDUP_SIMILARITY_THRESHOLD` | Similarity threshold for deduplication | `0.85` |
| `SCORE_THRESHOLD` | Minimum score to keep article | `6.0` |
| `MAX_REFLECTION_RETRIES` | Max retries for reflection | `3` |
| `MAX_CONCURRENT_SCORERS` | Parallel scoring limit | `5` |
| `MAX_CONCURRENT_WRITERS` | Parallel writing limit | `3` |

## Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app tests/
```

## Project Structure

```
tech_news_aggregator/
├── app/
│   ├── api/           # FastAPI routes
│   ├── models/        # Database models
│   ├── agents/        # LangGraph agents
│   ├── workflow/      # Workflow definition
│   ├── services/      # Business logic
│   ├── core/          # Exceptions, logging
│   └── utils/         # Helpers, constants
├── alembic/           # Database migrations
├── tests/             # Test files
├── pyproject.toml
└── requirements.txt
```

## License

MIT