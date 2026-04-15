---
name: refinery-cli
description: Use Refinery CLI commands to run news aggregation workflows, chat with articles, build knowledge graphs, and manage services
invocable: true
---

# Refinery CLI Agent Skill

This skill helps agents use the Refinery AI-powered tech news aggregation service via CLI commands.

## Prerequisites

Before using CLI commands, ensure:

1. **Services are running**: PostgreSQL and Redis containers
2. **Configuration is set**: Either via `refinery init` or environment variables
3. **Package is installed**: `pip install -e .` or `uv sync`

## CLI Commands Overview

All commands use the `refinery` CLI entry point:

```bash
refinery --help                 # Show all commands
refinery <command> --help       # Show command-specific help
```

### 1. Initialization

```bash
# Interactive setup wizard
refinery init

# Force overwrite existing config
refinery init --force

# Non-interactive (uses environment variables)
refinery init --non-interactive
```

Configuration is stored at `~/.refinery/config.toml` (global) and `.refinery.toml` (local override).

### 2. Services Management

```bash
# Start Docker services (PostgreSQL, Redis)
refinery services start

# Start without waiting for health checks
refinery services start --no-wait

# Check services status
refinery services status

# Stop services
refinery services stop

# Stop and remove volumes (deletes all data)
refinery services stop -v

# Restart services
refinery services restart

# View logs
refinery services logs
refinery services logs postgres
refinery services logs -f           # Follow mode

# Check if running
refinery services is-running
```

### 3. Workflow Commands

```bash
# Run the full news aggregation workflow
refinery workflow run

# Run with options
refinery workflow run --dry-run     # Preview without storing
refinery workflow run --limit 10    # Limit articles processed
refinery workflow run --json        # JSON output

# List workflow runs
refinery workflow runs

# Get specific run details
refinery workflow runs <run-id>

# List articles
refinery workflow articles
refinery workflow articles --limit 20
refinery workflow articles --status scored

# Show article details
refinery workflow articles <article-id>
```

### 4. Article Commands

```bash
# List articles
refinery article list
refinery article list --scored      # Only scored articles
refinery article list --limit 50

# Show article details
refinery article show <article-id>

# Run deep search on article
refinery article deep-search <article-id>

# Run GraphRAG analysis
refinery article analyze <article-id>
```

### 5. Chat Commands

```bash
# Start interactive chat about an article
refinery chat <article-id>

# Chat with specific user
refinery chat -u <user-id> <article-id>

# Exit commands: quit, exit, q
```

### 6. GraphRAG Commands

```bash
# Build knowledge graph from articles
refinery graph build <article-id-1> <article-id-2>

# Analyze graph
refinery graph analyze <article-id>
refinery graph analyze <article-id> --hops 3
refinery graph analyze <article-id> --expansion 100

# JSON output
refinery graph build <article-id> --json
refinery graph analyze <article-id> --json
```

### 7. Feed Management

```bash
# List RSS feeds
refinery feed list

# Add new feed
refinery feed add <url> --name <name>

# Delete feed
refinery feed delete <feed-id>

# Toggle feed active status
refinery feed toggle <feed-id>
```

## Environment Variables

Key environment variables (can be set in `.env` or via config):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection URL |
| `OPENAI_API_KEY` | Yes | API key for LLM |
| `OPENAI_CHAT_MODEL` | Yes | Chat model name |
| `OPENAI_EMBEDDING_MODEL` | Yes | Embedding model name |
| `OPENAI_BASE_URL` | No | Custom API endpoint |
| `WEB_SEARCH_PROVIDER` | No | "duckduckgo" or "tavily" |

## Common Workflows

### Full News Aggregation Pipeline

```bash
# 1. Ensure services running
refinery services start

# 2. Run workflow
refinery workflow run

# 3. Check results
refinery article list --scored

# 4. Deep search specific articles
refinery article deep-search <article-id>
```

### Interactive Chat with Article

```bash
# 1. Find article
refinery article list --limit 10

# 2. Start chat
refinery chat <article-id>

# 3. Ask questions interactively
# Type 'quit' to exit
```

### Knowledge Graph Analysis

```bash
# 1. Build graph from articles
refinery graph build <article-id-1> <article-id-2>

# 2. Analyze with expansion
refinery graph analyze <article-id> --hops 2
```

## Architecture Overview

The project uses LangGraph for workflow orchestration:

- **Main Workflow**: Entry → Scout (RSS) → Dedup → Scoring → Writing → Reflection → Storage
- **Deep Search**: ReAct loop with web search tools
- **GraphRAG**: Entity extraction → Community detection → Graph analysis

All LangGraph workflows use `context_schema` for dependency injection.

## Database Models

Key tables in PostgreSQL with pgvector:

- `news_articles`: Main article storage
- `rss_feeds`: RSS feed sources
- `workflow_runs`: Workflow execution history
- `chat_conversations`: Chat session tracking
- `graph_entities`: Knowledge graph entities
- `graph_relationships`: Entity relationships
- `graph_communities`: Detected communities

## Troubleshooting

### Command Not Found

```bash
# Reinstall package
pip install -e .
# Or with uv
uv sync
```

### Services Not Running

```bash
# Check status
refinery services status

# Start services
refinery services start

# Check Docker
docker ps
docker-compose ps
```

### Database Connection Error

Check:
- PostgreSQL container running: `docker ps | grep postgres`
- DATABASE_URL correct in config
- pgvector extension installed

### API Key Issues

- Verify OPENAI_API_KEY in config or environment
- Check OPENAI_BASE_URL if using custom endpoint (DashScope, Azure, etc.)

## Tips for Agents

1. **Always check services first**: Run `refinery services status` before workflow operations
2. **Use --json for parsing**: When you need structured output for programmatic use
3. **Validate UUIDs**: Article IDs must be valid UUIDs (e.g., `e31cb856-7d57-48a3-a6bd-158b5e8a7bca`)
4. **Use dry-run for testing**: `refinery workflow run --dry-run` to preview without storing
5. **Check logs on errors**: `refinery services logs <service>` for debugging