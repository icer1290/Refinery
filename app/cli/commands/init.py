"""Init command for Refinery CLI.

Interactive setup wizard for configuring Refinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from app.cli.config_manager import ConfigManager, DEFAULT_CONFIG

console = Console()

# Create Typer app for init command
init_app = typer.Typer(
    name="init",
    help="Initialize Refinery configuration with interactive setup wizard",
)


@init_app.callback(invoke_without_command=True)
def init_callback(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force overwrite existing configuration",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run non-interactively (requires environment variables)",
    ),
) -> None:
    """Initialize Refinery configuration.

    Interactive setup wizard that guides through:
    - LLM API configuration (API key, models, base URL)
    - Database connection settings
    - Redis service configuration
    - Web search provider settings
    - Advanced RAG and scoring parameters

    Examples:
        refinery init
        refinery init --force
        refinery init --non-interactive
    """
    raise typer.Exit(run_init(force=force, non_interactive=non_interactive))


def run_init(force: bool = False, non_interactive: bool = False) -> int:
    """Run the initialization wizard.

    Args:
        force: Force overwrite existing config.
        non_interactive: Run non-interactively (use defaults or fail).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config_manager = ConfigManager()

    # Check if config already exists
    if config_manager.exists() and not force:
        console.print(
            "[yellow]Configuration already exists at ~/.refinery/config.toml[/yellow]"
        )
        console.print("Use --force to overwrite existing configuration.")
        return 1

    if non_interactive:
        return _run_non_interactive(config_manager)

    return _run_interactive(config_manager, force)


def _run_non_interactive(config_manager: ConfigManager) -> int:
    """Run non-interactive initialization.

    Requires all required values to be set via environment variables.

    Returns:
        Exit code.
    """
    console.print("[bold]Initializing refinery (non-interactive mode)[/bold]")

    # Load config from environment only
    config = config_manager.load(use_env=True)

    # Validate
    errors = config_manager.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print(
            "\n[yellow]Set the required environment variables or run interactively.[/yellow]"
        )
        return 1

    # Save to global config
    try:
        path = config_manager.save(config, global_config=True)
        console.print(f"[green]Configuration saved to {path}[/green]")
        return 0
    except Exception as e:
        console.print(f"[red]Failed to save configuration: {e}[/red]")
        return 1


def _run_interactive(config_manager: ConfigManager, force: bool) -> int:
    """Run interactive initialization wizard.

    Args:
        config_manager: ConfigManager instance.
        force: Force overwrite existing config.

    Returns:
        Exit code.
    """
    console.print(
        Panel.fit(
            "[bold blue]AI Engine Setup Wizard[/bold blue]\n"
            "This will guide you through setting up your refinery configuration.",
            border_style="blue",
        )
    )

    config: dict[str, Any] = {}

    # Step 1: LLM Configuration
    console.print("\n[bold cyan]Step 1: LLM Configuration[/bold cyan]")

    api_key = Prompt.ask(
        "[bold]API Key[/bold]",
        password=True,
        default="",
    )
    if not api_key:
        console.print("[red]API key is required.[/red]")
        return 1
    config.setdefault("llm", {})["api_key"] = api_key

    # Base URL (optional)
    base_url = Prompt.ask(
        "[bold]Base URL[/bold] (leave empty for OpenAI, or use custom like DashScope)",
        default="",
    )
    if base_url:
        config["llm"]["base_url"] = base_url
    else:
        config["llm"]["base_url"] = None

    # Chat model
    console.print(
        "\n[dim]Popular chat models:[/dim]"
        "\n[dim]  - gpt-4o-mini, gpt-4o (OpenAI)[/dim]"
        "\n[dim]  - qwen3.5-35b-a3b, qwen-max (DashScope)[/dim]"
        "\n[dim]  - deepseek-chat (DeepSeek)[/dim]"
        "\n[dim]  - claude-3-5-sonnet-latest (Anthropic via OpenAI-compatible API)[/dim]"
    )
    chat_model = Prompt.ask(
        "[bold]Chat Model[/bold]",
        default="gpt-4o-mini",
    )
    config["llm"]["chat_model"] = chat_model

    # Embedding model
    console.print(
        "\n[dim]Popular embedding models:[/dim]"
        "\n[dim]  - text-embedding-3-small, text-embedding-3-large (OpenAI)[/dim]"
        "\n[dim]  - text-embedding-v4 (DashScope)[/dim]"
    )
    embedding_model = Prompt.ask(
        "[bold]Embedding Model[/bold]",
        default="text-embedding-3-small",
    )
    config["llm"]["embedding_model"] = embedding_model

    # Temperature
    temperature = Prompt.ask(
        "[bold]LLM Temperature[/bold]",
        default="0.3",
    )
    try:
        config["llm"]["temperature"] = float(temperature)
    except ValueError:
        config["llm"]["temperature"] = 0.3

    # Step 2: Database Configuration
    console.print("\n[bold cyan]Step 2: Database Configuration[/bold cyan]")

    db_url = Prompt.ask(
        "[bold]Database URL[/bold]",
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/news_aggregator",
    )
    config["database"] = {"url": db_url}

    # Step 3: Services Configuration
    console.print("\n[bold cyan]Step 3: Services Configuration[/bold cyan]")

    redis_enabled = Confirm.ask(
        "[bold]Enable Redis?[/bold] (for session caching)",
        default=False,
    )
    config["services"] = {"redis_enabled": redis_enabled}

    if redis_enabled:
        redis_url = Prompt.ask(
            "[bold]Redis URL[/bold]",
            default="redis://localhost:6379/0",
        )
        config["services"]["redis_url"] = redis_url
    else:
        config["services"]["redis_url"] = "redis://localhost:6379/0"

    # Step 4: Web Search Configuration
    console.print("\n[bold cyan]Step 4: Web Search Configuration[/bold cyan]")

    console.print(
        "[dim]Web search providers:[/dim]"
        "\n[dim]  - duckduckgo (free, no API key required)[/dim]"
        "\n[dim]  - tavily (requires API key, better results)[/dim]"
    )
    search_provider = Prompt.ask(
        "[bold]Web Search Provider[/bold]",
        choices=["duckduckgo", "tavily"],
        default="duckduckgo",
    )
    config["web_search"] = {"provider": search_provider}

    if search_provider == "tavily":
        tavily_key = Prompt.ask(
            "[bold]Tavily API Key[/bold]",
            password=True,
        )
        config["web_search"]["api_key"] = tavily_key if tavily_key else None
    else:
        config["web_search"]["api_key"] = None

    # Step 5: Advanced Configuration
    console.print("\n[bold cyan]Step 5: Advanced Configuration[/bold cyan]")

    advanced = Confirm.ask(
        "[bold]Configure advanced settings?[/bold]",
        default=False,
    )

    if advanced:
        # RAG settings
        console.print("\n[dim]RAG Settings[/dim]")
        chunk_size = Prompt.ask(
            "  [bold]Chunk Size[/bold]",
            default="2000",
        )
        chunk_overlap = Prompt.ask(
            "  [bold]Chunk Overlap[/bold]",
            default="400",
        )
        config["rag"] = {
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "vector_weight": 0.6,
            "fts_weight": 0.4,
        }

        # Scoring settings
        console.print("\n[dim]Scoring Settings[/dim]")
        score_threshold = Prompt.ask(
            "  [bold]Score Threshold[/bold] (minimum score to keep article)",
            default="5.0",
        )
        config["scoring"] = {
            "weight_industry_impact": 0.4,
            "weight_milestone": 0.35,
            "weight_attention": 0.25,
            "score_threshold": float(score_threshold),
        }

        # Dedup settings
        console.print("\n[dim]Deduplication Settings[/dim]")
        dedup_threshold = Prompt.ask(
            "  [bold]Similarity Threshold[/bold] (for deduplication)",
            default="0.85",
        )
        config["dedup"] = {
            "similarity_threshold": float(dedup_threshold),
        }
    else:
        # Use defaults
        config["rag"] = DEFAULT_CONFIG["rag"].copy()
        config["scoring"] = DEFAULT_CONFIG["scoring"].copy()
        config["dedup"] = DEFAULT_CONFIG["dedup"].copy()

    # Summary
    console.print("\n")
    console.print(
        Panel.fit(
            _format_config_summary(config),
            title="[bold]Configuration Summary[/bold]",
            border_style="green",
        )
    )

    # Confirm
    if not Confirm.ask("\n[bold]Save this configuration?[/bold]", default=True):
        console.print("[yellow]Configuration not saved.[/yellow]")
        return 1

    # Save configuration
    try:
        path = config_manager.save(config, global_config=True)
        console.print(f"\n[green]Configuration saved to {path}[/green]")
    except Exception as e:
        console.print(f"\n[red]Failed to save configuration: {e}[/red]")
        return 1

    # Offer to start Docker services
    if Confirm.ask("\n[bold]Would you like to start Docker services?[/bold]", default=False):
        _start_docker_services()

    console.print("\n[green]Setup complete![/green]")
    console.print("[dim]You can now run 'refinery workflow run' to start the news aggregator.[/dim]")

    return 0


def _format_config_summary(config: dict[str, Any]) -> str:
    """Format configuration summary for display.

    Args:
        config: Configuration dictionary.

    Returns:
        Formatted summary string.
    """
    lines: list[str] = []

    llm = config.get("llm", {})
    lines.append("[bold]LLM:[/bold]")
    lines.append(f"  Model: {llm.get('chat_model', 'not set')}")
    lines.append(f"  Embedding: {llm.get('embedding_model', 'not set')}")
    lines.append(f"  Base URL: {llm.get('base_url') or 'OpenAI default'}")
    lines.append(f"  Temperature: {llm.get('temperature', 0.3)}")

    db = config.get("database", {})
    lines.append("\n[bold]Database:[/bold]")
    lines.append(f"  URL: {db.get('url', 'not set')}")

    services = config.get("services", {})
    lines.append("\n[bold]Services:[/bold]")
    lines.append(f"  Redis: {'enabled' if services.get('redis_enabled') else 'disabled'}")

    web_search = config.get("web_search", {})
    lines.append("\n[bold]Web Search:[/bold]")
    lines.append(f"  Provider: {web_search.get('provider', 'duckduckgo')}")

    return "\n".join(lines)


def _start_docker_services() -> bool:
    """Attempt to start Docker services.

    Returns:
        True if successful, False otherwise.
    """
    import subprocess

    console.print("\n[cyan]Starting Docker services...[/cyan]")

    # Check if docker-compose.yml exists
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        console.print("[yellow]No docker-compose.yml found in current directory.[/yellow]")
        return False

    try:
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            console.print("[green]Docker services started successfully.[/green]")
            return True
        else:
            console.print(f"[red]Failed to start Docker services: {result.stderr}[/red]")
            return False
    except FileNotFoundError:
        console.print("[red]docker-compose not found. Please start Docker services manually.[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]Docker services startup timed out.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error starting Docker services: {e}[/red]")
        return False