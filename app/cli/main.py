"""Typer CLI entry point for Refinery."""

import typer
from rich.console import Console

app = typer.Typer(
    name="refinery",
    help="AI-powered tech news aggregation service with LangGraph.",
    add_completion=False,
)
console = Console()


# Import and register subcommand groups
from app.cli.commands import (
    init_app,
    workflow_app,
    article_app,
    search_app,
    graph_app,
    chat_app,
    services_app,
)

# Register subcommand groups
app.add_typer(init_app, name="init", help="Initialize configuration with setup wizard")
app.add_typer(workflow_app, name="workflow", help="Run and manage workflows")
app.add_typer(article_app, name="article", help="Manage articles")
app.add_typer(search_app, name="search", help="Deep search operations")
app.add_typer(graph_app, name="graph", help="GraphRAG operations")
app.add_typer(chat_app, name="chat", help="Interactive chat")
app.add_typer(services_app, name="services", help="Manage Docker services")


def main() -> None:
    """Main entry point for the Refinery CLI."""
    app()


if __name__ == "__main__":
    main()