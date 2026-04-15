"""Deep search CLI commands."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

import typer
from sqlalchemy import select

from app.cli.output import console, format_panel, print_json
from app.cli.session import get_cli_session
from app.deep_search.graph import run_deep_search
from app.models.orm_models import NewsArticle

search_app = typer.Typer(help="Deep search commands")


@search_app.command("run")
def search_run(
    article_id: str = typer.Argument(..., help="Article ID to search"),
    iterations: int = typer.Option(
        5,
        "--iterations",
        "-i",
        help="Maximum number of ReAct iterations",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Run deep search for an article.

    Executes a ReAct loop to perform comprehensive research on the article topic
    using web search tools and generates a detailed report.
    """

    async def _run() -> dict:
        async with get_cli_session() as session:
            result = await run_deep_search(
                session=session,
                article_id=article_id,
                max_iterations=iterations,
            )
            return result

    console.print(
        f"[bold blue]Starting deep search for article {article_id}...[/bold blue]"
    )
    console.print(f"[dim]Max iterations: {iterations}[/dim]")
    console.print()

    with console.status(
        "[bold green]Running deep search (this may take a while)..."
    ):
        try:
            result = asyncio.run(_run())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if json_output:
        print_json(result)
    else:
        # Display summary
        console.print()
        console.print(
            format_panel(
                f"[bold]Article ID:[/bold] {result.get('article_id', article_id)}\n"
                f"[bold]Status:[/bold] {'[green]Completed[/green]' if result.get('is_complete') else '[red]Incomplete[/red]'}\n"
                f"[bold]Iterations:[/bold] {result.get('current_iteration', 0)}/{result.get('max_iterations', iterations)}\n"
                f"[bold]Tools Used:[/bold] {len(result.get('tool_history', []))}\n"
                f"[bold]Info Collected:[/bold] {len(result.get('collected_info', []))} items\n"
                + (
                    f"\n[bold]Errors:[/bold] {len(result.get('errors', []))}"
                    if result.get("errors")
                    else ""
                ),
                title="Deep Search Summary",
            )
        )

        # Display final report in panel
        if result.get("final_report"):
            console.print()
            console.print(
                format_panel(
                    result["final_report"],
                    title="Final Report",
                    style="green",
                )
            )
        else:
            console.print()
            console.print("[yellow]No final report was generated.[/yellow]")

        # Display errors if any
        if result.get("errors"):
            console.print()
            console.print("[yellow]Errors encountered:[/yellow]")
            for error in result["errors"]:
                console.print(
                    f"  - [{error.get('phase', 'unknown')}] {error.get('message', str(error))}"
                )


@search_app.command("status")
def search_status(
    article_id: str = typer.Argument(..., help="Article ID"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Check if deep search has been performed for an article."""

    async def _check() -> dict | None:
        async with get_cli_session() as session:
            stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if not article:
                return None

            return {
                "id": str(article.id),
                "has_deepsearch": article.deepsearch_report is not None,
                "deepsearch_performed_at": _format_datetime(article.deepsearch_performed_at),
                "report_preview": (
                    article.deepsearch_report
                    if article.deepsearch_report
                    else None
                ),
            }

    with console.status("[bold blue]Checking deep search status..."):
        try:
            status = asyncio.run(_check())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if not status:
        console.print(f"[red]Article not found: {article_id}[/red]")
        raise typer.Exit(code=1)

    if json_output:
        print_json(status)
    else:
        if status["has_deepsearch"]:
            console.print(
                format_panel(
                    f"[bold]Article ID:[/bold] {status['id']}\n"
                    f"[bold]Status:[/bold] [green]Deep search performed[/green]\n"
                    f"[bold]Performed at:[/bold] {status['deepsearch_performed_at']}\n\n"
                    f"[bold]Report Preview:[/bold]\n{status['report_preview']}",
                    title="Deep Search Status",
                )
            )
        else:
            console.print(
                format_panel(
                    f"[bold]Article ID:[/bold] {status['id']}\n"
                    f"[bold]Status:[/bold] [yellow]No deep search performed[/yellow]\n\n"
                    f"[dim]Use 'search run {article_id}' to perform deep search.[/dim]",
                    title="Deep Search Status",
                )
            )


def _format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _truncate(text: str | None, max_length: int = 60) -> str:
    """Truncate text with ellipsis."""
    if text is None:
        return "-"
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."