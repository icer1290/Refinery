"""Workflow CLI commands."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import typer
from sqlalchemy import desc, func, select

from app.cli.output import console, format_panel, print_json, print_table
from app.cli.session import get_cli_session
from app.models.orm_models import WorkflowRun
from app.workflow.graph import run_workflow

workflow_app = typer.Typer(help="Workflow management commands")


@workflow_app.command("run")
def workflow_run(
    feed: Optional[list[str]] = typer.Option(
        None,
        "--feed",
        "-f",
        help="Specific RSS feed URL(s) to fetch (can be used multiple times)",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Override score threshold for articles",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reprocessing of existing articles",
    ),
    hours: int = typer.Option(
        24,
        "--hours",
        "-h",
        help="Hours to look back for RSS entries",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Run the news aggregation workflow.

    Executes the full workflow: Scout -> Dedup -> Scoring -> Writing -> Reflection -> Storage
    """

    async def _run() -> dict:
        async with get_cli_session() as session:
            return await run_workflow(
                session=session,
                feed_urls=feed if feed else None,
                score_threshold=threshold,
                force_reprocess=force,
                hours_back=hours,
            )

    with console.status("[bold blue]Running workflow..."):
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
                f"[bold green]Workflow completed successfully[/bold green]\n\n"
                f"[bold]Run ID:[/bold] {result['run_id']}\n"
                f"[bold]Status:[/bold] {result['current_phase']}\n"
                f"[bold]Feeds Fetched:[/bold] {result['total_feeds_fetched']}\n"
                f"[bold]Articles Found:[/bold] {result['total_articles_found']}\n"
                f"[bold]After Dedup:[/bold] {result['total_articles_after_dedup']}\n"
                f"[bold]After Scoring:[/bold] {result['total_articles_after_scoring']}\n"
                f"[bold]Articles Stored:[/bold] {result['total_articles_stored']}",
                title="Workflow Result",
            )
        )

        if result.get("errors"):
            console.print()
            console.print("[yellow]Warnings/Errors:[/yellow]")
            for error in result["errors"]:
                console.print(
                    f"  - [{error.get('phase', 'unknown')}] {error.get('message', str(error))}"
                )


@workflow_app.command("list")
def workflow_list(
    hours: int = typer.Option(
        24,
        "--hours",
        "-h",
        help="Show runs from the last N hours",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of runs to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """List workflow run history."""

    async def _list() -> list[dict]:
        async with get_cli_session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            stmt = (
                select(WorkflowRun)
                .where(WorkflowRun.started_at >= cutoff)
                .order_by(desc(WorkflowRun.started_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            runs = result.scalars().all()

            return [
                {
                    "id": str(run.id),
                    "status": run.status,
                    "started_at": _format_datetime(run.started_at),
                    "completed_at": _format_datetime(run.completed_at),
                    "total_feeds_fetched": run.total_feeds_fetched,
                    "total_articles_found": run.total_articles_found,
                    "total_articles_after_dedup": run.total_articles_after_dedup,
                    "total_articles_after_scoring": run.total_articles_after_scoring,
                    "total_articles_stored": run.total_articles_stored,
                    "errors": run.errors,
                }
                for run in runs
            ]

    with console.status("[bold blue]Fetching workflow runs..."):
        try:
            runs = asyncio.run(_list())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if json_output:
        print_json(runs)
    else:
        if not runs:
            console.print("[yellow]No workflow runs found.[/yellow]")
            return

        # Format for table display
        table_data = [
            {
                "ID": run["id"][:8],
                "Status": _format_status(run["status"]),
                "Started": run["started_at"],
                "Stored": str(run["total_articles_stored"]),
                "Errors": str(len(run.get("errors") or [])),
            }
            for run in runs
        ]
        print_table(table_data, title="Workflow Runs")


@workflow_app.command("show")
def workflow_show(
    run_id: str = typer.Argument(..., help="Workflow run ID"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Show details of a specific workflow run."""

    async def _show() -> dict | None:
        async with get_cli_session() as session:
            stmt = select(WorkflowRun).where(WorkflowRun.id == UUID(run_id))
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()

            if not run:
                return None

            return {
                "id": str(run.id),
                "status": run.status,
                "started_at": _format_datetime(run.started_at),
                "completed_at": _format_datetime(run.completed_at),
                "total_feeds_fetched": run.total_feeds_fetched,
                "total_articles_found": run.total_articles_found,
                "total_articles_after_dedup": run.total_articles_after_dedup,
                "total_articles_after_scoring": run.total_articles_after_scoring,
                "total_articles_stored": run.total_articles_stored,
                "errors": run.errors,
                "metadata": run.metadata_,
            }

    with console.status("[bold blue]Fetching workflow run..."):
        try:
            run = asyncio.run(_show())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if not run:
        console.print(f"[red]Workflow run not found: {run_id}[/red]")
        raise typer.Exit(code=1)

    if json_output:
        print_json(run)
    else:
        error_list = ""
        if run.get("errors"):
            error_list = "\n[bold]Errors:[/bold]\n" + "\n".join(
                f"  - {e}" for e in run["errors"]
            )

        console.print(
            format_panel(
                f"[bold]Run ID:[/bold] {run['id']}\n"
                f"[bold]Status:[/bold] {_format_status(run['status'])}\n"
                f"[bold]Started:[/bold] {run['started_at']}\n"
                f"[bold]Completed:[/bold] {run['completed_at']}\n\n"
                f"[bold]Statistics:[/bold]\n"
                f"  Feeds Fetched: {run['total_feeds_fetched']}\n"
                f"  Articles Found: {run['total_articles_found']}\n"
                f"  After Dedup: {run['total_articles_after_dedup']}\n"
                f"  After Scoring: {run['total_articles_after_scoring']}\n"
                f"  Articles Stored: {run['total_articles_stored']}\n"
                f"{error_list}",
                title="Workflow Run Details",
            )
        )


def _format_status(status: str) -> str:
    """Format status with color."""
    status_colors = {
        "running": "[blue]running[/blue]",
        "completed": "[green]completed[/green]",
        "completed_with_errors": "[yellow]completed_with_errors[/yellow]",
        "failed": "[red]failed[/red]",
    }
    return status_colors.get(status, status)


def _format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")