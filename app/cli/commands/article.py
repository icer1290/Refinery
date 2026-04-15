"""Article CLI commands."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

import typer
from sqlalchemy import desc, func, select

from app.cli.output import console, format_panel, print_json, print_table
from app.cli.session import get_cli_session
from app.models.orm_models import NewsArticle

article_app = typer.Typer(help="Article management commands")


@article_app.command("list")
def article_list(
    page: int = typer.Option(
        1,
        "--page",
        "-p",
        help="Page number",
    ),
    size: int = typer.Option(
        20,
        "--size",
        "-s",
        help="Page size",
    ),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        "-m",
        help="Minimum score filter",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Filter by source name",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """List articles with pagination."""

    async def _list() -> tuple[list[dict], int]:
        async with get_cli_session() as session:
            # Build query with filters
            stmt = select(NewsArticle)

            if min_score is not None:
                stmt = stmt.where(NewsArticle.total_score >= min_score)
            if source:
                stmt = stmt.where(NewsArticle.source_name.ilike(f"%{source}%"))

            # Get total count
            count_stmt = select(func.count()).select_from(stmt.subquery())
            count_result = await session.execute(count_stmt)
            total = count_result.scalar()

            # Apply pagination and ordering
            stmt = stmt.order_by(desc(NewsArticle.published_at)).offset(
                (page - 1) * size
            ).limit(size)

            result = await session.execute(stmt)
            articles = result.scalars().all()

            return [
                {
                    "id": str(article.id),
                    "source_name": article.source_name,
                    "original_title": article.original_title,
                    "chinese_title": article.chinese_title,
                    "total_score": article.total_score,
                    "published_at": _format_datetime(article.published_at),
                    "source_url": article.source_url,
                }
                for article in articles
            ], total

    with console.status("[bold blue]Fetching articles..."):
        try:
            articles, total = asyncio.run(_list())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if json_output:
        print_json({"articles": articles, "total": total, "page": page, "size": size})
    else:
        if not articles:
            console.print("[yellow]No articles found.[/yellow]")
            return

        total_pages = (total + size - 1) // size
        console.print(
            f"\n[bold]Articles[/bold] (Page {page}/{total_pages}, Total: {total})"
        )

        # Format for table display
        table_data = [
            {
                "ID": article["id"][:8],
                "Score": _format_score(article["total_score"]),
                "Source": _truncate(article["source_name"], 15),
                "Title": _truncate(
                    article["chinese_title"] or article["original_title"], 50
                ),
                "Published": article["published_at"],
            }
            for article in articles
        ]
        print_table(table_data)

        console.print()
        console.print(
            f"[dim]Use --page {page + 1} for next page, --min-score to filter by score[/dim]"
        )


@article_app.command("show")
def article_show(
    article_id: str = typer.Argument(..., help="Article ID"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Show details of a specific article."""

    async def _show() -> dict | None:
        async with get_cli_session() as session:
            stmt = select(NewsArticle).where(NewsArticle.id == UUID(article_id))
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if not article:
                return None

            return {
                "id": str(article.id),
                "source_name": article.source_name,
                "source_url": article.source_url,
                "original_title": article.original_title,
                "original_description": article.original_description,
                "chinese_title": article.chinese_title,
                "chinese_summary": article.chinese_summary,
                "full_content": article.full_content,
                "total_score": article.total_score,
                "industry_impact_score": article.industry_impact_score,
                "milestone_score": article.milestone_score,
                "attention_score": article.attention_score,
                "published_at": _format_datetime(article.published_at),
                "processed_at": _format_datetime(article.processed_at),
                "reflection_passed": article.reflection_passed,
                "reflection_feedback": article.reflection_feedback,
                "deepsearch_report": article.deepsearch_report,
                "deepsearch_performed_at": _format_datetime(article.deepsearch_performed_at),
            }

    with console.status("[bold blue]Fetching article..."):
        try:
            article = asyncio.run(_show())
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

    if not article:
        console.print(f"[red]Article not found: {article_id}[/red]")
        raise typer.Exit(code=1)

    if json_output:
        print_json(article)
    else:
        # Build content for panel
        content_lines = [
            f"[bold]ID:[/bold] {article['id']}",
            f"[bold]Source:[/bold] {article['source_name']}",
            f"[bold]URL:[/bold] {article['source_url']}",
            "",
            f"[bold]Original Title:[/bold] {article['original_title']}",
        ]

        if article["original_description"]:
            content_lines.append(
                f"[bold]Description:[/bold] {article['original_description']}"
            )

        content_lines.append("")
        content_lines.append(
            f"[bold]Chinese Title:[/bold] {article['chinese_title'] or '-'}"
        )

        if article["chinese_summary"]:
            content_lines.append(
                f"[bold]Chinese Summary:[/bold] {article['chinese_summary']}"
            )

        content_lines.append("")
        content_lines.append("[bold]Scores:[/bold]")
        content_lines.append(
            f"  Total: {_format_score(article['total_score'])}  |  "
            f"Industry Impact: {_format_score(article['industry_impact_score'])}  |  "
            f"Milestone: {_format_score(article['milestone_score'])}  |  "
            f"Attention: {_format_score(article['attention_score'])}"
        )

        content_lines.append("")
        content_lines.append("[bold]Timestamps:[/bold]")
        content_lines.append(
            f"  Published: {article['published_at']}  |  "
            f"Processed: {article['processed_at']}"
        )

        content_lines.append("")
        content_lines.append(
            f"[bold]Reflection:[/bold] {'[green]Passed[/green]' if article['reflection_passed'] else '[red]Not Passed[/red]'}"
        )

        if article["deepsearch_report"]:
            content_lines.append(
                f"[bold]Deep Search:[/bold] [green]Available[/green] ({article['deepsearch_performed_at']})"
            )
        else:
            content_lines.append("[bold]Deep Search:[/bold] [dim]Not performed[/dim]")

        console.print()
        console.print(
            format_panel("\n".join(content_lines), title="Article Details")
        )

        # Show full content if available
        if article["full_content"]:
            console.print()
            console.print(
                format_panel(
                    article["full_content"],
                    title="Full Content",
                    style="dim",
                )
            )

        # Show deep search report if available
        if article["deepsearch_report"]:
            console.print()
            console.print(
                format_panel(
                    article["deepsearch_report"],
                    title="Deep Search Report",
                    style="green",
                )
            )


def _format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_score(score: float | None) -> str:
    """Format score for display."""
    if score is None:
        return "-"
    return f"{score:.1f}"


def _truncate(text: str | None, max_length: int = 60) -> str:
    """Truncate text with ellipsis."""
    if text is None:
        return "-"
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."