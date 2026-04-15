"""Output formatting utilities for CLI commands."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def format_json(data: Any) -> str:
    """Format data as JSON string.

    Args:
        data: Data to format (can be dict, list, Pydantic model, etc.)

    Returns:
        JSON string with proper indentation.
    """
    return json.dumps(data, cls=CustomEncoder, indent=2, ensure_ascii=False)


def print_json(data: Any, title: Optional[str] = None) -> None:
    """Print data as formatted JSON.

    Args:
        data: Data to print.
        title: Optional title for the panel.
    """
    json_str = format_json(data)
    if title:
        console.print(Panel(json_str, title=title, border_style="blue"))
    else:
        console.print(json_str)


def format_panel(content: str, title: Optional[str] = None, style: Optional[str] = None) -> Panel:
    """Create a Rich panel from content.

    Args:
        content: Content to display in the panel.
        title: Optional panel title.
        style: Optional border style.

    Returns:
        Rich Panel object ready to print.
    """
    return Panel(
        content,
        title=title,
        border_style=style or "blue",
        expand=False,
    )


def format_table(
    data: list[dict] = None,
    title: Optional[str] = None,
    show_header: bool = True,
    show_lines: bool = False,
    columns: list[dict] = None,
    rows: list[dict] = None,
) -> Table:
    """Create a Rich table from data.

    Supports two formats:
    1. Simple: Pass list of dicts with same keys as `data`
    2. Advanced: Pass `columns` with column specs and `rows` with row data

    Args:
        data: List of dictionaries with the same keys (simple format).
        title: Optional table title.
        show_header: Whether to show the header row.
        show_lines: Whether to show row separator lines.
        columns: List of column specs with keys: key, header, style, justify, no_wrap, width.
        rows: List of row dicts matching column keys.

    Returns:
        Rich Table object ready to print.
    """
    table = Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        border_style="blue",
    )

    # Advanced format with columns and rows
    if columns:
        for col in columns:
            table.add_column(
                col.get("header", col.get("key")),
                style=col.get("style"),
                justify=col.get("justify"),
                no_wrap=col.get("no_wrap", False),
                width=col.get("width"),
            )
        if rows:
            for row in rows:
                table.add_row(*[str(row.get(col["key"], "")) for col in columns])
        return table

    # Simple format with data list
    if not data:
        table.add_column("(no data)")
        return table

    columns_from_data = list(data[0].keys())
    for col in columns_from_data:
        table.add_column(col, style="cyan", no_wrap=False)

    for item in data:
        row = [str(item.get(col, "")) for col in columns_from_data]
        table.add_row(*row)

    return table


def print_table(
    data: list[dict],
    title: Optional[str] = None,
    show_header: bool = True,
    show_lines: bool = False,
) -> None:
    """Print data as a Rich table.

    Args:
        data: List of dictionaries with the same keys.
        title: Optional table title.
        show_header: Whether to show the header row.
        show_lines: Whether to show row separator lines.
    """
    table = format_table(data, title, show_header, show_lines)
    console.print(table)


def format_datetime(dt: datetime | str | None) -> str:
    """Format datetime for display.

    Args:
        dt: Datetime object or ISO string.

    Returns:
        Formatted datetime string.
    """
    if dt is None:
        return "-"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return dt
    return dt.strftime("%Y-%m-%d %H:%M")


def format_score(score: float | None) -> str:
    """Format score for display with color.

    Args:
        score: Score value.

    Returns:
        Formatted score string with Rich markup.
    """
    if score is None:
        return "-"
    if score >= 8:
        return f"[green]{score:.1f}[/green]"
    if score >= 6:
        return f"[yellow]{score:.1f}[/yellow]"
    return f"[dim]{score:.1f}[/dim]"


def truncate_text(text: str, max_length: int = None) -> str:
    """Return full text (truncation disabled).

    Args:
        text: Text to display.
        max_length: Ignored (kept for backward compatibility).

    Returns:
        Full text without truncation.
    """
    if not text:
        return ""
    return text


def format_article(article: dict | Any) -> str:
    """Format article data for display.

    Args:
        article: Article data (dict or Pydantic model).

    Returns:
        Formatted string for display.
    """
    if hasattr(article, "model_dump"):
        data = article.model_dump()
    elif hasattr(article, "__dict__"):
        data = article.__dict__
    else:
        data = article

    lines = []
    lines.append(f"[bold cyan]ID:[/] {data.get('id', 'N/A')}")
    lines.append(f"[bold cyan]Title:[/] {data.get('chinese_title') or data.get('original_title', 'N/A')}")
    lines.append(f"[bold cyan]Source:[/] {data.get('source', 'N/A')}")
    lines.append(f"[bold cyan]Score:[/] {data.get('score', 'N/A')}")
    lines.append(f"[bold cyan]Published:[/] {data.get('published_at', 'N/A')}")
    lines.append(f"[bold cyan]Processed:[/] {data.get('processed_at', 'N/A')}")

    if data.get("url"):
        lines.append(f"[bold cyan]URL:[/] [link={data['url']}]{data['url']}[/link]")

    if data.get("chinese_summary"):
        lines.append(f"[bold cyan]Summary:[/] {data['chinese_summary']}")

    return "\n".join(lines)


def print_article(article: dict | Any) -> None:
    """Print article data with formatting.

    Args:
        article: Article data (dict or Pydantic model).
    """
    text = format_article(article)
    console.print(Panel(text, border_style="green"))


def print_error(message: str, title: str = "Error") -> None:
    """Print error message.

    Args:
        message: Error message.
        title: Panel title.
    """
    console.print(Panel(message, title=title, border_style="red"))


def print_success(message: str, title: str = "Success") -> None:
    """Print success message.

    Args:
        message: Success message.
        title: Panel title.
    """
    console.print(Panel(message, title=title, border_style="green"))


def print_warning(message: str, title: str = "Warning") -> None:
    """Print warning message.

    Args:
        message: Warning message.
        title: Panel title.
    """
    console.print(Panel(message, title=title, border_style="yellow"))


def print_info(message: str, title: str = "Info") -> None:
    """Print info message.

    Args:
        message: Info message.
        title: Panel title.
    """
    console.print(Panel(message, title=title, border_style="blue"))


def print_code(code: str, language: str = "python") -> None:
    """Print code with syntax highlighting.

    Args:
        code: Code string.
        language: Programming language for syntax highlighting.
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)
