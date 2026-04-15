"""GraphRAG CLI commands for building and analyzing knowledge graphs.

Commands:
- graph build <article_ids>: Build knowledge graph from articles
- graph analyze <article_ids>: Analyze graph and generate report
"""

import asyncio
import json as json_lib
import uuid

import typer
from rich.panel import Panel
from rich.table import Table

from app.cli.output import console
from app.cli.session import get_cli_session

graph_app = typer.Typer(
    name="graph",
    help="GraphRAG commands for knowledge graph operations",
)


@graph_app.command("build")
def graph_build(
    article_ids: list[str] = typer.Argument(
        ...,
        help="Article IDs to build graph from (UUIDs)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON",
    ),
) -> None:
    """Build knowledge graph from specified articles.

    Extracts entities, relationships, and communities from articles
    and stores them in the database.

    Example:
        refinery graph build abc123 def456
        refinery graph build abc123 --json
    """
    async def _build() -> dict:
        async with get_cli_session() as session:
            from app.deep_graph.graph_builder import run_graph_builder

            # Validate UUIDs
            valid_ids = []
            for aid in article_ids:
                try:
                    uuid.UUID(aid)
                    valid_ids.append(aid)
                except ValueError:
                    console.print(f"[yellow]Warning: Invalid UUID skipped: {aid}[/yellow]")

            if not valid_ids:
                console.print("[red]Error: No valid article IDs provided[/red]")
                raise typer.Exit(1)

            with console.status("[bold green]Building knowledge graph..."):
                result = await run_graph_builder(session, valid_ids)

            return result

    try:
        result = asyncio.run(_build())

        if json_output:
            output = {
                "run_id": result.get("run_id"),
                "status": result.get("current_phase"),
                "entities_count": result.get("entities_count", 0),
                "relationships_count": result.get("relationships_count", 0),
                "communities_count": result.get("communities_count", 0),
                "errors": result.get("errors", []),
            }
            console.print(json_lib.dumps(output, indent=2))
        else:
            # Display results in table
            table = Table(title="Graph Build Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green")

            table.add_row("Entities Extracted", str(result.get("entities_count", 0)))
            table.add_row("Relationships Extracted", str(result.get("relationships_count", 0)))
            table.add_row("Communities Detected", str(result.get("communities_count", 0)))
            table.add_row("Status", result.get("current_phase", "unknown"))

            console.print(table)

            if result.get("errors"):
                console.print(
                    Panel(
                        "\n".join(str(e) for e in result["errors"]),
                        title="[red]Errors[/red]",
                        border_style="red",
                    )
                )

            if result.get("current_phase") == "complete":
                console.print("[green]Graph build completed successfully[/green]")
            else:
                console.print("[red]Graph build failed[/red]")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@graph_app.command("analyze")
def graph_analyze(
    article_ids: list[str] = typer.Argument(
        ...,
        help="Article IDs to analyze (UUIDs)",
    ),
    hops: int = typer.Option(
        2,
        "--hops",
        "-h",
        help="Maximum hops for graph expansion",
    ),
    expansion: int = typer.Option(
        50,
        "--expansion",
        "-e",
        help="Maximum entities to add through expansion",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON",
    ),
) -> None:
    """Analyze knowledge graph and generate comprehensive report.

    Fetches seed subgraph from articles, expands via graph traversal,
    and generates analysis report.

    Example:
        refinery graph analyze abc123
        refinery graph analyze abc123 def456 --hops 3 --expansion 100
        refinery graph analyze abc123 --json
    """
    async def _analyze() -> dict:
        async with get_cli_session() as session:
            from app.deep_graph.graph_analyst import run_deep_graph_analyst

            # Validate UUIDs
            valid_ids = []
            for aid in article_ids:
                try:
                    uuid.UUID(aid)
                    valid_ids.append(aid)
                except ValueError:
                    console.print(f"[yellow]Warning: Invalid UUID skipped: {aid}[/yellow]")

            if not valid_ids:
                console.print("[red]Error: No valid article IDs provided[/red]")
                raise typer.Exit(1)

            with console.status(
                f"[bold green]Analyzing graph (hops={hops}, expansion={expansion})..."
            ):
                result = await run_deep_graph_analyst(
                    session,
                    valid_ids,
                    max_hops=hops,
                    expansion_limit=expansion,
                )

            return result

    try:
        result = asyncio.run(_analyze())

        if json_output:
            output = {
                "status": result.get("current_phase"),
                "nodes_count": len(result.get("graph_nodes", [])),
                "edges_count": len(result.get("graph_edges", [])),
                "communities_count": len(result.get("communities", [])),
                "report": result.get("final_report"),
                "visualization": result.get("visualization_data"),
                "errors": result.get("errors", []),
            }
            console.print(json_lib.dumps(output, indent=2))
        else:
            # Display summary stats
            table = Table(title="Graph Analysis Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Graph Nodes", str(len(result.get("graph_nodes", []))))
            table.add_row("Graph Edges", str(len(result.get("graph_edges", []))))
            table.add_row("Communities", str(len(result.get("communities", []))))
            table.add_row("Status", result.get("current_phase", "unknown"))

            console.print(table)

            # Display analysis report
            report = result.get("final_report", "")
            if report:
                console.print(
                    Panel(
                        report,
                        title="[bold blue]Analysis Report[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                )

            if result.get("errors"):
                console.print(
                    Panel(
                        "\n".join(str(e) for e in result["errors"]),
                        title="[red]Errors[/red]",
                        border_style="red",
                    )
                )

            if result.get("current_phase") == "complete":
                console.print("[green]Graph analysis completed successfully[/green]")
            else:
                console.print("[red]Graph analysis failed[/red]")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)