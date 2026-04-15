"""Services management commands for ai-engine CLI.

Provides commands to start, stop, restart, and check status of
ai-engine Docker services (PostgreSQL, Redis, ai-engine).
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.cli.service_manager import (
    ServiceManager,
    ServiceManagerError,
    DockerNotAvailableError,
    ServiceTimeoutError,
)

app = typer.Typer(
    name="services",
    help="Manage ai-engine Docker services (PostgreSQL, Redis, ai-engine)",
)
console = Console()


def get_service_manager() -> ServiceManager:
    """Get ServiceManager instance.

    Returns:
        ServiceManager instance

    Raises:
        typer.Exit: If project root cannot be determined
    """
    try:
        return ServiceManager()
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def start(
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for services to become healthy before returning",
    ),
    services: Optional[list[str]] = typer.Argument(
        None,
        help="Specific services to start (default: all services)",
    ),
) -> None:
    """Start ai-engine Docker services.

    Starts all services defined in docker-compose.yml by default.
    Use --no-wait to return immediately without waiting for health checks.

    Examples:
        ai-engine services start
        ai-engine services start postgres redis
        ai-engine services start --no-wait
    """
    manager = get_service_manager()

    try:
        manager.start(wait=wait, services=services)
    except DockerNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ServiceTimeoutError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Tip: Check service logs with 'ai-engine services logs'[/yellow]")
        raise typer.Exit(1)
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stop(
    remove_volumes: bool = typer.Option(
        False,
        "--remove-volumes/-v",
        help="Remove volumes (warning: deletes all data)",
    ),
) -> None:
    """Stop ai-engine Docker services.

    Stops all running services. Use --remove-volumes to also delete
    persistent data (PostgreSQL data, Redis data).

    Examples:
        ai-engine services stop
        ai-engine services stop -v  # Also removes volumes
    """
    manager = get_service_manager()

    if remove_volumes:
        console.print("[yellow]Warning: This will delete all persisted data![/yellow]")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("[blue]Cancelled.[/blue]")
            raise typer.Exit(0)

    try:
        manager.stop(remove_volumes=remove_volumes)
    except DockerNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def restart(
    services: Optional[list[str]] = typer.Argument(
        None,
        help="Specific services to restart (default: all services)",
    ),
) -> None:
    """Restart ai-engine Docker services.

    Restarts all services by default, or specific services if provided.

    Examples:
        ai-engine services restart
        ai-engine services restart postgres
    """
    manager = get_service_manager()

    try:
        manager.restart(services=services)
    except DockerNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def status() -> None:
    """Show status of ai-engine Docker services.

    Displays a table with service name, state, health status, and ports.

    Example:
        ai-engine services status
    """
    manager = get_service_manager()

    try:
        services = manager.status()
    except DockerNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not services:
        console.print("[yellow]No services are currently running.[/yellow]")
        console.print("[blue]Start services with: ai-engine services start[/blue]")
        return

    table = Table(title="ai-engine Services")
    table.add_column("Service", style="cyan")
    table.add_column("State", style="magenta")
    table.add_column("Health", style="green")
    table.add_column("Ports")

    for name, info in services.items():
        # Color-code state
        state = info.get('state', 'unknown')
        if state == 'running':
            state_display = f"[green]{state}[/green]"
        elif state == 'exited':
            state_display = f"[red]{state}[/red]"
        else:
            state_display = f"[yellow]{state}[/yellow]"

        # Color-code health
        health = info.get('health', 'unknown')
        if health == 'healthy':
            health_display = f"[green]{health}[/green]"
        elif health == 'unhealthy':
            health_display = f"[red]{health}[/red]"
        elif health == 'starting':
            health_display = f"[yellow]{health}[/yellow]"
        else:
            health_display = f"[dim]{health}[/dim]"

        # Format ports
        ports = info.get('ports', [])
        if ports:
            port_str = ", ".join(
                f"{p.get('PublishedPort', '?')}:{p.get('TargetPort', '?')}"
                for p in ports
                if p.get('PublishedPort')
            )
        else:
            port_str = "-"

        table.add_row(name, state_display, health_display, port_str or "-")

    console.print(table)


@app.command()
def logs(
    service: Optional[str] = typer.Argument(
        None,
        help="Specific service to view logs for",
    ),
    follow: bool = typer.Option(
        False,
        "--follow/-f",
        help="Follow log output",
    ),
    tail: int = typer.Option(
        100,
        "--tail",
        "-n",
        help="Number of lines to show from the end of logs",
    ),
) -> None:
    """View logs from ai-engine Docker services.

    Shows logs from all services by default, or from a specific service.
    Use -f to follow log output in real-time.

    Examples:
        ai-engine services logs
        ai-engine services logs postgres
        ai-engine services logs -f
        ai-engine services logs ai-engine --tail=50
    """
    manager = get_service_manager()

    try:
        manager.logs(service=service, follow=follow, tail=tail)
    except DockerNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ServiceManagerError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def is_running(
    service: Optional[str] = typer.Argument(
        None,
        help="Specific service to check",
    ),
) -> None:
    """Check if services are running.

    Returns exit code 0 if services are running, 1 otherwise.
    Useful for scripts and automation.

    Examples:
        ai-engine services is-running
        ai-engine services is-running postgres
    """
    manager = get_service_manager()

    try:
        running = manager.is_running(service)
    except (DockerNotAvailableError, ServiceManagerError):
        running = False

    if running:
        if service:
            console.print(f"[green]Service '{service}' is running[/green]")
        else:
            console.print("[green]Services are running[/green]")
        raise typer.Exit(0)
    else:
        if service:
            console.print(f"[red]Service '{service}' is not running[/red]")
        else:
            console.print("[red]No services are running[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()