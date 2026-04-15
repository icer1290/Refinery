"""Docker service management for ai-engine CLI.

This module provides the ServiceManager class that wraps docker-compose
commands for managing the ai-engine services (PostgreSQL, Redis, ai-engine).
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class ServiceManagerError(Exception):
    """Base exception for ServiceManager errors."""
    pass


class DockerNotAvailableError(ServiceManagerError):
    """Raised when Docker or docker-compose is not installed."""
    pass


class ServiceTimeoutError(ServiceManagerError):
    """Raised when a service fails to become healthy within timeout."""
    pass


class ServiceManager:
    """Manages ai-engine Docker services via docker-compose.

    Wraps docker-compose commands to start, stop, and check status
    of the ai-engine services (PostgreSQL, Redis, ai-engine).

    Attributes:
        COMPOSE_FILE: Name of the docker-compose file (relative to project root)
        HEALTH_CHECK_TIMEOUT: Maximum seconds to wait for services to become healthy
        HEALTH_CHECK_INTERVAL: Seconds between health check attempts
    """

    COMPOSE_FILE = "docker-compose.yml"
    HEALTH_CHECK_TIMEOUT = 120  # seconds
    HEALTH_CHECK_INTERVAL = 2  # seconds

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize ServiceManager.

        Args:
            project_root: Path to the project root directory. If not provided,
                          will attempt to find it automatically.
        """
        if project_root is None:
            # Find project root by looking for docker-compose.yml
            project_root = self._find_project_root()

        self.project_root = Path(project_root).resolve()
        self.compose_file = self.project_root / self.COMPOSE_FILE

        if not self.compose_file.exists():
            raise ServiceManagerError(
                f"Docker compose file not found: {self.compose_file}"
            )

    def _find_project_root(self) -> Path:
        """Find project root by looking for docker-compose.yml."""
        current = Path.cwd()

        # Check current directory and parents
        for parent in [current] + list(current.parents):
            if (parent / self.COMPOSE_FILE).exists():
                return parent

        # Fall back to working directory
        return current

    def _run_docker_compose(
        self,
        *args: str,
        capture_output: bool = True,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a docker-compose command.

        Args:
            *args: Arguments to pass to docker-compose
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise an exception on non-zero exit

        Returns:
            CompletedProcess instance with result

        Raises:
            DockerNotAvailableError: If docker-compose is not installed
            ServiceManagerError: If the command fails and check=True
        """
        cmd = [
            "docker-compose",
            "-f", str(self.compose_file),
            *args
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=False
            )

            if check and result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise ServiceManagerError(f"Docker-compose command failed: {error_msg}")

            return result

        except FileNotFoundError:
            raise DockerNotAvailableError(
                "docker-compose is not installed or not in PATH. "
                "Please install Docker and docker-compose to use this feature."
            )

    def _check_docker_available(self) -> None:
        """Check if Docker and docker-compose are available.

        Raises:
            DockerNotAvailableError: If Docker is not available
        """
        # Check docker
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True
            )
        except FileNotFoundError:
            raise DockerNotAvailableError(
                "Docker is not installed or not in PATH. "
                "Please install Docker to use this feature."
            )

        # Check docker-compose
        try:
            subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                check=True
            )
        except FileNotFoundError:
            # Try docker compose (v2 syntax)
            try:
                subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    check=True
                )
            except FileNotFoundError:
                raise DockerNotAvailableError(
                    "docker-compose is not installed or not in PATH. "
                    "Please install Docker Compose to use this feature."
                )

    def _get_service_status_dict(self) -> dict:
        """Parse docker-compose ps output into a status dictionary.

        Returns:
            Dictionary mapping service names to their status info
        """
        result = self._run_docker_compose("ps", "--format", "json")

        services = {}

        if result.stdout and result.stdout.strip():
            import json
            try:
                # docker-compose ps --format json returns one JSON object per line
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        data = json.loads(line)
                        service_name = data.get('Service', data.get('Name', 'unknown'))
                        services[service_name] = {
                            'name': service_name,
                            'state': data.get('State', 'unknown'),
                            'status': data.get('Status', 'unknown'),
                            'health': self._parse_health(data.get('Status', '')),
                            'ports': data.get('Publishers', []),
                        }
            except json.JSONDecodeError:
                # Fall back to parsing plain text output
                services = self._parse_ps_text_output(result.stdout)

        return services

    def _parse_health(self, status: str) -> str:
        """Parse health status from docker-compose ps status string.

        Args:
            status: The status string from docker-compose ps

        Returns:
            Health status: 'healthy', 'unhealthy', 'starting', or 'unknown'
        """
        status_lower = status.lower()
        if 'healthy' in status_lower:
            return 'healthy'
        elif 'unhealthy' in status_lower:
            return 'unhealthy'
        elif 'health: starting' in status_lower:
            return 'starting'
        else:
            return 'unknown'

    def _parse_ps_text_output(self, output: str) -> dict:
        """Parse plain text docker-compose ps output as fallback.

        Args:
            output: The plain text output from docker-compose ps

        Returns:
            Dictionary mapping service names to their status info
        """
        services = {}
        lines = output.strip().split('\n')

        # Skip header line
        for line in lines[1:]:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                services[name] = {
                    'name': name,
                    'state': parts[3] if len(parts) > 3 else 'unknown',
                    'status': ' '.join(parts[3:]) if len(parts) > 3 else 'unknown',
                    'health': self._parse_health(' '.join(parts[3:])),
                    'ports': [],
                }

        return services

    def _wait_for_healthy(self, service_names: Optional[list] = None) -> bool:
        """Wait for services to become healthy.

        Args:
            service_names: List of service names to check. If None, checks all services.

        Returns:
            True if all services are healthy, False if timeout reached

        Raises:
            ServiceTimeoutError: If services don't become healthy within timeout
        """
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Waiting for services to be healthy...", total=None)

            while time.time() - start_time < self.HEALTH_CHECK_TIMEOUT:
                status = self._get_service_status_dict()

                if not status:
                    # No services found, they might still be starting
                    time.sleep(self.HEALTH_CHECK_INTERVAL)
                    continue

                # Check if target services are healthy
                target_services = service_names if service_names else list(status.keys())

                all_healthy = True
                for svc_name in target_services:
                    if svc_name in status:
                        svc = status[svc_name]
                        if svc['state'] != 'running':
                            all_healthy = False
                            break
                        # For services with health checks
                        if svc['health'] == 'unhealthy':
                            raise ServiceTimeoutError(
                                f"Service '{svc_name}' is unhealthy"
                            )
                        if svc['health'] not in ('healthy', 'unknown'):
                            all_healthy = False
                            break
                    else:
                        all_healthy = False
                        break

                if all_healthy:
                    progress.update(task, description="All services are healthy!")
                    return True

                time.sleep(self.HEALTH_CHECK_INTERVAL)

        raise ServiceTimeoutError(
            f"Services did not become healthy within {self.HEALTH_CHECK_TIMEOUT} seconds"
        )

    def start(self, wait: bool = True, services: Optional[list] = None) -> None:
        """Start ai-engine services.

        Args:
            wait: Whether to wait for services to become healthy
            services: List of specific services to start. If None, starts all services.

        Raises:
            DockerNotAvailableError: If Docker is not installed
            ServiceManagerError: If services fail to start
            ServiceTimeoutError: If services don't become healthy within timeout
        """
        self._check_docker_available()

        console.print("[bold blue]Starting ai-engine services...[/bold blue]")

        # Build command arguments
        args = ["up", "-d"]
        if services:
            args.extend(services)

        self._run_docker_compose(*args)

        console.print("[green]Services started![/green]")

        if wait:
            self._wait_for_healthy(services)
            console.print("[bold green]All services are healthy and ready![/bold green]")

    def stop(self, remove_volumes: bool = False) -> None:
        """Stop ai-engine services.

        Args:
            remove_volumes: Whether to remove volumes (warning: deletes data)

        Raises:
            DockerNotAvailableError: If Docker is not installed
            ServiceManagerError: If services fail to stop
        """
        self._check_docker_available()

        console.print("[bold blue]Stopping ai-engine services...[/bold blue]")

        args = ["down"]
        if remove_volumes:
            args.append("-v")

        self._run_docker_compose(*args)

        console.print("[green]Services stopped![/green]")

    def status(self) -> dict:
        """Get status of ai-engine services.

        Returns:
            Dictionary with service status information:
            {
                'service_name': {
                    'name': str,
                    'state': str,  # 'running', 'exited', etc.
                    'status': str,
                    'health': str,  # 'healthy', 'unhealthy', 'starting', 'unknown'
                    'ports': list,
                },
                ...
            }

        Raises:
            DockerNotAvailableError: If Docker is not installed
        """
        self._check_docker_available()
        return self._get_service_status_dict()

    def is_running(self, service: Optional[str] = None) -> bool:
        """Check if services are running.

        Args:
            service: Specific service name to check. If None, checks if any service is running.

        Returns:
            True if the specified service (or any service) is running
        """
        try:
            status = self._get_service_status_dict()
        except (DockerNotAvailableError, ServiceManagerError):
            return False

        if not status:
            return False

        if service:
            return service in status and status[service]['state'] == 'running'

        return any(svc['state'] == 'running' for svc in status.values())

    def restart(self, services: Optional[list] = None) -> None:
        """Restart ai-engine services.

        Args:
            services: List of specific services to restart. If None, restarts all.

        Raises:
            DockerNotAvailableError: If Docker is not installed
            ServiceManagerError: If services fail to restart
        """
        self._check_docker_available()

        console.print("[bold blue]Restarting ai-engine services...[/bold blue]")

        args = ["restart"]
        if services:
            args.extend(services)

        self._run_docker_compose(*args)

        console.print("[green]Services restarted![/green]")

    def logs(self, service: Optional[str] = None, follow: bool = False, tail: int = 100) -> None:
        """View logs from services.

        Args:
            service: Specific service to view logs for. If None, shows all logs.
            follow: Whether to follow log output (like tail -f)
            tail: Number of lines to show from the end of logs
        """
        self._check_docker_available()

        args = ["logs", f"--tail={tail}"]
        if follow:
            args.append("-f")
        if service:
            args.append(service)

        # Don't capture output - let it stream to console
        try:
            subprocess.run(
                ["docker-compose", "-f", str(self.compose_file)] + args,
                check=False
            )
        except FileNotFoundError:
            raise DockerNotAvailableError(
                "docker-compose is not installed or not in PATH."
            )