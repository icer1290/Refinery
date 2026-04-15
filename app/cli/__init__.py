"""CLI package for ai-engine."""

from app.cli.config_manager import ConfigManager, get_config_manager
from app.cli.service_manager import ServiceManager
from app.cli.output import console

__version__ = "0.1.0"

__all__ = ["ConfigManager", "get_config_manager", "ServiceManager", "__version__", "console"]
