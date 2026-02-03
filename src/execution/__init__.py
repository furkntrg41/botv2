"""Execution layer - NautilusTrader strategies and live trading.

This module provides:

- Configuration Loading:
    - StrategyConfig: Load and validate strategy parameters
    - save_live_params: Export research results for live trading
    - load_live_params: Load parameters for execution

- Strategies (Linux/Docker only):
    - EMACrossStrategy: EMA crossover strategy for NautilusTrader

Note:
    NautilusTrader requires Linux. On Windows, strategy imports are
    gracefully skipped to allow development of other components.

Example:
    >>> from src.execution import StrategyConfig, load_live_params
    >>> config = load_live_params("config/live_params.json")
    >>> print(config.get_param("fast_window"))
    12
"""

from src.execution.config_loader import (
    StrategyConfig,
    save_live_params,
    load_live_params,
    ConfigLoadError,
    ConfigValidationError,
)

# Conditionally import strategies (only available on Linux with nautilus_trader)
try:
    from src.execution.strategies import EMACrossStrategy
    _STRATEGIES_AVAILABLE = True
except ImportError:
    _STRATEGIES_AVAILABLE = False
    EMACrossStrategy = None  # type: ignore[misc,assignment]

__all__ = [
    # Configuration
    "StrategyConfig",
    "save_live_params",
    "load_live_params",
    "ConfigLoadError",
    "ConfigValidationError",
    # Strategies (may be None on Windows)
    "EMACrossStrategy",
    "_STRATEGIES_AVAILABLE",
]
