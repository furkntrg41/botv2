"""
Strategy Configuration Loader.

This module provides a bridge between the Research layer (BacktestEngine)
and the Execution layer (NautilusTrader). It reads optimized parameters
from JSON configuration files and provides typed access to them.

Architecture:
    ┌─────────────────┐       JSON        ┌─────────────────┐
    │  run_research   │  ─────────────►   │ live_params.json│
    │  (BacktestEngine)│                  └────────┬────────┘
    └─────────────────┘                            │
                                                   ▼
    ┌─────────────────┐      TypedDict     ┌──────────────────┐
    │ EMACrossStrategy│  ◄─────────────    │  StrategyConfig  │
    │ (NautilusTrader)│                    │  (config_loader) │
    └─────────────────┘                    └──────────────────┘
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class EMAConfigDict(TypedDict):
    """TypedDict for EMA Crossover strategy parameters."""

    fast_window: int
    slow_window: int


class RSIConfigDict(TypedDict):
    """TypedDict for RSI Mean Reversion strategy parameters."""

    rsi_window: int
    oversold: int
    overbought: int


class LiveParamsDict(TypedDict):
    """TypedDict for the complete live_params.json structure."""

    strategy_type: str
    symbol: str
    timeframe: str
    parameters: dict[str, int | float]
    metrics: dict[str, float]
    generated_at: str
    research_sharpe: float


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class ConfigLoadError(Exception):
    """Raised when configuration cannot be loaded."""


class ConfigValidationError(Exception):
    """Raised when configuration is invalid."""


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyConfig:
    """
    Immutable container for strategy configuration.

    This class loads and validates strategy parameters from JSON files,
    providing type-safe access to configuration values.

    Attributes:
        strategy_type: Name of the strategy (e.g., "ema_crossover").
        symbol: Trading pair symbol (e.g., "BTC/USDT").
        timeframe: Data timeframe (e.g., "1h").
        parameters: Dictionary of strategy-specific parameters.
        metrics: Performance metrics from research phase.
        generated_at: Timestamp when config was generated.
        research_sharpe: Sharpe ratio achieved in research.

    Example:
        >>> config = StrategyConfig.from_json("config/live_params.json")
        >>> print(config.strategy_type)
        'ema_crossover'
        >>> print(config.get_param("fast_window", default=12))
        10
    """

    strategy_type: str
    symbol: str
    timeframe: str
    parameters: dict[str, int | float]
    metrics: dict[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    research_sharpe: float = 0.0
    config_path: str | None = None

    @classmethod
    def from_json(cls, filepath: str | Path) -> StrategyConfig:
        """
        Load strategy configuration from a JSON file.

        Args:
            filepath: Path to the JSON configuration file.

        Returns:
            StrategyConfig instance with loaded parameters.

        Raises:
            ConfigLoadError: If file cannot be read.
            ConfigValidationError: If file content is invalid.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise ConfigLoadError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data: LiveParamsDict = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Invalid JSON in {filepath}: {e}") from e
        except OSError as e:
            raise ConfigLoadError(f"Cannot read {filepath}: {e}") from e

        # Validate required fields
        required_fields = ["strategy_type", "symbol", "parameters"]
        for field_name in required_fields:
            if field_name not in data:
                raise ConfigValidationError(
                    f"Missing required field '{field_name}' in {filepath}"
                )

        # Parse timestamp
        generated_at = datetime.utcnow()
        if "generated_at" in data:
            try:
                generated_at = datetime.fromisoformat(data["generated_at"])
            except ValueError:
                logger.warning(f"Invalid timestamp format in {filepath}, using current time")

        config = cls(
            strategy_type=data["strategy_type"],
            symbol=data["symbol"],
            timeframe=data.get("timeframe", "1h"),
            parameters=data["parameters"],
            metrics=data.get("metrics", {}),
            generated_at=generated_at,
            research_sharpe=data.get("research_sharpe", 0.0),
            config_path=str(filepath),
        )

        logger.info(
            f"Loaded config | Strategy: {config.strategy_type} | "
            f"Symbol: {config.symbol} | Sharpe: {config.research_sharpe:.3f}"
        )

        return config

    def get_param(self, name: str, default: int | float | None = None) -> int | float:
        """
        Get a parameter value with optional default.

        Args:
            name: Parameter name.
            default: Default value if parameter not found.

        Returns:
            Parameter value.

        Raises:
            KeyError: If parameter not found and no default provided.
        """
        if name in self.parameters:
            return self.parameters[name]
        if default is not None:
            return default
        raise KeyError(f"Parameter '{name}' not found in config")

    def get_ema_params(self) -> EMAConfigDict:
        """
        Get EMA crossover specific parameters.

        Returns:
            EMAConfigDict with fast_window and slow_window.

        Raises:
            ConfigValidationError: If required parameters are missing.
        """
        try:
            return EMAConfigDict(
                fast_window=int(self.get_param("fast_window")),
                slow_window=int(self.get_param("slow_window")),
            )
        except KeyError as e:
            raise ConfigValidationError(
                f"Missing EMA parameter: {e}. Available: {list(self.parameters.keys())}"
            ) from e

    def get_rsi_params(self) -> RSIConfigDict:
        """
        Get RSI mean reversion specific parameters.

        Returns:
            RSIConfigDict with rsi_window, oversold, and overbought.

        Raises:
            ConfigValidationError: If required parameters are missing.
        """
        try:
            return RSIConfigDict(
                rsi_window=int(self.get_param("rsi_window")),
                oversold=int(self.get_param("oversold")),
                overbought=int(self.get_param("overbought")),
            )
        except KeyError as e:
            raise ConfigValidationError(
                f"Missing RSI parameter: {e}. Available: {list(self.parameters.keys())}"
            ) from e

    def to_dict(self) -> LiveParamsDict:
        """Convert config back to dictionary for serialization."""
        return LiveParamsDict(
            strategy_type=self.strategy_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            parameters=self.parameters,
            metrics=self.metrics,
            generated_at=self.generated_at.isoformat(),
            research_sharpe=self.research_sharpe,
        )

    def __str__(self) -> str:
        """Return string representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"StrategyConfig({self.strategy_type}: {params_str})"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def save_live_params(
    strategy_type: str,
    symbol: str,
    timeframe: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    sharpe_ratio: float,
    output_path: str | Path = "config/live_params.json",
) -> Path:
    """
    Save optimized parameters to JSON file for live trading.

    This function is called by run_research.py after optimization
    to persist the best parameters for the execution layer.

    Args:
        strategy_type: Name of the strategy.
        symbol: Trading pair symbol.
        timeframe: Data timeframe.
        parameters: Strategy parameters dictionary.
        metrics: Performance metrics dictionary.
        sharpe_ratio: Sharpe ratio from research.
        output_path: Path to save the JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_data: LiveParamsDict = {
        "strategy_type": strategy_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "parameters": {k: int(v) if isinstance(v, (int, float)) and v == int(v) else v 
                      for k, v in parameters.items()},
        "metrics": metrics,
        "generated_at": datetime.utcnow().isoformat(),
        "research_sharpe": sharpe_ratio,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    logger.success(f"Live parameters saved to: {output_path.absolute()}")
    return output_path


def load_live_params(config_path: str | Path = "config/live_params.json") -> StrategyConfig:
    """
    Convenience function to load live parameters.

    Args:
        config_path: Path to configuration file.

    Returns:
        StrategyConfig instance.
    """
    return StrategyConfig.from_json(config_path)
