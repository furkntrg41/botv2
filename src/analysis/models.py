"""
Data Transfer Objects for the Analysis Layer.

This module contains strongly-typed dataclasses for passing
structured data between calculation and presentation layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import vectorbt as vbt


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StrategyParameters:
    """
    Immutable container for strategy parameters.

    Attributes:
        strategy_type: Name of the strategy (e.g., "sma_crossover").
        parameters: Dictionary of parameter names to values.
    """

    strategy_type: str
    parameters: dict[str, Any]

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.strategy_type}({params_str})"


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Immutable container for backtest performance metrics.

    Attributes:
        total_return: Total return as a decimal (0.15 = 15%).
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio.
        max_drawdown: Maximum drawdown as a decimal.
        win_rate: Percentage of winning trades.
        profit_factor: Ratio of gross profits to gross losses.
        total_trades: Total number of trades executed.
        avg_trade_duration: Average trade duration in hours.
        calmar_ratio: Calmar ratio (annual return / max drawdown).
        volatility: Annualized volatility.
    """

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    calmar_ratio: float
    volatility: float

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
        }


@dataclass
class OptimizationResult:
    """
    Container for complete backtest optimization results.

    This is the primary data transfer object between the
    calculation layer (BacktestEngine) and presentation layer (Visualizer).

    Attributes:
        best_parameters: The optimal strategy parameters found.
        metrics: Performance metrics for the best parameters.
        price_data: Original OHLCV price data used.
        entries: Entry signal series (boolean).
        exits: Exit signal series (boolean).
        equity_curve: Portfolio equity over time.
        drawdown_curve: Drawdown series over time.
        trades: DataFrame of individual trades.
        portfolio: Raw VectorBT portfolio object for advanced analysis.
        optimization_space: DataFrame of all parameter combinations tested.
        timestamp: When the optimization was run.
    """

    best_parameters: StrategyParameters
    metrics: PerformanceMetrics
    price_data: pd.DataFrame
    entries: pd.Series
    exits: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: pd.DataFrame
    portfolio: Any  # vbt.Portfolio - kept as Any to avoid import issues
    optimization_space: pd.DataFrame | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def symbol(self) -> str:
        """Extract symbol from price data if available."""
        if hasattr(self.price_data, "name"):
            return str(self.price_data.name)
        return "UNKNOWN"

    @property
    def timeframe(self) -> str:
        """Infer timeframe from price data index."""
        if len(self.price_data) < 2:
            return "unknown"
        delta = self.price_data.index[1] - self.price_data.index[0]
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(hours * 60)}m"
        if hours < 24:
            return f"{int(hours)}h"
        return f"{int(hours / 24)}d"


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyConfig:
    """
    Configuration for a backtesting strategy.

    Attributes:
        strategy_type: Type of strategy to run.
        param_ranges: Dictionary of parameter names to (min, max, step) tuples.
        initial_capital: Starting capital for backtest.
        fees: Trading fees as decimal (0.001 = 0.1%).
        slippage: Slippage as decimal.
    """

    strategy_type: str
    param_ranges: dict[str, tuple[int, int, int]]
    initial_capital: float = 10_000.0
    fees: float = 0.001
    slippage: float = 0.0005

    def get_param_arrays(self) -> dict[str, range]:
        """Convert param ranges to range objects for iteration."""
        return {
            name: range(start, stop + 1, step)
            for name, (start, stop, step) in self.param_ranges.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT METADATA
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReportMetadata:
    """
    Metadata for generated reports.

    Attributes:
        title: Report title.
        symbol: Trading pair symbol.
        timeframe: Data timeframe.
        date_range: Start and end dates of data.
        generated_at: When the report was generated.
        filepath: Path to saved report file.
    """

    title: str
    symbol: str
    timeframe: str
    date_range: tuple[datetime, datetime]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    filepath: str | None = None
