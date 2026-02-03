"""
Analysis Module - Research & Backtesting Layer.

This module provides the following components:

Models (DTOs):
    - OptimizationResult: Complete backtest/optimization results
    - PerformanceMetrics: Strategy performance statistics
    - StrategyParameters: Optimal parameter configuration
    - StrategyConfig: Full strategy configuration
    - ReportMetadata: Report generation metadata

Engine (Calculation):
    - BacktestEngine: VectorBT-powered backtesting engine
    - BaseStrategy: Abstract base for custom strategies
    - SMACrossoverStrategy: Simple Moving Average crossover
    - RSIMeanReversionStrategy: RSI mean reversion
    - EMACrossoverStrategy: Exponential Moving Average crossover
    - StrategyRegistry: Strategy registration and lookup

Visualizer (Presentation):
    - StrategyVisualizer: Professional report generation

Example:
    >>> from src.analysis import (
    ...     BacktestEngine,
    ...     StrategyVisualizer,
    ...     SMACrossoverStrategy,
    ... )
    >>> engine = BacktestEngine()
    >>> result = engine.run_optimization(data, "sma_crossover")
    >>> visualizer = StrategyVisualizer()
    >>> visualizer.plot_performance(result, "report")
"""

from src.analysis.engine import (
    BacktestEngine,
    BaseStrategy,
    EMACrossoverStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    StrategyRegistry,
)
from src.analysis.models import (
    OptimizationResult,
    PerformanceMetrics,
    ReportMetadata,
    StrategyConfig,
    StrategyParameters,
)
from src.analysis.visualizer import StrategyVisualizer

__all__ = [
    # Models (DTOs)
    "OptimizationResult",
    "PerformanceMetrics",
    "StrategyParameters",
    "StrategyConfig",
    "ReportMetadata",
    # Engine (Calculation)
    "BacktestEngine",
    "BaseStrategy",
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "EMACrossoverStrategy",
    "StrategyRegistry",
    # Visualizer (Presentation)
    "StrategyVisualizer",
]