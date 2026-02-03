"""
Backtest Engine - Pure Calculation Layer.

This module contains the BacktestEngine class responsible for
running vectorized backtests and parameter optimization.

NO VISUALIZATION CODE ALLOWED IN THIS MODULE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from src.analysis.models import (
    OptimizationResult,
    PerformanceMetrics,
    StrategyConfig,
    StrategyParameters,
)

if TYPE_CHECKING:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestEngineError(Exception):
    """Base exception for backtest engine errors."""


class InsufficientDataError(BacktestEngineError):
    """Raised when there is insufficient data for backtesting."""


class InvalidStrategyError(BacktestEngineError):
    """Raised when an invalid strategy type is specified."""


class OptimizationError(BacktestEngineError):
    """Raised when optimization fails."""


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY BASE CLASS (Strategy Pattern)
# ═══════════════════════════════════════════════════════════════════════════════


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Implements the Strategy Pattern to allow easy addition of new strategies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @property
    @abstractmethod
    def default_params(self) -> dict[str, tuple[int, int, int]]:
        """Return default parameter ranges as (min, max, step)."""
        ...

    @abstractmethod
    def generate_signals(
        self,
        price: pd.Series,
        **params: int,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals.

        Args:
            price: Close price series.
            **params: Strategy-specific parameters.

        Returns:
            Tuple of (entries, exits) boolean series.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# CONCRETE STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════


class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates buy signals when fast SMA crosses above slow SMA,
    and sell signals when fast SMA crosses below slow SMA.
    """

    @property
    def name(self) -> str:
        return "sma_crossover"

    @property
    def default_params(self) -> dict[str, tuple[int, int, int]]:
        return {
            "fast_window": (5, 50, 5),
            "slow_window": (20, 200, 10),
        }

    def generate_signals(
        self,
        price: pd.Series,
        fast_window: int = 10,
        slow_window: int = 50,
    ) -> tuple[pd.Series, pd.Series]:
        """Generate SMA crossover signals."""
        fast_sma = vbt.MA.run(price, window=fast_window, short_name="fast").ma
        slow_sma = vbt.MA.run(price, window=slow_window, short_name="slow").ma

        # Entry: fast crosses above slow
        entries = fast_sma.vbt.crossed_above(slow_sma)

        # Exit: fast crosses below slow
        exits = fast_sma.vbt.crossed_below(slow_sma)

        return entries, exits


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Generates buy signals when RSI is oversold and sell signals when overbought.
    """

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    @property
    def default_params(self) -> dict[str, tuple[int, int, int]]:
        return {
            "rsi_window": (7, 21, 2),
            "oversold": (20, 35, 5),
            "overbought": (65, 80, 5),
        }

    def generate_signals(
        self,
        price: pd.Series,
        rsi_window: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> tuple[pd.Series, pd.Series]:
        """Generate RSI mean reversion signals."""
        rsi = vbt.RSI.run(price, window=rsi_window).rsi

        # Entry: RSI below oversold
        entries = rsi.vbt.crossed_below(oversold)

        # Exit: RSI above overbought
        exits = rsi.vbt.crossed_above(overbought)

        return entries, exits


class EMACrossoverStrategy(BaseStrategy):
    """
    Exponential Moving Average Crossover Strategy.

    Similar to SMA but uses EMA for faster response to price changes.
    """

    @property
    def name(self) -> str:
        return "ema_crossover"

    @property
    def default_params(self) -> dict[str, tuple[int, int, int]]:
        return {
            "fast_window": (5, 30, 5),
            "slow_window": (20, 100, 10),
        }

    def generate_signals(
        self,
        price: pd.Series,
        fast_window: int = 12,
        slow_window: int = 26,
    ) -> tuple[pd.Series, pd.Series]:
        """Generate EMA crossover signals."""
        fast_ema = vbt.MA.run(price, window=fast_window, ewm=True, short_name="fast").ma
        slow_ema = vbt.MA.run(price, window=slow_window, ewm=True, short_name="slow").ma

        entries = fast_ema.vbt.crossed_above(slow_ema)
        exits = fast_ema.vbt.crossed_below(slow_ema)

        return entries, exits


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class StrategyRegistry:
    """
    Registry for available trading strategies.

    Implements the Registry Pattern for strategy management.
    """

    _strategies: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: type[BaseStrategy]) -> type[BaseStrategy]:
        """Register a strategy class."""
        instance = strategy_class()
        cls._strategies[instance.name] = strategy_class
        return strategy_class

    @classmethod
    def get(cls, name: str) -> BaseStrategy:
        """Get a strategy instance by name."""
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise InvalidStrategyError(
                f"Unknown strategy: {name}. Available: {available}"
            )
        return cls._strategies[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())


# Register built-in strategies
StrategyRegistry.register(SMACrossoverStrategy)
StrategyRegistry.register(RSIMeanReversionStrategy)
StrategyRegistry.register(EMACrossoverStrategy)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestEngine:
    """
    High-performance vectorized backtest engine.

    This class is responsible for PURE CALCULATION only.
    No visualization or plotting code should exist here.

    Uses VectorBT for efficient vectorized backtesting and
    parameter optimization.

    Example:
        >>> engine = BacktestEngine(initial_capital=10000, fees=0.001)
        >>> result = engine.run_optimization(
        ...     price_data=df['close'],
        ...     strategy_type='sma_crossover',
        ...     fast_window=(5, 30, 5),
        ...     slow_window=(20, 100, 10),
        ... )
        >>> print(result.metrics.sharpe_ratio)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fees: float = 0.001,
        slippage: float = 0.0005,
        freq: str = "1h",
    ) -> None:
        """
        Initialize the BacktestEngine.

        Args:
            initial_capital: Starting capital for backtests.
            fees: Trading fees as decimal (0.001 = 0.1%).
            slippage: Slippage as decimal.
            freq: Data frequency for return calculations.
        """
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.freq = freq

        logger.info(
            f"BacktestEngine initialized | "
            f"Capital: ${initial_capital:,.2f} | "
            f"Fees: {fees*100:.2f}% | "
            f"Freq: {freq}"
        )

    def run_optimization(
        self,
        price_data: pd.DataFrame | pd.Series,
        strategy_type: str = "sma_crossover",
        **param_ranges: tuple[int, int, int],
    ) -> OptimizationResult:
        """
        Run parameter optimization for a trading strategy.

        Args:
            price_data: DataFrame with OHLCV columns or Series of close prices.
            strategy_type: Name of the strategy to optimize.
            **param_ranges: Parameter ranges as (min, max, step) tuples.
                           If not provided, uses strategy defaults.

        Returns:
            OptimizationResult containing best parameters and metrics.

        Raises:
            InsufficientDataError: If price data is too short.
            InvalidStrategyError: If strategy_type is unknown.
            OptimizationError: If optimization fails.
        """
        logger.info(f"Starting optimization | Strategy: {strategy_type}")

        # Extract close price
        close = self._extract_close(price_data)

        # Validate data length
        if len(close) < 100:
            raise InsufficientDataError(
                f"Insufficient data for backtesting. Got {len(close)} rows, need at least 100."
            )

        # Get strategy
        strategy = StrategyRegistry.get(strategy_type)

        # Merge provided params with defaults
        final_params = strategy.default_params.copy()
        final_params.update(param_ranges)

        logger.debug(f"Parameter ranges: {final_params}")

        try:
            # Run vectorized optimization
            result = self._optimize_strategy(
                close=close,
                price_data=price_data,
                strategy=strategy,
                param_ranges=final_params,
            )

            logger.success(
                f"Optimization complete | "
                f"Best params: {result.best_parameters} | "
                f"Sharpe: {result.metrics.sharpe_ratio:.3f}"
            )

            return result

        except Exception as e:
            logger.exception(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}") from e

    def _extract_close(self, price_data: pd.DataFrame | pd.Series) -> pd.Series:
        """Extract close price series from input data."""
        if isinstance(price_data, pd.Series):
            return price_data

        if isinstance(price_data, pd.DataFrame):
            if "close" in price_data.columns:
                return price_data["close"]
            if "Close" in price_data.columns:
                return price_data["Close"]
            # Assume first column is close
            return price_data.iloc[:, 0]

        raise ValueError(f"Unsupported price data type: {type(price_data)}")

    def _optimize_strategy(
        self,
        close: pd.Series,
        price_data: pd.DataFrame | pd.Series,
        strategy: BaseStrategy,
        param_ranges: dict[str, tuple[int, int, int]],
    ) -> OptimizationResult:
        """
        Run vectorized parameter optimization.

        Uses VectorBT's broadcasting capabilities for efficient
        grid search over parameter space.
        """
        # Build parameter grid
        param_names = list(param_ranges.keys())
        param_arrays = [
            np.arange(start, stop + 1, step)
            for start, stop, step in param_ranges.values()
        ]

        # Generate all parameter combinations using meshgrid
        if len(param_arrays) == 2:
            # 2D optimization (most common case)
            param1, param2 = np.meshgrid(param_arrays[0], param_arrays[1], indexing="ij")
            param_grid = {
                param_names[0]: param1.flatten(),
                param_names[1]: param2.flatten(),
            }
        else:
            # General case: create product of all parameters
            from itertools import product

            combinations = list(product(*param_arrays))
            param_grid = {
                name: np.array([c[i] for c in combinations])
                for i, name in enumerate(param_names)
            }

        # Optional constraint: ensure slow_window > fast_window if both exist
        if "fast_window" in param_grid and "slow_window" in param_grid:
            valid_mask = param_grid["slow_window"] > param_grid["fast_window"]
            if not valid_mask.any():
                raise OptimizationError("No valid parameter combinations: slow_window must be > fast_window")
            param_grid = {
                name: values[valid_mask]
                for name, values in param_grid.items()
            }

        n_combinations = len(list(param_grid.values())[0])
        logger.info(f"Testing {n_combinations} parameter combinations...")

        # Generate signals for all parameter combinations
        all_entries = []
        all_exits = []

        for i in range(n_combinations):
            params = {name: int(param_grid[name][i]) for name in param_names}
            entries, exits = strategy.generate_signals(close, **params)
            all_entries.append(entries)
            all_exits.append(exits)

        # Stack signals
        entries_df = pd.concat(all_entries, axis=1)
        exits_df = pd.concat(all_exits, axis=1)

        # Run vectorized portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_df,
            exits=exits_df,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
        )

        # Find best parameters by Sharpe ratio
        sharpe_ratios = portfolio.sharpe_ratio()

        # Handle NaN values
        sharpe_ratios = sharpe_ratios.fillna(-np.inf)
        best_idx = sharpe_ratios.argmax()

        # Extract best parameters
        best_params = {name: int(param_grid[name][best_idx]) for name in param_names}

        # Re-run with best parameters to get detailed results
        best_entries, best_exits = strategy.generate_signals(close, **best_params)

        best_portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=best_entries,
            exits=best_exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
        )

        # Extract metrics
        metrics = self._extract_metrics(best_portfolio)

        # Build optimization space DataFrame for analysis
        opt_space = pd.DataFrame(param_grid)
        opt_space["sharpe_ratio"] = sharpe_ratios.values
        opt_space["total_return"] = portfolio.total_return().values

        # Build result object
        result = OptimizationResult(
            best_parameters=StrategyParameters(
                strategy_type=strategy.name,
                parameters=best_params,
            ),
            metrics=metrics,
            price_data=price_data if isinstance(price_data, pd.DataFrame) else close.to_frame("close"),
            entries=best_entries,
            exits=best_exits,
            equity_curve=best_portfolio.value(),
            drawdown_curve=best_portfolio.drawdown(),
            trades=self._extract_trades(best_portfolio),
            portfolio=best_portfolio,
            optimization_space=opt_space,
        )

        return result

    def _extract_metrics(self, portfolio: vbt.Portfolio) -> PerformanceMetrics:
        """Extract performance metrics from a VectorBT portfolio."""
        stats = portfolio.stats()

        # Safely extract metrics with defaults
        def safe_get(key: str, default: float = 0.0) -> float:
            try:
                val = stats.get(key, default)
                if pd.isna(val):
                    return default
                return float(val)
            except (KeyError, TypeError):
                return default

        # Get trade statistics
        try:
            trades = portfolio.trades.records_readable
            total_trades = len(trades) if trades is not None else 0
            if total_trades > 0 and "Duration" in trades.columns:
                avg_duration = trades["Duration"].mean().total_seconds() / 3600
            else:
                avg_duration = 0.0
        except Exception:
            total_trades = 0
            avg_duration = 0.0

        return PerformanceMetrics(
            total_return=safe_get("Total Return [%]", 0.0) / 100,
            sharpe_ratio=safe_get("Sharpe Ratio", 0.0),
            sortino_ratio=safe_get("Sortino Ratio", 0.0),
            max_drawdown=abs(safe_get("Max Drawdown [%]", 0.0)) / 100,
            win_rate=safe_get("Win Rate [%]", 0.0) / 100,
            profit_factor=safe_get("Profit Factor", 0.0),
            total_trades=total_trades,
            avg_trade_duration=avg_duration,
            calmar_ratio=safe_get("Calmar Ratio", 0.0),
            volatility=safe_get("Annualized Volatility [%]", 0.0) / 100,
        )

    def _extract_trades(self, portfolio: vbt.Portfolio) -> pd.DataFrame:
        """Extract trades DataFrame from portfolio."""
        try:
            trades = portfolio.trades.records_readable
            if trades is None or len(trades) == 0:
                return pd.DataFrame()
            return trades
        except Exception:
            return pd.DataFrame()

    def run_single_backtest(
        self,
        price_data: pd.DataFrame | pd.Series,
        strategy_type: str = "sma_crossover",
        **params: int,
    ) -> OptimizationResult:
        """
        Run a single backtest with fixed parameters.

        Args:
            price_data: DataFrame with OHLCV columns or Series of close prices.
            strategy_type: Name of the strategy.
            **params: Fixed strategy parameters.

        Returns:
            OptimizationResult for the single parameter set.
        """
        logger.info(f"Running single backtest | Strategy: {strategy_type} | Params: {params}")

        close = self._extract_close(price_data)

        if len(close) < 50:
            raise InsufficientDataError(
                f"Insufficient data. Got {len(close)} rows, need at least 50."
            )

        strategy = StrategyRegistry.get(strategy_type)

        # Generate signals
        entries, exits = strategy.generate_signals(close, **params)

        # Run portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
        )

        # Extract metrics
        metrics = self._extract_metrics(portfolio)

        return OptimizationResult(
            best_parameters=StrategyParameters(
                strategy_type=strategy.name,
                parameters=params,
            ),
            metrics=metrics,
            price_data=price_data if isinstance(price_data, pd.DataFrame) else close.to_frame("close"),
            entries=entries,
            exits=exits,
            equity_curve=portfolio.value(),
            drawdown_curve=portfolio.drawdown(),
            trades=self._extract_trades(portfolio),
            portfolio=portfolio,
            optimization_space=None,
        )

    @staticmethod
    def list_strategies() -> list[str]:
        """List all available strategy types."""
        return StrategyRegistry.list_strategies()
