"""
Tests for the Analysis Layer.

Tests BacktestEngine (calculation) and StrategyVisualizer (presentation)
following strict separation of concerns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    BacktestEngine,
    OptimizationResult,
    PerformanceMetrics,
    StrategyParameters,
    StrategyVisualizer,
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    EMACrossoverStrategy,
    StrategyRegistry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    
    # Generate realistic price movement
    returns = np.random.randn(500) * 0.02
    close = 40000 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(500) * 0.001),
            "high": close * (1 + np.abs(np.random.randn(500) * 0.005)),
            "low": close * (1 - np.abs(np.random.randn(500) * 0.005)),
            "close": close,
            "volume": np.random.randint(100, 10000, 500).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def backtest_engine() -> BacktestEngine:
    """Create a BacktestEngine instance."""
    return BacktestEngine(initial_capital=10_000.0)


@pytest.fixture
def sample_optimization_result(sample_ohlcv_data: pd.DataFrame) -> OptimizationResult:
    """Generate a sample OptimizationResult for testing visualization."""
    close = sample_ohlcv_data["close"]
    dates = sample_ohlcv_data.index
    
    # Create synthetic equity curve
    equity = 10000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.005))
    equity_series = pd.Series(equity, index=dates)
    
    # Calculate drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    
    # Create synthetic entries/exits
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    entry_indices = np.random.choice(len(dates), size=20, replace=False)
    for idx in entry_indices:
        entries.iloc[idx] = True
        exit_idx = min(idx + np.random.randint(5, 20), len(dates) - 1)
        exits.iloc[exit_idx] = True
    
    # Create synthetic trades with Return column
    trades = pd.DataFrame({
        "Entry Time": dates[entry_indices[:10]],
        "Exit Time": dates[np.minimum(entry_indices[:10] + 10, len(dates) - 1)],
        "Return": np.random.randn(10) * 0.05,
        "PnL": np.random.randn(10) * 500,
    })
    
    return OptimizationResult(
        metrics=PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.08,
            volatility=0.12,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=20,
            avg_trade_duration=24.0,  # hours
        ),
        best_parameters=StrategyParameters(
            strategy_type="sma_crossover",
            parameters={"fast_window": 10, "slow_window": 30},
        ),
        equity_curve=equity_series,
        drawdown_curve=drawdown,
        trades=trades,
        entries=entries,
        exits=exits,
        price_data=sample_ohlcv_data,
        portfolio=None,  # Mock portfolio
        optimization_space=None,
    )


@pytest.fixture
def temp_reports_dir(tmp_path: Path) -> Path:
    """Create temporary reports directory."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return reports_dir


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_engine_initialization(self, backtest_engine: BacktestEngine) -> None:
        """Test engine initializes with correct parameters."""
        assert backtest_engine.initial_capital == 10_000.0
        assert backtest_engine.fees == 0.001

    def test_engine_initialization_custom_params(self) -> None:
        """Test engine with custom parameters."""
        engine = BacktestEngine(initial_capital=50_000.0, fees=0.002)
        assert engine.initial_capital == 50_000.0
        assert engine.fees == 0.002

    def test_run_optimization(
        self, backtest_engine: BacktestEngine, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test parameter optimization."""
        result = backtest_engine.run_optimization(
            price_data=sample_ohlcv_data,
            strategy_type="sma_crossover",
            fast_window=(5, 15, 5),
            slow_window=(20, 40, 10),
        )

        assert isinstance(result, OptimizationResult)
        assert result.best_parameters.strategy_type == "sma_crossover"
        assert "fast_window" in result.best_parameters.parameters
        assert "slow_window" in result.best_parameters.parameters

    def test_invalid_strategy_raises_error(
        self, backtest_engine: BacktestEngine, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test that invalid strategy name raises ValueError."""
        from src.analysis.engine import InvalidStrategyError
        
        with pytest.raises(InvalidStrategyError, match="Unknown strategy"):
            backtest_engine.run_optimization(
                price_data=sample_ohlcv_data,
                strategy_type="nonexistent_strategy",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategies:
    """Tests for strategy implementations."""

    def test_sma_crossover_signal_generation(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test SMA crossover generates valid signals."""
        strategy = SMACrossoverStrategy()
        close = sample_ohlcv_data["close"]
        entries, exits = strategy.generate_signals(
            close, fast_window=10, slow_window=30
        )

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)
        assert len(exits) == len(sample_ohlcv_data)

    def test_rsi_mean_reversion_signal_generation(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test RSI mean reversion generates valid signals."""
        strategy = RSIMeanReversionStrategy()
        close = sample_ohlcv_data["close"]
        entries, exits = strategy.generate_signals(
            close, rsi_window=14, oversold=30, overbought=70
        )

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)

    def test_ema_crossover_signal_generation(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """Test EMA crossover generates valid signals."""
        strategy = EMACrossoverStrategy()
        close = sample_ohlcv_data["close"]
        entries, exits = strategy.generate_signals(
            close, fast_window=12, slow_window=26
        )

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_list_registered_strategies(self) -> None:
        """Test listing registered strategies."""
        strategies = StrategyRegistry.list_strategies()
        
        assert "sma_crossover" in strategies
        assert "rsi_mean_reversion" in strategies
        assert "ema_crossover" in strategies

    def test_get_valid_strategy(self) -> None:
        """Test getting a valid strategy."""
        strategy = StrategyRegistry.get("sma_crossover")
        assert isinstance(strategy, SMACrossoverStrategy)

    def test_get_invalid_strategy_raises_error(self) -> None:
        """Test getting invalid strategy raises error."""
        from src.analysis.engine import InvalidStrategyError
        
        with pytest.raises(InvalidStrategyError, match="Unknown strategy"):
            StrategyRegistry.get("invalid_strategy")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyVisualizer:
    """Tests for StrategyVisualizer class."""

    def test_visualizer_initialization(self, temp_reports_dir: Path) -> None:
        """Test visualizer initializes correctly."""
        visualizer = StrategyVisualizer(output_dir=temp_reports_dir)
        
        assert visualizer.output_dir == temp_reports_dir
        assert visualizer.dpi == 150
        assert visualizer.figsize == (16, 12)

    def test_visualizer_creates_output_directory(self, tmp_path: Path) -> None:
        """Test visualizer creates output directory if not exists."""
        new_dir = tmp_path / "new_reports"
        assert not new_dir.exists()
        
        visualizer = StrategyVisualizer(output_dir=new_dir)
        assert new_dir.exists()

    def test_plot_performance_creates_file(
        self,
        temp_reports_dir: Path,
        sample_optimization_result: OptimizationResult,
    ) -> None:
        """Test plot_performance creates report file."""
        visualizer = StrategyVisualizer(output_dir=temp_reports_dir, dpi=72)
        
        metadata = visualizer.plot_performance(
            result=sample_optimization_result,
            filename="test_report",
        )

        report_path = temp_reports_dir / "test_report.png"
        assert report_path.exists()
        assert metadata.filepath == str(report_path)

    def test_plot_equity_comparison_creates_file(
        self,
        temp_reports_dir: Path,
        sample_optimization_result: OptimizationResult,
    ) -> None:
        """Test equity comparison plot creates file."""
        visualizer = StrategyVisualizer(output_dir=temp_reports_dir, dpi=72)
        
        metadata = visualizer.plot_equity_comparison(
            results=[sample_optimization_result, sample_optimization_result],
            labels=["Strategy A", "Strategy B"],
            filename="comparison_test",
        )

        report_path = temp_reports_dir / "comparison_test.png"
        assert report_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModels:
    """Tests for data transfer objects."""

    def test_performance_metrics_dataclass(self) -> None:
        """Test PerformanceMetrics dataclass."""
        metrics = PerformanceMetrics(
            total_return=0.1,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=0.1,
            volatility=0.15,
            win_rate=0.6,
            profit_factor=1.5,
            total_trades=100,
            avg_trade_duration=12.5,
        )

        assert metrics.total_return == 0.1
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_trades == 100

    def test_strategy_parameters_dataclass(self) -> None:
        """Test StrategyParameters dataclass."""
        params = StrategyParameters(
            strategy_type="sma_crossover",
            parameters={"fast_window": 10, "slow_window": 30},
        )

        assert params.strategy_type == "sma_crossover"
        assert params.parameters["fast_window"] == 10
