#!/usr/bin/env python3
"""
Research Pipeline Orchestrator.

This script demonstrates the complete research workflow:
1. Load historical data from ArcticDB
2. Run strategy optimization using BacktestEngine
3. Print structured metrics to console
4. Generate professional reports using StrategyVisualizer

This is the integration point that ties Data Layer and Analysis Layer together.

Usage:
    poetry run python run_research.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)
logger.add(
    "logs/research_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - Clean Architecture Layers
# ═══════════════════════════════════════════════════════════════════════════════

from src.data import CryptoDataLoader
from src.analysis import (
    BacktestEngine,
    StrategyVisualizer,
    OptimizationResult,
    PerformanceMetrics,
)
from src.execution.config_loader import save_live_params


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
EXCHANGE = "binanceus"
LOOKBACK_DAYS = 90
INITIAL_CAPITAL = 10_000.0
OOS_DAYS = int(os.getenv("OOS_DAYS", "90"))

# Parameter ranges for optimization (as tuples: min, max, step)
SMA_PARAM_RANGES = {
    "fast_window": (5, 25, 5),    # 5, 10, 15, 20, 25
    "slow_window": (20, 50, 10),  # 20, 30, 40, 50
}

RSI_PARAM_RANGES = {
    "rsi_window": (7, 21, 7),       # 7, 14, 21
    "oversold": (20, 35, 5),        # 20, 25, 30, 35
    "overbought": (65, 80, 5),      # 65, 70, 75, 80
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    width = 80
    logger.info("═" * width)
    logger.info(f"  {title.upper()}")
    logger.info("═" * width)


def print_metrics(metrics: PerformanceMetrics, strategy_name: str) -> None:
    """Print formatted performance metrics."""
    print_section_header(f"{strategy_name} Performance Metrics")
    
    metrics_data = [
        ("Total Return", f"{metrics.total_return * 100:+.2f}%"),
        ("Sharpe Ratio", f"{metrics.sharpe_ratio:.4f}"),
        ("Sortino Ratio", f"{metrics.sortino_ratio:.4f}"),
        ("Calmar Ratio", f"{metrics.calmar_ratio:.4f}"),
        ("Max Drawdown", f"{metrics.max_drawdown * 100:.2f}%"),
        ("Win Rate", f"{metrics.win_rate * 100:.1f}%"),
        ("Profit Factor", f"{metrics.profit_factor:.2f}"),
        ("Total Trades", f"{metrics.total_trades}"),
        ("Avg Trade Duration", f"{metrics.avg_trade_duration:.1f}h"),
        ("Volatility", f"{metrics.volatility * 100:.2f}%"),
    ]
    
    for name, value in metrics_data:
        logger.info(f"  {name:<20} : {value}")


def print_parameters(result: OptimizationResult) -> None:
    """Print optimal parameters."""
    print_section_header("Optimal Parameters")
    
    logger.info(f"  Strategy Type: {result.best_parameters.strategy_type}")
    logger.info(f"  Symbol: {result.symbol}")
    logger.info(f"  Timeframe: {result.timeframe}")
    
    for param, value in result.best_parameters.parameters.items():
        logger.info(f"  {param:<20} : {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run_research_pipeline() -> None:
    """
    Execute the complete research pipeline.
    
    Flow:
        1. Load Data (using CryptoDataLoader)
        2. Initialize BacktestEngine and run optimization
        3. Print structured metrics to console
        4. Initialize StrategyVisualizer and save the report
    """
    print_section_header("Algorithmic Trading Research Pipeline")
    logger.info(f"Started at: {datetime.utcnow().isoformat()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Data Loading
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Step 1: Data Loading")
    
    loader = CryptoDataLoader(exchange_id=EXCHANGE)
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    logger.info(f"Loading data for {SYMBOL} from {start_date.date()} to {end_date.date()}")
    
    # Try to load from ArcticDB first, fetch if not available
    try:
        data = loader.get_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=start_date,
            end_date=end_date,
        )
        logger.success(f"Loaded {len(data)} rows from ArcticDB")
    except Exception as e:
        logger.warning(f"Data not in ArcticDB, fetching from exchange: {e}")
        data = loader.fetch_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            limit=LOOKBACK_DAYS * 24,  # hourly data
        )
        loader.store_data(data, symbol=SYMBOL, timeframe=TIMEFRAME)
        logger.success(f"Fetched and stored {len(data)} rows")
    
    logger.info(f"Data range: {data.index.min()} to {data.index.max()}")
    logger.info(f"Data shape: {data.shape}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Strategy Optimization (Train) + OOS Split
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Step 2: Strategy Optimization")

    timeframe_hours = {
        "1h": 1,
        "4h": 4,
        "1d": 24,
    }.get(TIMEFRAME, 1)
    bars_per_day = max(1, int(24 / timeframe_hours))
    oos_bars = min(len(data) // 3, OOS_DAYS * bars_per_day)

    if oos_bars < 50:
        logger.warning("OOS split too small; running full-sample optimization")
        train_data = data
        oos_data = None
    else:
        train_data = data.iloc[:-oos_bars]
        oos_data = data.iloc[-oos_bars:]
        logger.info(
            f"Train bars: {len(train_data)} | OOS bars: {len(oos_data)}"
        )

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)

    # Run SMA Crossover optimization (train)
    logger.info("Running SMA Crossover optimization (train)...")
    sma_result = engine.run_optimization(
        price_data=train_data,
        strategy_type="sma_crossover",
        **SMA_PARAM_RANGES,
    )
    logger.success(f"SMA optimization complete. Best Sharpe: {sma_result.metrics.sharpe_ratio:.4f}")

    # Run RSI Mean Reversion optimization (train)
    logger.info("Running RSI Mean Reversion optimization (train)...")
    rsi_result = engine.run_optimization(
        price_data=train_data,
        strategy_type="rsi_mean_reversion",
        **RSI_PARAM_RANGES,
    )
    logger.success(f"RSI optimization complete. Best Sharpe: {rsi_result.metrics.sharpe_ratio:.4f}")

    # Determine best strategy (train)
    best_result = max([sma_result, rsi_result], key=lambda r: r.metrics.sharpe_ratio)
    best_strategy_name = best_result.best_parameters.strategy_type
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Print Metrics
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Step 3: Performance Analysis")
    
    print_metrics(sma_result.metrics, "SMA Crossover")
    print_parameters(sma_result)
    
    print_metrics(rsi_result.metrics, "RSI Mean Reversion")
    print_parameters(rsi_result)
    
    # Best strategy summary
    print_section_header("Best Strategy Summary")
    logger.info(f"  Winner: {best_strategy_name.upper()}")
    logger.info(f"  Sharpe Ratio: {best_result.metrics.sharpe_ratio:.4f}")
    logger.info(f"  Total Return: {best_result.metrics.total_return * 100:+.2f}%")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: OOS Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if oos_data is not None:
        print_section_header("Step 4: OOS Evaluation")
        oos_result = engine.run_single_backtest(
            price_data=oos_data,
            strategy_type=best_strategy_name,
            **best_result.best_parameters.parameters,
        )
        print_metrics(oos_result.metrics, f"{best_strategy_name} (OOS)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Generate Reports
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Step 5: Report Generation")
    
    visualizer = StrategyVisualizer(output_dir="reports")
    
    # Generate individual reports
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    sma_report = visualizer.plot_performance(
        result=sma_result,
        filename=f"sma_crossover_{timestamp}",
    )
    logger.success(f"SMA report saved: {sma_report.filepath}")
    
    rsi_report = visualizer.plot_performance(
        result=rsi_result,
        filename=f"rsi_mean_reversion_{timestamp}",
    )
    logger.success(f"RSI report saved: {rsi_report.filepath}")
    
    # Generate comparison report
    comparison_report = visualizer.plot_equity_comparison(
        results=[sma_result, rsi_result],
        labels=["SMA Crossover", "RSI Mean Reversion"],
        filename=f"strategy_comparison_{timestamp}",
    )
    logger.success(f"Comparison report saved: {comparison_report.filepath}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Export Best Parameters for Live Trading
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Step 5: Configuration Export")
    
    # Save best parameters to config/live_params.json
    # This bridges Research → Execution layers
    config_path = save_live_params(
        strategy_type=best_result.best_parameters.strategy_type,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        parameters=best_result.best_parameters.parameters,
        metrics=best_result.metrics.to_dict(),
        sharpe_ratio=best_result.metrics.sharpe_ratio,
        output_path="config/live_params.json",
    )
    logger.success(f"Best parameters exported for live trading: {config_path}")
    
    # Log the exported configuration
    logger.info(f"  Strategy: {best_result.best_parameters.strategy_type}")
    logger.info(f"  Symbol: {SYMBOL}")
    logger.info(f"  Timeframe: {TIMEFRAME}")
    for param, value in best_result.best_parameters.parameters.items():
        logger.info(f"  {param}: {value}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE COMPLETE
    # ─────────────────────────────────────────────────────────────────────────
    print_section_header("Pipeline Complete")
    logger.info(f"Completed at: {datetime.utcnow().isoformat()}")
    logger.info(f"Reports directory: {Path('reports').absolute()}")
    logger.info(f"Config directory: {Path('config').absolute()}")
    logger.info("─" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_research_pipeline()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
