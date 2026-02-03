#!/usr/bin/env python3
"""
VectorBT Strategy Optimization Runner.

Bu script Windows'ta çalışır ve optimal EMA parametrelerini bulur.
Sonuçları config/live_params.json dosyasına kaydeder.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.analysis.engine import BacktestEngine, EMACrossoverStrategy
from src.data.loader import CryptoDataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_DAYS = 180  # 6 ay veri
INITIAL_CAPITAL = 10000
COMMISSION = 0.001  # 0.1% (Binance spot)

# Optimization ranges - EXPANDED for better coverage
# Test more strategies: fast trend-following, medium, and slow trend-following
FAST_RANGE = (5, 50, 1)      # (min, max, step) - more granular
SLOW_RANGE = (20, 200, 5)    # (min, max, step) - wider range


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run optimization and save results."""
    logger.info("=" * 80)
    logger.info("VECTORBT STRATEGY OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Timeframe: {TIMEFRAME}")
    logger.info(f"Lookback: {LOOKBACK_DAYS} days")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Step 1: Load Data
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("📥 Loading market data...")
    
    loader = CryptoDataLoader(
        exchange_id="binance",
        use_testnet=False,
    )

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    # Fetch data
    df = loader.fetch_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        since=start_date,
        until=end_date,
    )

    logger.success(f"✅ Loaded {len(df)} candles")
    logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Step 2: Initialize Backtest Engine
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("🔧 Initializing backtest engine...")
    
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        fees=COMMISSION,
        freq=TIMEFRAME,
    )

    logger.success(f"✅ Engine initialized")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Step 3: Run Optimization
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("🚀 Starting parameter optimization...")
    logger.info(f"   Fast EMA range: {FAST_RANGE}")
    logger.info(f"   Slow EMA range: {SLOW_RANGE}")
    
    # Calculate total combinations
    fast_steps = (FAST_RANGE[1] - FAST_RANGE[0]) // FAST_RANGE[2] + 1
    slow_steps = (SLOW_RANGE[1] - SLOW_RANGE[0]) // SLOW_RANGE[2] + 1
    total_combinations = fast_steps * slow_steps
    
    logger.info(f"   Total combinations: {total_combinations:,}")
    logger.info("")
    logger.info("⏳ This may take a few minutes...")
    logger.info("")

    # Run optimization
    result = engine.run_optimization(
        price_data=df["close"],
        strategy_type="ema_crossover",
        fast_window=FAST_RANGE,
        slow_window=SLOW_RANGE,
    )

    # ───────────────────────────────────────────────────────────────────────────
    # Step 4: Display Results
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info("")

    logger.info("🏆 BEST PARAMETERS:")
    best_fast = result.best_parameters.parameters["fast_window"]
    best_slow = result.best_parameters.parameters["slow_window"]
    logger.info(f"   Fast EMA: {best_fast}")
    logger.info(f"   Slow EMA: {best_slow}")
    logger.info("")

    logger.info("📊 PERFORMANCE METRICS:")
    metrics = result.metrics
    logger.info(f"   Total Return: {metrics.total_return:.2f}%")
    logger.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
    logger.info(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
    logger.info(f"   Win Rate: {metrics.win_rate:.2f}%")
    logger.info(f"   Total Trades: {metrics.total_trades}")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Step 5: Save Configuration
    # ───────────────────────────────────────────────────────────────────────────
    config_path = Path("config/live_params.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "strategy": "ema_crossover",
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "parameters": {
            "fast_period": best_fast,
            "slow_period": best_slow,
        },
        "performance": {
            "total_return": float(metrics.total_return),
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "sortino_ratio": float(metrics.sortino_ratio),
            "max_drawdown": float(metrics.max_drawdown),
            "win_rate": float(metrics.win_rate),
            "total_trades": int(metrics.total_trades),
        },
        "optimization_date": datetime.now().isoformat(),
        "data_range": {
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "candles": len(df),
        },
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.success(f"✅ Configuration saved to: {config_path}")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Step 6: Generate Report (OPTIONAL - commented out for now)
    # ───────────────────────────────────────────────────────────────────────────
    # report_path = Path(f"reports/optimization_{datetime.now():%Y%m%d_%H%M%S}.json")
    # report_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # report_data = {
    #     **config_data,
    #     "all_results": [
    #         {
    #             "fast_window": int(p.fast_window),
    #             "slow_window": int(p.slow_window),
    #             "total_return": float(m.total_return),
    #             "sharpe_ratio": float(m.sharpe_ratio),
    #             "max_drawdown": float(m.max_drawdown),
    #         }
    #         for p, m in zip(result.all_params, result.all_metrics)
    #     ],
    # }
    #
    # with open(report_path, "w") as f:
    #     json.dump(report_data, f, indent=2)
    #
    # logger.success(f"✅ Full report saved to: {report_path}")
    logger.info("")

    # ───────────────────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Review the results above")
    logger.info("2. If satisfied, upload config to server:")
    logger.info(f"   scp {config_path} hetzner:/opt/trading-bot/config/")
    logger.info("3. Restart Docker container:")
    logger.info('   ssh hetzner "docker restart algo_trading_bot"')
    logger.info("4. Monitor logs:")
    logger.info('   ssh hetzner "docker logs algo_trading_bot -f"')
    logger.info("")
    logger.success("🎉 Optimization complete!")


if __name__ == "__main__":
    main()
