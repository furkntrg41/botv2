#!/usr/bin/env python3
"""
Fast VectorBT Strategy Optimization - Minimal version.
Runs in 2-3 minutes instead of 30+.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from src.data.loader import CryptoDataLoader

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_DAYS = 180
INITIAL_CAPITAL = 10000
COMMISSION = 0.001

# Parameter ranges to test
FAST_PARAMS = list(range(5, 51, 2))      # 5, 7, 9, ..., 49 = 23 values
SLOW_PARAMS = list(range(20, 201, 10))   # 20, 30, ..., 200 = 19 values
# Total: 23 * 19 = 437 combinations (fast vs slow only ones)


def calculate_ema_returns(price_series: pd.Series, fast: int, slow: int) -> dict:
    """Calculate returns for one EMA pair combination."""
    try:
        # Calculate EMAs
        fast_ema = price_series.ewm(span=fast).mean()
        slow_ema = price_series.ewm(span=slow).mean()
        
        # Generate signals
        crossover_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        crossover_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        
        # Count trades
        entries = crossover_up.sum()
        exits = crossover_down.sum()
        num_trades = min(entries, exits)
        
        if num_trades == 0:
            return {
                "fast": fast,
                "slow": slow,
                "trades": 0,
                "return_pct": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "score": -999,
            }
        
        # Simple buy-and-hold long returns calculation
        positions = pd.Series(0, index=price_series.index)
        entry_price = None
        pnl_list = []
        
        for i, (date, entry_signal) in enumerate(crossover_up.items()):
            if entry_signal and entry_price is None:
                entry_price = price_series.loc[date]
                positions.loc[date:] = 1
            
            exit_signal = crossover_down.loc[date]
            if exit_signal and entry_price is not None:
                exit_price = price_series.loc[date]
                pnl = ((exit_price - entry_price) / entry_price - COMMISSION) * 100
                pnl_list.append(pnl)
                entry_price = None
                positions.loc[date:] = 0
        
        # Calculate metrics
        total_return = sum(pnl_list) if pnl_list else 0.0
        win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100 if pnl_list else 0
        sharpe = np.mean(pnl_list) / (np.std(pnl_list) + 0.001) if pnl_list else 0
        
        # Composite score (prefer: high return, high sharpe, high win rate)
        score = total_return * 0.5 + sharpe * 10 + win_rate * 0.2
        
        return {
            "fast": fast,
            "slow": slow,
            "trades": num_trades,
            "return_pct": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "score": score,
        }
    except Exception as e:
        logger.warning(f"Error for {fast}/{slow}: {e}")
        return {
            "fast": fast,
            "slow": slow,
            "trades": 0,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "score": -999,
        }


def main() -> None:
    """Run fast optimization."""
    logger.info("=" * 80)
    logger.info("FAST EMA OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Lookback: {LOOKBACK_DAYS} days")
    logger.info(f"Parameter combinations: {len(FAST_PARAMS)} × {len(SLOW_PARAMS)} = {len(FAST_PARAMS) * len(SLOW_PARAMS)}")
    logger.info("")

    # Load data
    logger.info("📥 Loading data...")
    loader = CryptoDataLoader(exchange_id="binance", use_testnet=False)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    df = loader.fetch_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        since=start_date,
        until=end_date,
    )
    
    logger.success(f"✅ Loaded {len(df)} candles")
    logger.info("")

    # Optimize
    logger.info("🚀 Running optimization (this takes 2-3 minutes)...")
    results = []
    total = len(FAST_PARAMS) * len(SLOW_PARAMS)
    count = 0
    
    for slow in SLOW_PARAMS:
        for fast in FAST_PARAMS:
            if fast >= slow:  # Skip invalid combinations
                continue
            
            result = calculate_ema_returns(df["close"], fast, slow)
            results.append(result)
            
            count += 1
            if count % 50 == 0:
                logger.info(f"  Progress: {count}/{total} combinations tested")
    
    logger.info(f"  Progress: {count}/{total} combinations tested")
    logger.success("✅ Optimization complete")
    logger.info("")

    # Find best
    best = max(results, key=lambda x: x["score"])
    
    logger.info("=" * 80)
    logger.info("🏆 BEST PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"Fast EMA: {best['fast']}")
    logger.info(f"Slow EMA: {best['slow']}")
    logger.info("")
    logger.info("📊 PERFORMANCE:")
    logger.info(f"  Total Return: {best['return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    logger.info(f"  Win Rate: {best['win_rate']:.1f}%")
    logger.info(f"  Total Trades: {best['trades']}")
    logger.info("")

    # Top 10
    logger.info("=" * 80)
    logger.info("🔝 TOP 10 COMBINATIONS")
    logger.info("=" * 80)
    top_10 = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    for i, r in enumerate(top_10, 1):
        logger.info(
            f"{i:2d}. EMA({r['fast']:2d}/{r['slow']:3d}) → "
            f"Return: {r['return_pct']:7.2f}% | "
            f"Sharpe: {r['sharpe']:6.2f} | "
            f"WinRate: {r['win_rate']:5.1f}% | "
            f"Score: {r['score']:7.2f}"
        )
    logger.info("")

    # Save config
    config_path = Path("config/live_params.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "strategy": "ema_crossover",
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "parameters": {
            "fast_period": best["fast"],
            "slow_period": best["slow"],
        },
        "performance": {
            "total_return": best["return_pct"],
            "sharpe_ratio": best["sharpe"],
            "win_rate": best["win_rate"],
            "total_trades": best["trades"],
        },
        "optimization_date": datetime.now().isoformat(),
        "data_range": {
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "candles": len(df),
        },
        "top_10_combinations": [
            {
                "fast": r["fast"],
                "slow": r["slow"],
                "return_pct": r["return_pct"],
                "sharpe": r["sharpe"],
                "win_rate": r["win_rate"],
            }
            for r in top_10
        ],
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.success(f"✅ Config saved to: {config_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Review the top 10 results above")
    logger.info("2. Deploy to server:")
    logger.info(f"   scp {config_path} hetzner:/opt/trading-bot/config/")
    logger.info('3. Restart: ssh hetzner "docker restart algo_trading_bot"')
    logger.info("")


if __name__ == "__main__":
    main()
