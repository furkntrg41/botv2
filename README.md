# Algo Trading Bot

Institutional-grade hybrid pipeline algorithmic trading system.

## Architecture

- **Research Layer:** VectorBT for high-speed vectorized backtesting
- **Data Layer:** CCXT + ArcticDB for time-series storage
- **Execution Layer:** NautilusTrader for low-latency live trading

## Setup

```bash
poetry install
poetry shell
```

## Project Structure

```
algo_trading_bot/
├── src/
│   ├── data/        # ArcticDB and CCXT adapters
│   ├── analysis/    # VectorBT research modules
│   ├── execution/   # NautilusTrader strategies
│   ├── models/      # ONNX/AI models
│   └── utils/       # Logging and helpers
├── tests/
├── scripts/
└── notebooks/
```
