#!/usr/bin/env python3
"""
NautilusTrader Runner Script.

This script runs INSIDE the Docker container where NautilusTrader is available.
It bridges the Research layer (VectorBT optimization) with the Execution layer
(NautilusTrader paper/live trading).

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         DOCKER CONTAINER (Linux)                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   ┌─────────────┐      ┌─────────────┐      ┌─────────────────────┐   │
    │   │ ArcticDB    │ ───► │ Data Bridge │ ───► │ NautilusTrader      │   │
    │   │ (Pandas DF) │      │ (to Bars)   │      │ BacktestEngine      │   │
    │   └─────────────┘      └─────────────┘      └──────────┬──────────┘   │
    │                                                        │              │
    │                                              ┌─────────▼─────────┐    │
    │   ┌─────────────┐                            │  EMACrossStrategy │    │
    │   │ live_params │ ─────────────────────────► │  (from config)    │    │
    │   │   .json     │                            └───────────────────┘    │
    │   └─────────────┘                                                     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PATH SETUP (For Docker environment)
# ═══════════════════════════════════════════════════════════════════════════════

# Add /app to Python path for Docker container
APP_DIR = Path("/app")
if APP_DIR.exists():
    sys.path.insert(0, str(APP_DIR))
else:
    # Local development - add parent directory
    sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP (Before any other imports)
# ═══════════════════════════════════════════════════════════════════════════════

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level=os.getenv("LOG_LEVEL", "INFO"),
    colorize=True,
)
logger.add(
    "logs/nautilus_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

# ═══════════════════════════════════════════════════════════════════════════════
# NAUTILUS TRADER IMPORTS (Linux Only)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD, USDT
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import (
        AccountType,
        AggregationSource,
        BarAggregation,
        OmsType,
        PriceType,
    )
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    NAUTILUS_AVAILABLE = True
    logger.success("NautilusTrader imported successfully!")

except ImportError as e:
    logger.error(f"NautilusTrader import failed: {e}")
    logger.error("This script must run inside the Docker container.")
    NAUTILUS_AVAILABLE = False
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd

from src.data import CryptoDataLoader
from src.execution.config_loader import StrategyConfig, load_live_params

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
EXCHANGE = "binanceus"
VENUE_NAME = "BINANCE"
LOOKBACK_DAYS = 30
INITIAL_CAPITAL = 10_000.0
CONFIG_PATH = "config/live_params.json"

# Position sizing (can be overridden via .env)
SIZING_MODE = os.getenv("SIZING_MODE", "risk_pct").lower()  # "risk_pct" or "fixed"
RISK_PCT = Decimal(os.getenv("RISK_PCT", "0.01"))
MAX_POSITION_PCT = Decimal(os.getenv("MAX_POSITION_PCT", "0.10"))
FIXED_TRADE_SIZE = Decimal(os.getenv("TRADE_SIZE", "0.01"))
MIN_TRADE_SIZE = Decimal(os.getenv("MIN_TRADE_SIZE", "0.000001"))
SIZE_INCREMENT = Decimal(os.getenv("SIZE_INCREMENT", "0.000001"))
SIZE_PRECISION = int(os.getenv("SIZE_PRECISION", "6"))

# Risk management
STOP_LOSS_PCT = Decimal(os.getenv("STOP_LOSS_PCT", "0.03"))
TAKE_PROFIT_PCT = Decimal(os.getenv("TAKE_PROFIT_PCT", "0.05"))
MAX_DRAWDOWN_PCT = Decimal(os.getenv("MAX_DRAWDOWN_PCT", "0.05"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "3"))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA BRIDGE: Pandas DataFrame → Nautilus Bar Objects
# ═══════════════════════════════════════════════════════════════════════════════


def create_instrument(symbol: str = "BTC/USDT") -> CurrencyPair:
    """
    Create a Nautilus CurrencyPair instrument.

    This defines the trading instrument with all necessary specifications
    for the backtesting/trading engine.
    """
    base, quote = symbol.replace("/", "").split("USDT")[0], "USDT"

    instrument_id = InstrumentId(
        symbol=Symbol(symbol.replace("/", "")),
        venue=Venue(VENUE_NAME),
    )

    # Use test provider for simplicity (production would use real specs)
    instrument = CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol.replace("/", "")),
        base_currency=USD,  # Simplified - would be BTC in production
        quote_currency=USDT,
        price_precision=2,
        size_precision=6,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.000001"),
        lot_size=Quantity.from_str("0.000001"),
        max_quantity=Quantity.from_str("9999.0"),
        min_quantity=Quantity.from_str("0.000001"),
        max_price=Price.from_str("1000000.0"),
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.02"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        ts_event=0,
        ts_init=0,
    )

    return instrument


def create_bar_type(instrument_id: InstrumentId, timeframe: str = "1h") -> BarType:
    """
    Create a Nautilus BarType specification.

    Maps common timeframe strings to Nautilus BarAggregation enums.
    """
    # Map timeframe string to aggregation
    timeframe_map = {
        "1m": (1, BarAggregation.MINUTE),
        "5m": (5, BarAggregation.MINUTE),
        "15m": (15, BarAggregation.MINUTE),
        "1h": (1, BarAggregation.HOUR),
        "4h": (4, BarAggregation.HOUR),
        "1d": (1, BarAggregation.DAY),
    }

    if timeframe not in timeframe_map:
        logger.warning(f"Unknown timeframe {timeframe}, defaulting to 1h")
        timeframe = "1h"

    step, aggregation = timeframe_map[timeframe]

    bar_spec = BarSpecification(
        step=step,
        aggregation=aggregation,
        price_type=PriceType.LAST,
    )

    return BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )


def convert_dataframe_to_bars(
    df: pd.DataFrame,
    bar_type: BarType,
) -> list[Bar]:
    """
    Convert a Pandas DataFrame to Nautilus Bar objects.

    This is the critical bridging function that converts data from
    ArcticDB (Pandas format) to NautilusTrader (native Bar format).

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex.
        bar_type: The BarType specification for these bars.

    Returns:
        List of Nautilus Bar objects.
    """
    bars = []

    logger.info(f"Converting {len(df)} rows to Nautilus Bars...")

    for idx, row in df.iterrows():
        # Convert timestamp to nanoseconds (Nautilus uses ns precision)
        ts_ns = int(pd.Timestamp(idx).value)

        try:
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{row['open']:.2f}"),
                high=Price.from_str(f"{row['high']:.2f}"),
                low=Price.from_str(f"{row['low']:.2f}"),
                close=Price.from_str(f"{row['close']:.2f}"),
                volume=Quantity.from_str(f"{row['volume']:.6f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)
        except Exception as e:
            logger.warning(f"Failed to convert bar at {idx}: {e}")
            continue

    logger.success(f"Converted {len(bars)} bars successfully")
    return bars


# ═══════════════════════════════════════════════════════════════════════════════
# EMA CROSSOVER STRATEGY (Nautilus Native)
# ═══════════════════════════════════════════════════════════════════════════════

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.strategy import Strategy


class NautilusEMACrossStrategy(Strategy):
    """
    EMA Crossover Strategy for NautilusTrader.

    This is the production strategy that runs inside NautilusTrader's
    event-driven engine. Parameters are loaded from config/live_params.json.
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        bar_type: BarType,
        fast_period: int = 12,
        slow_period: int = 26,
        trade_size: Decimal = Decimal("0.01"),
        sizing_mode: str = "risk_pct",
        risk_pct: Decimal = Decimal("0.01"),
        max_position_pct: Decimal = Decimal("0.10"),
        min_trade_size: Decimal = Decimal("0.000001"),
        size_increment: Decimal = Decimal("0.000001"),
        size_precision: int = 6,
        account_balance: Decimal = Decimal("10000"),
        stop_loss_pct: Decimal = Decimal("0.03"),
        take_profit_pct: Decimal = Decimal("0.05"),
        max_drawdown_pct: Decimal = Decimal("0.05"),
        cooldown_bars: int = 3,
    ) -> None:
        super().__init__()

        # Configuration
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        self.trade_size = trade_size
        self.sizing_mode = sizing_mode
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
        self.min_trade_size = min_trade_size
        self.size_increment = size_increment
        self.size_precision = size_precision
        self.account_balance = account_balance
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_bars = cooldown_bars

        # Indicators
        self.fast_ema = ExponentialMovingAverage(fast_period)
        self.slow_ema = ExponentialMovingAverage(slow_period)

        # State
        self._position_open = False
        self._position_size: Decimal | None = None
        self._entry_price: Decimal | None = None
        self._peak_price: Decimal | None = None
        self._cooldown_remaining = 0

        logger.info(
            f"NautilusEMACrossStrategy initialized | "
            f"Fast: {fast_period} | Slow: {slow_period} | "
            f"Sizing: {self.sizing_mode}"
        )

    def on_start(self) -> None:
        """Called when strategy starts."""
        # Register indicators to receive bar updates
        self.register_indicator_for_bars(self.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.bar_type, self.slow_ema)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        logger.info("Strategy started - subscribed to bars")

    def on_bar(self, bar: Bar) -> None:
        """Called on each new bar."""
        # Wait for indicators to warm up
        if not self.fast_ema.initialized or not self.slow_ema.initialized:
            return

        # Get current values
        fast_value = self.fast_ema.value
        slow_value = self.slow_ema.value

        price = Decimal(str(bar.close))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Risk management for open position
        if self._position_open and self._entry_price is not None and self._position_size is not None:
            self._peak_price = max(self._peak_price or self._entry_price, price)

            drawdown = (self._peak_price - price) / self._peak_price
            if drawdown >= self.max_drawdown_pct:
                logger.warning("Max drawdown hit, exiting position")
                self._exit_long(self._position_size)
                self._position_open = False
                self._position_size = None
                self._entry_price = None
                self._peak_price = None
                self._cooldown_remaining = self.cooldown_bars
                return

            if price <= self._entry_price * (Decimal("1") - self.stop_loss_pct):
                logger.warning("Stop-loss hit, exiting position")
                self._exit_long(self._position_size)
                self._position_open = False
                self._position_size = None
                self._entry_price = None
                self._peak_price = None
                self._cooldown_remaining = self.cooldown_bars
                return

            if price >= self._entry_price * (Decimal("1") + self.take_profit_pct):
                logger.info("Take-profit hit, exiting position")
                self._exit_long(self._position_size)
                self._position_open = False
                self._position_size = None
                self._entry_price = None
                self._peak_price = None
                self._cooldown_remaining = self.cooldown_bars
                return

        # Check for crossover
        if fast_value > slow_value and not self._position_open and self._cooldown_remaining == 0:
            # Bullish crossover - BUY
            trade_size = self._calculate_trade_size(price)
            if trade_size < self.min_trade_size:
                logger.warning(
                    f"Trade size too small ({trade_size}), skipping entry"
                )
                return
            self._enter_long(trade_size)
            self._position_open = True
            self._position_size = trade_size
            self._entry_price = price
            self._peak_price = price

        elif fast_value < slow_value and self._position_open:
            # Bearish crossover - SELL
            if self._position_size is None:
                logger.warning("No position size stored, skipping exit")
                return
            self._exit_long(self._position_size)
            self._position_open = False
            self._position_size = None
            self._entry_price = None
            self._peak_price = None
            self._cooldown_remaining = self.cooldown_bars

    def _enter_long(self, trade_size: Decimal) -> None:
        """Enter a long position."""
        size_str = self._format_trade_size(trade_size)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str(size_str),
        )
        self.submit_order(order)
        logger.info(f"🟢 BUY ORDER submitted: {size_str}")

    def _exit_long(self, trade_size: Decimal) -> None:
        """Exit long position."""
        size_str = self._format_trade_size(trade_size)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_str(size_str),
        )
        self.submit_order(order)
        logger.info(f"🔴 SELL ORDER submitted: {size_str}")

    def _calculate_trade_size(self, price: Decimal) -> Decimal:
        """Calculate trade size based on sizing mode and risk constraints."""
        if price <= 0:
            return Decimal("0")

        if self.sizing_mode == "fixed":
            raw_size = self.trade_size
        else:
            risk_amount = self.account_balance * self.risk_pct
            max_amount = self.account_balance * self.max_position_pct
            raw_size = min(risk_amount / price, max_amount / price)

        return self._round_down(raw_size)

    def _round_down(self, size: Decimal) -> Decimal:
        """Round size down to the nearest increment."""
        if self.size_increment <= 0:
            return size
        return (size / self.size_increment).to_integral_value(rounding=ROUND_DOWN) * self.size_increment

    def _format_trade_size(self, value: Decimal) -> str:
        """Format trade size with configured precision."""
        rounded = self._round_down(value)
        return f"{rounded:.{self.size_precision}f}"

    def on_stop(self) -> None:
        """Called when strategy stops."""
        logger.info("Strategy stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def load_config() -> tuple[int, int]:
    """
    Load strategy parameters from config file.

    Returns default values if config file is missing or invalid.
    """
    try:
        config = load_live_params(CONFIG_PATH)
        params = config.get_ema_params()
        logger.info(f"Loaded config: fast={params['fast_window']}, slow={params['slow_window']}")
        return params["fast_window"], params["slow_window"]
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return 12, 26  # Default EMA periods


def load_market_data() -> pd.DataFrame:
    """
    Load market data from ArcticDB or exchange.
    """
    logger.info(f"Loading market data for {SYMBOL}...")

    loader = CryptoDataLoader(exchange_id=EXCHANGE)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    try:
        # Try ArcticDB first
        data = loader.get_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date=start_date,
            end_date=end_date,
        )
        logger.success(f"Loaded {len(data)} rows from ArcticDB")
    except Exception:
        # Fetch from exchange
        logger.info("Fetching from exchange...")
        data = loader.fetch_data(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            limit=LOOKBACK_DAYS * 24,
        )
        loader.store_data(data, symbol=SYMBOL, timeframe=TIMEFRAME)
        logger.success(f"Fetched {len(data)} rows from exchange")

    return data


def run_backtest(
    bars: list[Bar],
    instrument: CurrencyPair,
    bar_type: BarType,
    fast_period: int,
    slow_period: int,
) -> None:
    """
    Run a backtest with NautilusTrader engine.
    """
    logger.info("=" * 80)
    logger.info("STARTING NAUTILUS TRADER BACKTEST")
    logger.info("=" * 80)

    # Configure engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
    )

    # Create engine
    engine = BacktestEngine(config=config)

    # Add venue
    engine.add_venue(
        venue=Venue(VENUE_NAME),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(INITIAL_CAPITAL, USDT)],
        fill_model=FillModel(),
    )

    # Add instrument
    engine.add_instrument(instrument)

    # Add data
    engine.add_data(bars)

    # Create and add strategy
    strategy = NautilusEMACrossStrategy(
        instrument_id=instrument.id,
        bar_type=bar_type,
        fast_period=fast_period,
        slow_period=slow_period,
        trade_size=FIXED_TRADE_SIZE,
        sizing_mode=SIZING_MODE,
        risk_pct=RISK_PCT,
        max_position_pct=MAX_POSITION_PCT,
        min_trade_size=MIN_TRADE_SIZE,
        size_increment=SIZE_INCREMENT,
        size_precision=SIZE_PRECISION,
        account_balance=Decimal(str(INITIAL_CAPITAL)),
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
        cooldown_bars=COOLDOWN_BARS,
    )
    engine.add_strategy(strategy)

    # Run backtest
    logger.info("Running backtest...")
    engine.run()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    # Get account info from cache directly
    try:
        accounts = engine.cache.accounts()
        for account in accounts:
            balance = account.balance_total(USDT)
            logger.info(f"💰 Account Balance: {balance}")

            # Calculate PnL
            pnl = float(balance) - INITIAL_CAPITAL
            pnl_pct = (pnl / INITIAL_CAPITAL) * 100
            logger.info(f"📈 Total PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    except Exception as e:
        logger.warning(f"Could not get account info: {e}")

    # Get statistics
    try:
        stats = engine.trader.generate_order_fills_report()
        logger.info(f"📊 Total Fills: {len(stats)}")
    except Exception as e:
        logger.warning(f"Could not generate fills report: {e}")

    # Cleanup
    engine.dispose()
    logger.success("Backtest complete!")


def main() -> None:
    """
    Main entry point for the Nautilus runner.
    """
    logger.info("=" * 80)
    logger.info("ALGO TRADING BOT - NAUTILUS RUNNER")
    logger.info(f"Mode: {os.getenv('TRADING_MODE', 'paper').upper()}")
    logger.info(f"Started at: {datetime.utcnow().isoformat()}")
    logger.info("=" * 80)

    # Step 1: Load config
    fast_period, slow_period = load_config()

    # Step 2: Load market data
    df = load_market_data()

    # Step 3: Create instrument and bar type
    instrument = create_instrument(SYMBOL)
    bar_type = create_bar_type(instrument.id, TIMEFRAME)

    # Step 4: Convert data to Nautilus format
    bars = convert_dataframe_to_bars(df, bar_type)

    if not bars:
        logger.error("No bars to process. Exiting.")
        sys.exit(1)

    # Step 5: Run backtest (paper trading simulation)
    run_backtest(
        bars=bars,
        instrument=instrument,
        bar_type=bar_type,
        fast_period=fast_period,
        slow_period=slow_period,
    )

    logger.info("=" * 80)
    logger.info("RUNNER COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Runner failed: {e}")
        sys.exit(1)
