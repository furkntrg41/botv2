"""
EMA Crossover Strategy for NautilusTrader.

This module implements an Exponential Moving Average crossover strategy
for live trading using NautilusTrader's event-driven architecture.

═══════════════════════════════════════════════════════════════════════════════
CROSS-PLATFORM COMPATIBILITY NOTE
═══════════════════════════════════════════════════════════════════════════════

NautilusTrader is a HIGH-PERFORMANCE trading framework written in Rust with
Python bindings. It has the following constraints:

1. LINUX ONLY: The core library (nautilus_trader) only compiles on Linux.
   Windows and macOS are NOT officially supported.

2. PYTHON VERSION: Requires Python 3.10-3.12 with specific build dependencies.

3. DEVELOPMENT WORKFLOW:
   - On Windows: We develop and test strategy LOGIC using mocked imports
   - On Linux/Docker: The actual nautilus_trader library is available

This file uses conditional imports to:
- Allow syntax checking and IDE support on Windows
- Enable full functionality when running in Docker/Linux

The TYPE_CHECKING pattern ensures:
- Type hints work correctly in IDEs
- No ImportError on Windows during development
- Full functionality in production (Linux)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.execution.config_loader import StrategyConfig, load_live_params


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL IMPORTS (Cross-Platform Compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

# TYPE_CHECKING is False at runtime, True during static analysis (mypy, IDE)
# This allows type hints without importing the actual module

if TYPE_CHECKING:
    # These imports are ONLY used for type hints (never executed at runtime)
    from nautilus_trader.indicators import ExponentialMovingAverage
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.orders import MarketOrder
    from nautilus_trader.trading.strategy import Strategy

# Runtime imports with graceful fallback
_NAUTILUS_AVAILABLE = False

try:
    # Attempt to import nautilus_trader (will succeed on Linux, fail on Windows)
    from nautilus_trader.indicators import ExponentialMovingAverage
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.orders import MarketOrder
    from nautilus_trader.trading.strategy import Strategy

    _NAUTILUS_AVAILABLE = True
    logger.debug("NautilusTrader imported successfully (Linux/Docker environment)")

except ImportError as e:
    # On Windows: Create mock classes to allow development without the library
    logger.warning(
        f"NautilusTrader not available: {e}. "
        "Using mock classes for Windows development."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MOCK CLASSES (Windows Development Only)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # These mock classes provide the same interface as NautilusTrader,
    # allowing code to be written, tested, and type-checked on Windows.
    # They are NOT used in production - only for development.

    class Strategy:  # type: ignore[no-redef]
        """
        Mock Strategy base class for Windows development.

        This mimics the NautilusTrader Strategy interface to allow
        code development without the actual library.
        """

        def __init__(self, config: Any = None) -> None:
            self.config = config
            self._indicators: list[Any] = []
            self.portfolio: Any = None
            self.clock: Any = None
            self.cache: Any = None

        def register_indicator_for_bars(
            self, bar_type: Any, indicator: Any
        ) -> None:
            """Register an indicator to receive bar updates."""
            self._indicators.append(indicator)

        def subscribe_bars(self, bar_type: Any) -> None:
            """Subscribe to bar data."""
            pass

        def submit_order(self, order: Any) -> None:
            """Submit an order (mock)."""
            logger.info(f"[MOCK] Order submitted: {order}")

    class ExponentialMovingAverage:  # type: ignore[no-redef]
        """Mock EMA indicator for Windows development."""

        def __init__(self, period: int) -> None:
            self.period = period
            self.value: float = 0.0
            self.initialized: bool = False
            self._values: list[float] = []

        def update_raw(self, value: float) -> None:
            """Update EMA with a new value."""
            self._values.append(value)
            if len(self._values) >= self.period:
                # Simple EMA approximation for mock
                alpha = 2 / (self.period + 1)
                if not self.initialized:
                    self.value = sum(self._values[-self.period:]) / self.period
                    self.initialized = True
                else:
                    self.value = alpha * value + (1 - alpha) * self.value

    class Bar:  # type: ignore[no-redef]
        """Mock Bar class for Windows development."""

        def __init__(
            self,
            open_: float = 0.0,
            high: float = 0.0,
            low: float = 0.0,
            close: float = 0.0,
            volume: float = 0.0,
        ) -> None:
            self.open = Decimal(str(open_))
            self.high = Decimal(str(high))
            self.low = Decimal(str(low))
            self.close = Decimal(str(close))
            self.volume = Decimal(str(volume))

    class BarType:  # type: ignore[no-redef]
        """Mock BarType class."""

        @classmethod
        def from_str(cls, value: str) -> BarType:
            return cls()

    class InstrumentId:  # type: ignore[no-redef]
        """Mock InstrumentId class."""

        @classmethod
        def from_str(cls, value: str) -> InstrumentId:
            return cls()

    class Instrument:  # type: ignore[no-redef]
        """Mock Instrument class."""

        id: InstrumentId = InstrumentId()

    class OrderSide:  # type: ignore[no-redef]
        """Mock OrderSide enum."""

        BUY = "BUY"
        SELL = "SELL"

    class MarketOrder:  # type: ignore[no-redef]
        """Mock MarketOrder class."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# EMA CROSSOVER STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════


class EMACrossStrategy(Strategy):
    """
    Exponential Moving Average Crossover Strategy.

    This strategy generates trading signals based on the crossover of
    two EMAs with different periods (fast and slow).

    Trading Logic:
        - ENTRY (BUY): Fast EMA crosses ABOVE Slow EMA
        - EXIT (SELL): Fast EMA crosses BELOW Slow EMA

    Configuration:
        Parameters are loaded from `config/live_params.json`, which is
        generated by the research pipeline (run_research.py) using VectorBT
        optimization.

    Architecture:
        ┌─────────────────────────────────────────────────────────────────┐
        │                        EMACrossStrategy                         │
        ├─────────────────────────────────────────────────────────────────┤
        │  on_start()                                                     │
        │    ├── Load config from live_params.json                        │
        │    ├── Initialize Fast EMA (period from config)                 │
        │    ├── Initialize Slow EMA (period from config)                 │
        │    └── Subscribe to bar data                                    │
        ├─────────────────────────────────────────────────────────────────┤
        │  on_bar(bar)                                                    │
        │    ├── Update EMAs with close price                             │
        │    ├── Check for crossover signals                              │
        │    │   ├── Fast > Slow AND was_below → BUY                      │
        │    │   └── Fast < Slow AND was_above → SELL                     │
        │    └── Submit market order if signal                            │
        └─────────────────────────────────────────────────────────────────┘

    Example:
        >>> # In NautilusTrader engine configuration
        >>> strategy = EMACrossStrategy(
        ...     config_path="config/live_params.json",
        ...     instrument_id="BTC/USDT.BINANCE",
        ... )
        >>> engine.add_strategy(strategy)
    """

    def __init__(
        self,
        config_path: str = "config/live_params.json",
        instrument_id: str | None = None,
        trade_size: float = 0.01,
    ) -> None:
        """
        Initialize the EMA Crossover Strategy.

        Args:
            config_path: Path to the live parameters JSON file.
            instrument_id: Trading instrument ID (e.g., "BTC/USDT.BINANCE").
                          If None, will be loaded from config.
            trade_size: Size of each trade in base currency units.
        """
        super().__init__()

        # Store initialization parameters
        self._config_path = config_path
        self._instrument_id_str = instrument_id
        self._trade_size = Decimal(str(trade_size))

        # Strategy state (initialized in on_start)
        self._config: StrategyConfig | None = None
        self._fast_ema: ExponentialMovingAverage | None = None
        self._slow_ema: ExponentialMovingAverage | None = None
        self._instrument: Instrument | None = None
        self._bar_type: BarType | None = None

        # Crossover tracking
        self._previous_fast: float = 0.0
        self._previous_slow: float = 0.0
        self._position_open: bool = False

        logger.info(
            f"EMACrossStrategy initialized | "
            f"Config: {config_path} | "
            f"Nautilus Available: {_NAUTILUS_AVAILABLE}"
        )

    def on_start(self) -> None:
        """
        Called when the strategy starts.

        This method:
        1. Loads optimized parameters from config file
        2. Initializes EMA indicators with those parameters
        3. Registers indicators to receive bar updates
        4. Subscribes to market data
        """
        logger.info("EMACrossStrategy starting...")

        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Load Configuration
        # ─────────────────────────────────────────────────────────────────
        try:
            self._config = load_live_params(self._config_path)
            logger.success(f"Configuration loaded: {self._config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Use default parameters as fallback
            self._config = StrategyConfig(
                strategy_type="ema_crossover",
                symbol="BTC/USDT",
                timeframe="1h",
                parameters={"fast_window": 12, "slow_window": 26},
            )
            logger.warning("Using default EMA parameters: fast=12, slow=26")

        # Extract EMA parameters
        ema_params = self._config.get_ema_params()
        fast_period = ema_params["fast_window"]
        slow_period = ema_params["slow_window"]

        logger.info(f"EMA Parameters | Fast: {fast_period} | Slow: {slow_period}")

        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Initialize Indicators
        # ─────────────────────────────────────────────────────────────────
        self._fast_ema = ExponentialMovingAverage(period=fast_period)
        self._slow_ema = ExponentialMovingAverage(period=slow_period)

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Setup Instrument and Bar Type (if using real Nautilus)
        # ─────────────────────────────────────────────────────────────────
        if _NAUTILUS_AVAILABLE and hasattr(self, 'cache') and self.cache is not None:
            # Get instrument from config or parameter
            instrument_id_str = self._instrument_id_str or f"{self._config.symbol}.BINANCE"
            instrument_id = InstrumentId.from_str(instrument_id_str)

            self._instrument = self.cache.instrument(instrument_id)
            if self._instrument is None:
                logger.error(f"Instrument not found: {instrument_id_str}")
                return

            # Create bar type for subscription
            bar_type_str = f"{instrument_id_str}-{self._config.timeframe}-LAST-EXTERNAL"
            self._bar_type = BarType.from_str(bar_type_str)

            # Register indicators to receive bar updates automatically
            self.register_indicator_for_bars(self._bar_type, self._fast_ema)
            self.register_indicator_for_bars(self._bar_type, self._slow_ema)

            # Subscribe to bar data
            self.subscribe_bars(self._bar_type)

            logger.info(f"Subscribed to bars: {bar_type_str}")

        logger.success("EMACrossStrategy started successfully")

    def on_bar(self, bar: Bar) -> None:
        """
        Called when a new bar is received.

        This method implements the core trading logic:
        1. Update EMA indicators with the new close price
        2. Check for crossover conditions
        3. Generate and submit orders if signals are detected

        Args:
            bar: The new bar data containing OHLCV values.
        """
        # Get close price as float for calculations
        close_price = float(bar.close)

        # Update indicators (may already be updated if registered)
        if not _NAUTILUS_AVAILABLE:
            # In mock mode, manually update indicators
            self._fast_ema.update_raw(close_price)
            self._slow_ema.update_raw(close_price)

        # Check if indicators are warmed up
        if not self._fast_ema.initialized or not self._slow_ema.initialized:
            logger.debug("EMAs not yet initialized, waiting for warmup...")
            return

        # Get current EMA values
        fast_value = self._fast_ema.value
        slow_value = self._slow_ema.value

        # ─────────────────────────────────────────────────────────────────
        # CROSSOVER DETECTION
        # ─────────────────────────────────────────────────────────────────
        #
        # Bullish Crossover (BUY signal):
        #   Previous: Fast < Slow
        #   Current:  Fast > Slow
        #   → Fast EMA crossed ABOVE Slow EMA
        #
        # Bearish Crossover (SELL signal):
        #   Previous: Fast > Slow
        #   Current:  Fast < Slow
        #   → Fast EMA crossed BELOW Slow EMA

        # Detect bullish crossover (Golden Cross)
        bullish_crossover = (
            self._previous_fast < self._previous_slow
            and fast_value > slow_value
            and self._previous_fast != 0  # Ensure we have previous values
        )

        # Detect bearish crossover (Death Cross)
        bearish_crossover = (
            self._previous_fast > self._previous_slow
            and fast_value < slow_value
            and self._previous_fast != 0
        )

        # ─────────────────────────────────────────────────────────────────
        # ORDER GENERATION
        # ─────────────────────────────────────────────────────────────────

        if bullish_crossover and not self._position_open:
            # Generate BUY signal
            logger.info(
                f"🟢 BULLISH CROSSOVER | "
                f"Fast EMA ({fast_value:.2f}) crossed above Slow EMA ({slow_value:.2f}) | "
                f"Price: {close_price:.2f}"
            )
            self._submit_market_order(OrderSide.BUY)
            self._position_open = True

        elif bearish_crossover and self._position_open:
            # Generate SELL signal
            logger.info(
                f"🔴 BEARISH CROSSOVER | "
                f"Fast EMA ({fast_value:.2f}) crossed below Slow EMA ({slow_value:.2f}) | "
                f"Price: {close_price:.2f}"
            )
            self._submit_market_order(OrderSide.SELL)
            self._position_open = False

        # Store current values for next comparison
        self._previous_fast = fast_value
        self._previous_slow = slow_value

    def _submit_market_order(self, side: OrderSide) -> None:
        """
        Submit a market order.

        Args:
            side: Order side (BUY or SELL).
        """
        if not _NAUTILUS_AVAILABLE or self._instrument is None:
            # Mock mode - just log the order
            logger.info(f"[MOCK ORDER] {side} {self._trade_size}")
            return

        # Create and submit market order
        order = self._instrument.make_market_order(
            strategy_id=self.id,
            side=side,
            quantity=self._trade_size,
        )

        self.submit_order(order)
        logger.info(f"Order submitted: {side} {self._trade_size}")

    def on_stop(self) -> None:
        """Called when the strategy stops."""
        logger.info("EMACrossStrategy stopped")

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self._previous_fast = 0.0
        self._previous_slow = 0.0
        self._position_open = False
        logger.info("EMACrossStrategy reset")
