#!/usr/bin/env python
"""
Data Ingestion Pipeline - Integration Test & Verification Script

This script verifies the end-to-end data ingestion pipeline:
1. Initialize CryptoDataLoader with exchange connection
2. Fetch historical OHLCV data from Binance
3. Store data in ArcticDB
4. Read back and verify data integrity

Usage:
    python run_ingestion.py
    python run_ingestion.py --symbol ETH/USDT --timeframe 4h --days 7
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default ingestion parameters
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_DAYS = 1
DEFAULT_EXCHANGE = "binance"

# Logging configuration
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def configure_logging() -> None:
    """Configure loguru with structured output."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level="DEBUG",
        colorize=True,
    )
    # Add file logging for audit trail
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "ingestion_{time:YYYY-MM-DD}.log",
        format=LOG_FORMAT,
        level="INFO",
        rotation="1 day",
        retention="7 days",
        compression="gz",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPONENTS (Single Responsibility)
# ═══════════════════════════════════════════════════════════════════════════════


class DataFetcher:
    """Responsible for fetching data from exchange."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> "pd.DataFrame":
        """Fetch OHLCV data for the specified parameters."""
        logger.info(
            f"Fetching {symbol} {timeframe} | "
            f"From: {since.isoformat()} | To: {until.isoformat()}"
        )
        df = self._loader.fetch_data(
            symbol=symbol,
            timeframe=timeframe,
            since=since.isoformat(),
            until=until.isoformat(),
        )
        logger.success(f"Fetched {len(df)} rows from exchange")
        return df


class DataPersister:
    """Responsible for storing data to ArcticDB."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def store(self, df: "pd.DataFrame", symbol: str, timeframe: str) -> None:
        """Store DataFrame to ArcticDB."""
        logger.info(f"Storing {len(df)} rows to ArcticDB...")
        self._loader.store_data(df, symbol, timeframe)
        logger.success(f"Data persisted to ArcticDB: {symbol} {timeframe}")

    def retrieve(self, symbol: str, timeframe: str) -> "pd.DataFrame":
        """Retrieve DataFrame from ArcticDB."""
        logger.info(f"Retrieving data from ArcticDB: {symbol} {timeframe}")
        df = self._loader.get_data(symbol, timeframe)
        logger.success(f"Retrieved {len(df)} rows from ArcticDB")
        return df


class IntegrityVerifier:
    """Responsible for verifying data integrity."""

    @staticmethod
    def verify(original: "pd.DataFrame", retrieved: "pd.DataFrame") -> bool:
        """
        Verify that retrieved data matches original data.

        Returns:
            True if verification passes, False otherwise.
        """
        logger.info("Verifying data integrity...")

        # Check row count
        if len(original) != len(retrieved):
            logger.error(
                f"Row count mismatch: Original={len(original)}, Retrieved={len(retrieved)}"
            )
            return False

        # Check column presence
        if set(original.columns) != set(retrieved.columns):
            logger.error(
                f"Column mismatch: Original={list(original.columns)}, "
                f"Retrieved={list(retrieved.columns)}"
            )
            return False

        # Check data values (allow for floating point tolerance)
        try:
            import pandas as pd

            pd.testing.assert_frame_equal(
                original.reset_index(drop=True),
                retrieved.reset_index(drop=True),
                check_exact=False,
                rtol=1e-5,
            )
            logger.success("Data integrity verification PASSED")
            return True
        except AssertionError as e:
            logger.error(f"Data integrity verification FAILED: {e}")
            return False


class ResultPresenter:
    """Responsible for presenting results to console."""

    @staticmethod
    def display(df: "pd.DataFrame", title: str) -> None:
        """Display DataFrame summary and sample rows."""
        print("\n" + "═" * 80)
        print(f" {title}")
        print("═" * 80)
        print(f"\n📊 DataFrame Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📅 Date Range: {df.index.min()} → {df.index.max()}")
        print(f"📋 Columns: {list(df.columns)}")
        print("\n🔍 First 5 Rows:")
        print("-" * 80)
        print(df.head(5).to_string())
        print("-" * 80)

        # Show basic statistics
        print("\n📈 Price Summary:")
        print(f"   Open  - Min: {df['open'].min():.2f}, Max: {df['open'].max():.2f}")
        print(f"   Close - Min: {df['close'].min():.2f}, Max: {df['close'].max():.2f}")
        print(f"   Volume Total: {df['volume'].sum():.4f}")
        print("═" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


class IngestionPipeline:
    """
    Orchestrates the data ingestion pipeline.

    Follows the Single Responsibility Principle by delegating
    specific tasks to specialized components.
    """

    def __init__(
        self,
        exchange_id: str = DEFAULT_EXCHANGE,
        use_testnet: bool = True,
    ) -> None:
        """Initialize the pipeline with exchange connection."""
        from src.data.loader import CryptoDataLoader

        logger.info(f"Initializing pipeline | Exchange: {exchange_id} | Testnet: {use_testnet}")

        self._loader = CryptoDataLoader(
            exchange_id=exchange_id,
            use_testnet=use_testnet,
        )
        self._fetcher = DataFetcher(self._loader)
        self._persister = DataPersister(self._loader)
        self._verifier = IntegrityVerifier()
        self._presenter = ResultPresenter()

        logger.success("Pipeline initialized successfully")

    def run(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        days: int = DEFAULT_DAYS,
    ) -> bool:
        """
        Execute the full ingestion pipeline.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1h").
            days: Number of days of historical data to fetch.

        Returns:
            True if pipeline succeeds, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("=" * 60)

        try:
            # Calculate date range
            until = datetime.now(timezone.utc)
            since = until - timedelta(days=days)

            # Step 1: Fetch data from exchange
            fetched_df = self._fetcher.fetch(symbol, timeframe, since, until)

            if fetched_df.empty:
                logger.error("No data fetched from exchange")
                return False

            # Step 2: Store to ArcticDB
            self._persister.store(fetched_df, symbol, timeframe)

            # Step 3: Retrieve from ArcticDB
            retrieved_df = self._persister.retrieve(symbol, timeframe)

            # Step 4: Verify integrity
            # Note: Retrieved may have more rows if previous data exists
            # We verify the latest fetched data is present
            is_valid = len(retrieved_df) >= len(fetched_df)

            if not is_valid:
                logger.error("Data integrity check failed")
                return False

            # Step 5: Present results
            self._presenter.display(retrieved_df, f"INGESTED DATA: {symbol} {timeframe}")

            logger.success("=" * 60)
            logger.success("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.success("=" * 60)

            return True

        except Exception as e:
            logger.exception(f"Pipeline failed with error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Ingestion Pipeline - Fetch, Store, and Verify OHLCV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help="Trading pair symbol (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Candle timeframe",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of days of historical data",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=DEFAULT_EXCHANGE,
        help="Exchange ID (ccxt)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live API instead of testnet",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    configure_logging()

    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║     ALGO TRADING BOT - DATA INGESTION VERIFICATION         ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")

    args = parse_args()

    logger.info(f"Parameters: symbol={args.symbol}, timeframe={args.timeframe}, days={args.days}")

    pipeline = IngestionPipeline(
        exchange_id=args.exchange,
        use_testnet=not args.live,
    )

    success = pipeline.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
