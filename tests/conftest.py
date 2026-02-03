"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_ohlcv_data() -> list[dict]:
    """Fixture providing sample OHLCV data for testing."""
    return [
        {"timestamp": 1704067200000, "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000.0},
        {"timestamp": 1704153600000, "open": 104.0, "high": 108.0, "low": 103.0, "close": 107.0, "volume": 1200.0},
    ]
