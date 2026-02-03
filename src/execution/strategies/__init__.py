"""
Execution Strategies Module.

This module contains NautilusTrader strategy implementations.
These strategies bridge the Research layer (VectorBT optimization)
with the Execution layer (live trading via NautilusTrader).

Note:
    NautilusTrader requires Linux. On Windows, imports are mocked
    to allow code development and testing without the full library.
"""

# Import strategies with graceful degradation for Windows
try:
    from src.execution.strategies.ema_cross import EMACrossStrategy
    
    __all__ = ["EMACrossStrategy"]
except ImportError:
    # On Windows without nautilus_trader, expose empty __all__
    __all__ = []
