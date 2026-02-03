"""
Strategy Visualizer - Pure Presentation Layer.

This module contains the StrategyVisualizer class responsible for
generating professional-grade reports and visualizations.

NO CALCULATION LOGIC ALLOWED IN THIS MODULE.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec

from src.analysis.models import OptimizationResult, ReportMetadata

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Professional color palette
COLORS = {
    "price": "#2962FF",
    "equity": "#00C853",
    "drawdown": "#FF1744",
    "buy": "#00E676",
    "sell": "#FF5252",
    "sma_fast": "#FF9800",
    "sma_slow": "#9C27B0",
    "grid": "#E0E0E0",
    "background": "#FAFAFA",
    "text": "#212121",
    "text_secondary": "#757575",
}

# Plot style configuration
STYLE_CONFIG = {
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlecolor": COLORS["text"],
    "xtick.color": COLORS["text_secondary"],
    "ytick.color": COLORS["text_secondary"],
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class StrategyVisualizer:
    """
    Professional-grade strategy visualization and reporting.

    This class is responsible for PURE PRESENTATION only.
    No calculation or trading logic should exist here.

    All data must come from OptimizationResult objects produced
    by the BacktestEngine.

    Example:
        >>> visualizer = StrategyVisualizer(output_dir="reports")
        >>> metadata = visualizer.plot_performance(
        ...     result=optimization_result,
        ...     filename="btc_sma_strategy",
        ... )
        >>> print(f"Report saved to: {metadata.filepath}")
    """

    def __init__(
        self,
        output_dir: str | Path = "reports",
        dpi: int = 150,
        figsize: tuple[int, int] = (16, 12),
    ) -> None:
        """
        Initialize the StrategyVisualizer.

        Args:
            output_dir: Directory to save reports.
            dpi: Resolution for saved figures.
            figsize: Default figure size (width, height).
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.figsize = figsize

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply style
        plt.rcParams.update(STYLE_CONFIG)

        logger.info(f"StrategyVisualizer initialized | Output: {self.output_dir}")

    def plot_performance(
        self,
        result: OptimizationResult,
        filename: str,
        show_signals: bool = True,
        show_trades: bool = True,
    ) -> ReportMetadata:
        """
        Generate a comprehensive performance report.

        Creates a multi-panel figure with:
        - Price chart with signals
        - Equity curve
        - Drawdown chart
        - Performance metrics table

        Args:
            result: OptimizationResult from BacktestEngine.
            filename: Output filename (without extension).
            show_signals: Whether to plot entry/exit signals.
            show_trades: Whether to annotate individual trades.

        Returns:
            ReportMetadata with report details and filepath.
        """
        logger.info(f"Generating performance report: {filename}")

        # Create figure with custom layout
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(4, 4, figure=fig, height_ratios=[3, 2, 1.5, 1])

        # Extract data
        close = self._get_close(result.price_data)

        # Panel 1: Price with Signals (top, spans 3 columns)
        ax_price = fig.add_subplot(gs[0, :3])
        self._plot_price_signals(
            ax=ax_price,
            close=close,
            entries=result.entries,
            exits=result.exits,
            show_signals=show_signals,
        )

        # Panel 2: Metrics Table (top right)
        ax_metrics = fig.add_subplot(gs[0, 3])
        self._plot_metrics_table(ax=ax_metrics, result=result)

        # Panel 3: Equity Curve (middle)
        ax_equity = fig.add_subplot(gs[1, :])
        self._plot_equity_curve(ax=ax_equity, equity=result.equity_curve)

        # Panel 4: Drawdown (lower)
        ax_drawdown = fig.add_subplot(gs[2, :])
        self._plot_drawdown(ax=ax_drawdown, drawdown=result.drawdown_curve)

        # Panel 5: Trade Distribution (bottom)
        ax_trades = fig.add_subplot(gs[3, :2])
        self._plot_trade_distribution(ax=ax_trades, trades=result.trades)

        # Panel 6: Parameter Heatmap (bottom right, if optimization data exists)
        ax_heatmap = fig.add_subplot(gs[3, 2:])
        self._plot_optimization_heatmap(ax=ax_heatmap, result=result)

        # Add title and metadata
        strategy_name = result.best_parameters.strategy_type.replace("_", " ").title()
        params_str = ", ".join(
            f"{k}={v}" for k, v in result.best_parameters.parameters.items()
        )
        title = f"{strategy_name} Strategy | {params_str}"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.success(f"Report saved: {filepath}")

        # Build metadata
        date_range = (
            close.index.min().to_pydatetime(),
            close.index.max().to_pydatetime(),
        )

        return ReportMetadata(
            title=title,
            symbol=result.symbol,
            timeframe=result.timeframe,
            date_range=date_range,
            generated_at=datetime.utcnow(),
            filepath=str(filepath),
        )

    def _get_close(self, price_data: pd.DataFrame | pd.Series) -> pd.Series:
        """Extract close price series."""
        if isinstance(price_data, pd.Series):
            return price_data
        if "close" in price_data.columns:
            return price_data["close"]
        if "Close" in price_data.columns:
            return price_data["Close"]
        return price_data.iloc[:, 0]

    def _plot_price_signals(
        self,
        ax: plt.Axes,
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        show_signals: bool = True,
    ) -> None:
        """Plot price chart with entry/exit signals."""
        # Plot price
        ax.plot(close.index, close.values, color=COLORS["price"], linewidth=1.2, label="Price")

        if show_signals:
            # Plot entry signals
            entry_mask = entries.fillna(False).astype(bool)
            if entry_mask.any():
                entry_points = close[entry_mask]
                ax.scatter(
                    entry_points.index,
                    entry_points.values,
                    marker="^",
                    color=COLORS["buy"],
                    s=80,
                    label="Buy",
                    zorder=5,
                    edgecolors="white",
                    linewidth=0.5,
                )

            # Plot exit signals
            exit_mask = exits.fillna(False).astype(bool)
            if exit_mask.any():
                exit_points = close[exit_mask]
                ax.scatter(
                    exit_points.index,
                    exit_points.values,
                    marker="v",
                    color=COLORS["sell"],
                    s=80,
                    label="Sell",
                    zorder=5,
                    edgecolors="white",
                    linewidth=0.5,
                )

        ax.set_title("Price & Signals", fontweight="bold", loc="left")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

    def _plot_equity_curve(self, ax: plt.Axes, equity: pd.Series) -> None:
        """Plot equity curve with fill."""
        ax.fill_between(
            equity.index,
            equity.values,
            alpha=0.3,
            color=COLORS["equity"],
        )
        ax.plot(
            equity.index,
            equity.values,
            color=COLORS["equity"],
            linewidth=1.5,
            label="Portfolio Value",
        )

        # Add starting capital line
        initial = equity.iloc[0]
        ax.axhline(y=initial, color=COLORS["text_secondary"], linestyle="--", alpha=0.5, label="Initial Capital")

        ax.set_title("Equity Curve", fontweight="bold", loc="left")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    def _plot_drawdown(self, ax: plt.Axes, drawdown: pd.Series) -> None:
        """Plot drawdown chart."""
        drawdown_pct = drawdown * 100

        ax.fill_between(
            drawdown_pct.index,
            drawdown_pct.values,
            0,
            alpha=0.4,
            color=COLORS["drawdown"],
        )
        ax.plot(
            drawdown_pct.index,
            drawdown_pct.values,
            color=COLORS["drawdown"],
            linewidth=1,
        )

        # Highlight max drawdown
        max_dd_idx = drawdown_pct.idxmin()
        max_dd_val = drawdown_pct.min()
        ax.scatter([max_dd_idx], [max_dd_val], color=COLORS["drawdown"], s=100, zorder=5)
        ax.annotate(
            f"Max DD: {max_dd_val:.1f}%",
            xy=(max_dd_idx, max_dd_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

        ax.set_title("Drawdown", fontweight="bold", loc="left")
        ax.set_ylabel("Drawdown (%)")
        ax.set_ylim(top=0)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

    def _plot_metrics_table(self, ax: plt.Axes, result: OptimizationResult) -> None:
        """Plot metrics as a formatted table."""
        ax.axis("off")

        metrics = result.metrics
        data = [
            ["Total Return", f"{metrics.total_return*100:+.2f}%"],
            ["Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}"],
            ["Sortino Ratio", f"{metrics.sortino_ratio:.3f}"],
            ["Max Drawdown", f"{metrics.max_drawdown*100:.2f}%"],
            ["Win Rate", f"{metrics.win_rate*100:.1f}%"],
            ["Profit Factor", f"{metrics.profit_factor:.2f}"],
            ["Total Trades", f"{metrics.total_trades}"],
            ["Volatility", f"{metrics.volatility*100:.1f}%"],
        ]

        table = ax.table(
            cellText=data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.6, 0.4],
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color header
        for j in range(2):
            table[(0, j)].set_facecolor(COLORS["price"])
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#F5F5F5")

        ax.set_title("Performance Metrics", fontweight="bold", loc="center", pad=20)

    def _plot_trade_distribution(self, ax: plt.Axes, trades: pd.DataFrame) -> None:
        """Plot trade return distribution."""
        if trades.empty or "Return" not in trades.columns:
            ax.text(
                0.5, 0.5, "No trades to display",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Trade Returns Distribution", fontweight="bold", loc="left")
            return

        returns = trades["Return"].dropna() * 100

        # Create histogram
        bins = np.linspace(returns.min(), returns.max(), 30)
        n, bins, patches = ax.hist(returns, bins=bins, edgecolor="white", alpha=0.7)

        # Color positive/negative
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor(COLORS["sell"])
            else:
                patch.set_facecolor(COLORS["buy"])

        # Add vertical line at zero
        ax.axvline(x=0, color=COLORS["text_secondary"], linestyle="--", alpha=0.5)

        # Add mean line
        mean_return = returns.mean()
        ax.axvline(x=mean_return, color=COLORS["price"], linestyle="-", linewidth=2, label=f"Mean: {mean_return:.1f}%")

        ax.set_title("Trade Returns Distribution", fontweight="bold", loc="left")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_optimization_heatmap(self, ax: plt.Axes, result: OptimizationResult) -> None:
        """Plot parameter optimization heatmap."""
        if result.optimization_space is None or len(result.optimization_space.columns) < 3:
            ax.text(
                0.5, 0.5, "Single parameter set\n(no optimization)",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            ax.set_title("Optimization Space", fontweight="bold", loc="center")
            return

        opt_df = result.optimization_space
        param_cols = [c for c in opt_df.columns if c not in ["sharpe_ratio", "total_return"]]

        if len(param_cols) < 2:
            ax.text(
                0.5, 0.5, "Need 2+ params for heatmap",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            return

        # Create pivot table for heatmap
        try:
            pivot = opt_df.pivot_table(
                values="sharpe_ratio",
                index=param_cols[0],
                columns=param_cols[1],
                aggfunc="mean",
            )

            # Plot heatmap
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")

            # Add labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.astype(int), fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(int), fontsize=8)

            ax.set_xlabel(param_cols[1].replace("_", " ").title())
            ax.set_ylabel(param_cols[0].replace("_", " ").title())

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Sharpe Ratio", fontsize=8)

            ax.set_title("Optimization Heatmap", fontweight="bold", loc="center")

        except Exception as e:
            logger.warning(f"Could not create heatmap: {e}")
            ax.text(
                0.5, 0.5, "Heatmap unavailable",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")

    def plot_equity_comparison(
        self,
        results: list[OptimizationResult],
        labels: list[str],
        filename: str,
    ) -> ReportMetadata:
        """
        Compare equity curves from multiple strategies.

        Args:
            results: List of OptimizationResult objects.
            labels: Labels for each result.
            filename: Output filename.

        Returns:
            ReportMetadata for the saved report.
        """
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)

        colors = plt.cm.tab10.colors

        for i, (result, label) in enumerate(zip(results, labels)):
            color = colors[i % len(colors)]
            equity_normalized = result.equity_curve / result.equity_curve.iloc[0] * 100
            ax.plot(
                equity_normalized.index,
                equity_normalized.values,
                label=f"{label} (Sharpe: {result.metrics.sharpe_ratio:.2f})",
                color=color,
                linewidth=1.5,
            )

        ax.axhline(y=100, color=COLORS["text_secondary"], linestyle="--", alpha=0.5)
        ax.set_title("Strategy Comparison - Normalized Equity Curves", fontweight="bold")
        ax.set_ylabel("Normalized Value (Starting = 100)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))

        plt.tight_layout()

        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.success(f"Comparison report saved: {filepath}")

        return ReportMetadata(
            title="Strategy Comparison",
            symbol="Multiple",
            timeframe="Various",
            date_range=(results[0].equity_curve.index.min(), results[0].equity_curve.index.max()),
            filepath=str(filepath),
        )
