import json
import re
from pathlib import Path

import matplotlib
# Force the non-interactive Agg backend only when running outside a Jupyter/IPython
# kernel. In notebooks the inline backend is already configured and must not be
# overridden, otherwise plt.show() produces no output.
try:
    from IPython import get_ipython as _get_ipython
    _in_notebook = _get_ipython() is not None
except ImportError:
    _in_notebook = False

if not _in_notebook:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _sanitize(name: str) -> str:
    """Convert a strategy name into a safe filename stem."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 1. NAV plots ─────────────────────────────────────────────────────────────

def save_nav_plot(
    results: list[dict],
    output_dir: str,
    filename: str,
    title: str = "NAV Curves",
) -> None:
    """Save a combined NAV plot and one individual plot per result.

    Args:
        results: List of result dicts from run_backtest().
        output_dir: Directory where PNG files are written (created if absent).
        filename: Stem for the combined figure, e.g. "single_stock_nav".
            Saved as {output_dir}/{filename}.png
        title: Title displayed on the combined figure.
    """
    out = _ensure_dir(output_dir)

    # ── combined figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for res in results:
        nav = res["nav_series"]
        ax.plot(nav.index, nav.values, label=res["strategy_name"], linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (normalized to 1.0)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    combined_path = out / f"{filename}.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reporter] saved combined NAV plot → {combined_path}")

    # ── individual figures ───────────────────────────────────────────────────
    for res in results:
        nav = res["nav_series"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(nav.index, nav.values, linewidth=1.5, color="steelblue")
        ax.set_title(res["strategy_name"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV (normalized to 1.0)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        stem = _sanitize(res["strategy_name"])
        path = out / f"{stem}_nav.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[reporter] saved individual NAV plot → {path}")


# ── 2. Metrics table ─────────────────────────────────────────────────────────

def save_metrics_table(
    results: list[dict],
    output_dir: str,
    filename: str,
) -> None:
    """Save a metrics comparison table as CSV and as a PNG image.

    Also prints the table to console.

    Args:
        results: List of result dicts from run_backtest().
        output_dir: Directory where files are written (created if absent).
        filename: Stem for output files, e.g. "single_stock_metrics".
            Produces {filename}.csv and {filename}_table.png.
    """
    out = _ensure_dir(output_dir)

    rows = []
    for res in results:
        m = res["metrics"]
        rows.append({
            "Strategy": res["strategy_name"],
            "Cumulative Return": f"{m['cumulative_return']:+.2%}",
            "Annual Volatility": f"{m['annualized_volatility']:.2%}",
            "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{-m['max_drawdown']:+.2%}",
            "Win Rate": f"{m['win_rate']:.2%}",
        })

    df = pd.DataFrame(rows)

    # Console print
    print(f"\n{'─' * 80}")
    print(df.to_string(index=False))
    print(f"{'─' * 80}\n")

    # CSV
    csv_path = out / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[reporter] saved metrics CSV → {csv_path}")

    # PNG table via matplotlib
    n_rows, n_cols = df.shape
    fig_h = max(1.2, 0.45 * (n_rows + 1))
    fig_w = max(8, 1.6 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(n_cols)))

    # Style header row
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for row_idx in range(1, n_rows + 1):
        color = "#f0f4f8" if row_idx % 2 == 0 else "white"
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(color)

    fig.tight_layout()
    table_path = out / f"{filename}_table.png"
    fig.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reporter] saved metrics table PNG → {table_path}")


# ── 3. Experiment log ─────────────────────────────────────────────────────────

def save_experiment_log(
    results: list[dict],
    output_dir: str,
    filename: str,
) -> None:
    """Save a JSON log of all experiment parameters and metrics.

    Args:
        results: List of result dicts from run_backtest().
        output_dir: Directory where the log is written (created if absent).
        filename: Stem for the log file, e.g. "single_stock_log".
            Saved as {output_dir}/{filename}.json.
    """
    out = _ensure_dir(output_dir)

    log_entries = []
    for res in results:
        nav = res["nav_series"]
        entry = {
            "strategy_name": res["strategy_name"],
            "transaction_cost": res["transaction_cost"],
            "date_range": {
                "start": str(nav.index[0].date()),
                "end": str(nav.index[-1].date()),
                "trading_days": len(nav),
            },
            "metrics": res["metrics"],
        }
        log_entries.append(entry)

    log_path = out / f"{filename}.json"
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    print(f"[reporter] saved experiment log → {log_path}")
