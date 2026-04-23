"""
Build comparison charts from saved console output:
  - results.txt          (optimizer best-params leave-one-week-out table)
  - baseline_results.txt (fixed + dynamic baseline tables)

Default paths are alongside this script. Override with --results and --baseline.

Example:
  cd 4-BettingAlgorithm
  python plot_strategy_comparison.py
  python plot_strategy_comparison.py -o my_comparison.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+"  # week, games, bets, W, L
    r"\S+%\s+\S+%\s+\S+%\s+"  # Acc%, ROI/G all%, ROI/G bet%
    r"\$\s*([-+]?\d+(?:\.\d+)?)\s*$"  # profit
)

TOT_RE = re.compile(
    r"^\s*TOT\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+"  # TOT games bets W L
    r"\S+%\s+\S+%\s+\S+%\s+\$\s*([-+]?\d+(?:\.\d+)?)"
)


def _parse_weekly_profit_table(lines: list[str], start_i: int) -> tuple[list[tuple[int, float]], dict]:
    """
    From line index start_i (first line after anchor), scan until TOT row.
    Returns (weekly_profit list, totals dict with bets, wins, losses, profit).
    """
    weekly: list[tuple[int, float]] = []
    totals: dict = {}
    in_data = False
    for j in range(start_i, len(lines)):
        line = lines[j]
        stripped = line.strip()
        if stripped.startswith("Week") and "Games" in stripped:
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("---"):
            continue
        tm = TOT_RE.match(line)
        if tm:
            totals = {
                "bets": int(tm.group(1)),
                "wins": int(tm.group(2)),
                "losses": int(tm.group(3)),
                "profit": float(tm.group(4)),
            }
            break
        m = ROW_RE.match(line)
        if m:
            wk = int(m.group(1))
            if 3 <= wk <= 19:
                weekly.append((wk, float(m.group(2))))
    weekly.sort(key=lambda x: x[0])
    return weekly, totals


def parse_results_file(path: Path) -> tuple[str, list[tuple[int, float]], dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    anchor = "LEAVE-ONE-WEEK-OUT DETAIL"
    for i, line in enumerate(lines):
        if anchor in line:
            label = "Optimized strategy (best trial)"
            weekly, totals = _parse_weekly_profit_table(lines, i + 1)
            if len(weekly) != 17:
                raise ValueError(
                    f"{path}: expected 17 test weeks in table after {anchor!r}, "
                    f"found {len(weekly)}"
                )
            return label, weekly, totals
    raise ValueError(f"{path}: missing anchor {anchor!r}")


def parse_baseline_file(path: Path) -> tuple[list[tuple[str, list[tuple[int, float]], dict]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out: list[tuple[str, list[tuple[int, float]], dict]] = []
    anchors = [
        ("Opening favorite — fixed stake", "[FIXED]"),
        ("Opening favorite — Kelly (dynamic)", "[DYNAMIC (Kelly)]"),
    ]
    for label, needle in anchors:
        idx = None
        for i, line in enumerate(lines):
            if needle in line:
                idx = i
                break
        if idx is None:
            raise ValueError(f"{path}: missing baseline section containing {needle!r}")
        weekly, totals = _parse_weekly_profit_table(lines, idx + 1)
        if len(weekly) != 17:
            raise ValueError(
                f"{path}: expected 17 weeks for {needle!r}, found {len(weekly)}"
            )
        out.append((label, weekly, totals))
    return out


def _cumsum(weekly: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    weeks = np.array([w for w, _ in weekly], dtype=float)
    prof = np.array([p for _, p in weekly], dtype=float)
    return weeks, np.cumsum(prof)


def plot_comparison(
    series: list[tuple[str, list[tuple[int, float]], dict]],
    output: Path,
    title: str = "Betting strategy vs baselines (leave-one-week-out, weeks 3–19)",
) -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="semibold")

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    names = [s[0] for s in series]
    totals_profit = [s[2].get("profit", sum(p for _, p in s[1])) for s in series]
    totals_bets = [s[2]["bets"] for s in series]

    # (0,0) Total profit — horizontal bars
    ax = axes[0, 0]
    y = np.arange(len(names))
    bars = ax.barh(y, totals_profit, color=colors[: len(names)], edgecolor="0.2", linewidth=0.6)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlabel("Total profit ($)")
    ax.set_title("Total profit (sum of test weeks)")
    for rect, v in zip(bars, totals_profit):
        x0 = rect.get_x()
        w = rect.get_width()
        cy = rect.get_y() + rect.get_height() / 2
        # Negative bars: matplotlib uses x0=0 and negative width; anchor at zero end so
        # labels sit inside the bar instead of on the y-axis tick labels.
        if w >= 0:
            ax.annotate(
                f"${v:,.0f}",
                xy=(x0 + w, cy),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="medium",
            )
        else:
            ax.annotate(
                f"${v:,.0f}",
                xy=(x0, cy),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=9,
                fontweight="medium",
            )

    # (0,1) Bets placed
    ax = axes[0, 1]
    xpos = np.arange(len(names))
    ax.bar(xpos, totals_bets, color=colors[: len(names)], edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(xpos, names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Bets placed")
    ax.set_title("Activity (placed bets, all test weeks)")
    for xi, b in zip(xpos, totals_bets):
        ax.annotate(str(b), xy=(xi, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # (1,0) Cumulative profit
    ax = axes[1, 0]
    for (name, weekly, _), c in zip(series, colors):
        wk, cum = _cumsum(weekly)
        ax.plot(wk, cum, marker="o", markersize=3, linewidth=1.8, label=name, color=c)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Test week")
    ax.set_ylabel("Cumulative profit ($)")
    ax.set_title("Cumulative profit by test week")
    ax.legend(loc="best", fontsize=8)

    # (1,1) Weekly profit grouped bars
    ax = axes[1, 1]
    weeks = list(range(3, 20))
    n_series = len(series)
    width = 0.25
    offsets = np.linspace(-(n_series - 1) * width / 2, (n_series - 1) * width / 2, n_series)
    for off, (name, weekly, _), c in zip(offsets, series, colors):
        prof_by_wk = {w: p for w, p in weekly}
        vals = [prof_by_wk[w] for w in weeks]
        x = np.array(weeks, dtype=float) + off
        ax.bar(x, vals, width=width * 0.92, label=name, color=c, edgecolor="0.2", linewidth=0.4)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Test week")
    ax.set_ylabel("Week profit ($)")
    ax.set_title("Profit each test week")
    ax.set_xticks(weeks)
    ax.legend(loc="lower left", fontsize=7, ncol=1)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot strategy vs baseline from saved run logs.")
    p.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results.txt",
        help="Path to optimizer results.txt",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_results.txt",
        help="Path to baseline_results.txt",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "strategy_comparison.png",
        help="Output PNG path",
    )
    args = p.parse_args()

    lab0, w0, t0 = parse_results_file(args.results)
    baselines = parse_baseline_file(args.baseline)
    series = [(lab0, w0, t0)] + [(lb, w, t) for lb, w, t in baselines]

    plot_comparison(series, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
