"""P10: summary table + 3 charts from a folder of prediction CSVs.

Works on any folder of per-player prediction files with the columns F3 writes
(move, prediction1-3, ply, game_id, active_elo): base Maia-2, Maia-3, or a
fine-tuned model.

Usage (Colab, from the repo root):
    python shared/p10_plots.py /content/drive/MyDrive/acm_chess/baselines --label "Base Maia-2"
    python shared/p10_plots.py /content/drive/MyDrive/acm_chess/baselines_maia3/pgn --label "Base Maia-3"

Writes into <folder>/plots/:
    summary.csv  top1_per_player.png  top1_by_phase.png  top1_vs_rating.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_metrics import evaluate_predictions

COLS = ["move", "prediction1", "prediction2", "prediction3", "ply", "game_id", "active_elo"]
PHASE_COLORS = {"opening": "#2a78d6", "middlegame": "#eb6834", "endgame": "#1baf7a"}
BAR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def summarize(folder):
    rows = []
    for path in sorted(Path(folder).glob("*.csv")):
        header = pd.read_csv(path, nrows=0).columns
        if not set(COLS) <= set(header):
            continue  # not a prediction file (e.g. comparison.csv)
        df = pd.read_csv(path, usecols=COLS, dtype={"game_id": str})
        r = evaluate_predictions(df)
        row = {
            "player": path.stem,
            "test_games": r["n_games"],
            "positions": r["n_positions"],
            "top1": r["top1"]["acc"] * 100,
            "top1_lo": r["top1"]["ci"][0] * 100,
            "top1_hi": r["top1"]["ci"][1] * 100,
            "top3": r["top3"]["acc"] * 100,
            "mean_elo": df["active_elo"].mean(),
        }
        for phase, s in r["phases"].items():
            row[phase] = s["acc"] * 100
        rows.append(row)
        print(f"{path.stem:<22} top-1 {row['top1']:.1f}%")
    if not rows:
        raise SystemExit(f"No prediction CSVs found in {folder}")
    return pd.DataFrame(rows).sort_values("top1", ascending=False).reset_index(drop=True)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


def plot_per_player(s, label, out):
    s = s.iloc[::-1]  # best at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.5))
    err = [s["top1"] - s["top1_lo"], s["top1_hi"] - s["top1"]]
    ax.barh(s["player"], s["top1"], color=BAR, height=0.6, xerr=err,
            error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
    mean = s["top1"].mean()
    ax.axvline(mean, color=MUTED, ls="--", lw=1.2, label=f"mean {mean:.1f}%")
    ax.set_xlim(max(0, s["top1_lo"].min() - 3), s["top1_hi"].max() + 2)
    ax.set_xlabel("Top-1 accuracy (%)", color=INK)
    ax.set_title(f"{label} top-1 per player (95% CI)", color=INK)
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="lower right", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "top1_per_player.png", dpi=150)
    plt.close(fig)


def plot_by_phase(s, label, out):
    phases = [p for p in PHASE_COLORS if p in s]
    x = np.arange(len(s))
    w = 0.8 / len(phases)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(s)), 4.5))
    for i, p in enumerate(phases):
        ax.bar(x + (i - (len(phases) - 1) / 2) * w, s[p], width=w * 0.92,
               color=PHASE_COLORS[p], label=p)
    ax.set_xticks(x, s["player"], rotation=30, ha="right")
    low = np.nanmin(s[phases].values)
    ax.set_ylim(max(0, np.floor(low / 5) * 5 - 5), np.nanmax(s[phases].values) + 2)
    ax.set_ylabel("Top-1 (%)", color=INK)
    ax.set_title(f"{label} top-1 by game phase", color=INK)
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, ncol=len(phases), loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "top1_by_phase.png", dpi=150)
    plt.close(fig)


def plot_vs_rating(s, label, out, n_perm=10000):
    x, y = s["mean_elo"].values, s["top1"].values
    r = np.corrcoef(x, y)[0, 1]
    # permutation p-value: how often a random pairing gives |r| at least this big
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    p = (np.abs(perm) >= abs(r)).mean()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    err = [s["top1"] - s["top1_lo"], s["top1_hi"] - s["top1"]]
    ax.errorbar(x, y, yerr=err, fmt="o", color=BAR, ms=7, capsize=3, lw=1)
    for xi, yi, name in zip(x, y, s["player"]):
        ax.annotate(name, (xi, yi), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=MUTED)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 20, x.max() + 20, 50)
    ax.plot(xs, slope * xs + intercept, ls="--", color=MUTED, lw=1.2,
            label=f"linear fit (r = {r:.2f}, p = {p:.2f}, n = {len(s)})")
    lo, hi = s["top1_lo"].min(), s["top1_hi"].max()
    ax.set_ylim(lo - 0.5, hi + (hi - lo) * 0.2)  # headroom so the legend clears the points
    ax.set_xlabel("Mean active_elo in test set", color=INK)
    ax.set_ylabel("Top-1 (%)", color=INK)
    ax.set_title(f"{label} top-1 vs player rating", color=INK)
    ax.grid(color=GRID)
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "top1_vs_rating.png", dpi=150)
    plt.close(fig)
    return r, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with one prediction CSV per player")
    ap.add_argument("--label", default="Base Maia-2", help="model name for the chart titles")
    ap.add_argument("--out", default=None, help="default: <folder>/plots")
    a = ap.parse_args()

    out = Path(a.out or Path(a.folder) / "plots")
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(a.folder)
    s.round(2).to_csv(out / "summary.csv", index=False)
    plot_per_player(s, a.label, out)
    plot_by_phase(s, a.label, out)
    r, p = plot_vs_rating(s, a.label, out)

    print(f"\nmean top-1 {s['top1'].mean():.1f}%   mean top-3 {s['top3'].mean():.1f}%")
    print(f"rating vs top-1: r = {r:.2f}, p = {p:.2f} (n = {len(s)})")
    print(f"saved summary.csv and 3 charts to {out}")


if __name__ == "__main__":
    main()