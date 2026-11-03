"""Score move predictions: top-1, top-3, per-phase accuracy, 95% confidence intervals.

Input is the table F3 writes: one row per position with
    move                         the move the human actually played
    prediction1..prediction3     the model's top 3 guesses
and optionally
    ply        half-move number, needed for the per-phase breakdown
    game_id    needed to resample whole games for the confidence interval

Every metric here is "the fraction of positions where the model was right",
so one confidence-interval function covers all of them.

Usage:
    import pandas as pd
    from evaluation_metrics import evaluate_predictions, format_report
    results = evaluate_predictions(pd.read_csv("data/thakt_baseline.csv"))
    print(format_report(results))
"""

import numpy as np

PRED_COLS = ["prediction1", "prediction2", "prediction3"]

# Phase boundaries by ply (half-moves): 1-20 = moves 1-10, 21-60 = moves 11-30.
PHASES = {
    "opening":    lambda ply: ply <= 20,
    "middlegame": lambda ply: (ply > 20) & (ply <= 60),
    "endgame":    lambda ply: ply > 60,
}


def correct_top_k(df, k=1):
    if not 1 <= k <= len(PRED_COLS):
        raise ValueError(f"k must be between 1 and {len(PRED_COLS)}")
    return df[PRED_COLS[:k]].eq(df["move"], axis=0).any(axis=1)


def bootstrap_ci(correct, game_ids=None, n_resamples=1000, confidence=0.95, seed=0):
    """Confidence interval for the fraction of True values in `correct`.
    """
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)

    if game_ids is not None:
        # Collapse to per-game totals, then resample games.
        _, group = np.unique(np.asarray(game_ids), return_inverse=True)
        hits = np.bincount(group, weights=correct)
        counts = np.bincount(group).astype(float)
        idx = rng.integers(0, len(hits), size=(n_resamples, len(hits)))
        stats = hits[idx].sum(axis=1) / counts[idx].sum(axis=1)
    else:
        idx = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
        stats = correct[idx].mean(axis=1)

    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return (float(low), float(high))


def _score(correct, game_ids, n_resamples, seed):
    """Accuracy + CI + count for one boolean Series."""
    n = int(len(correct))
    if n == 0:
        return {"acc": np.nan, "ci": (np.nan, np.nan), "n": 0}
    return {
        "acc": float(correct.mean()),
        "ci": bootstrap_ci(correct, game_ids, n_resamples, seed=seed),
        "n": n,
    }


def evaluate_predictions(df, n_resamples=1000, seed=0):
    """Full evaluation. Returns a dict; use format_report() to print it."""
    missing = [c for c in ["move"] + PRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}; expected F3's output")

    has_games = "game_id" in df.columns
    has_ply = "ply" in df.columns
    games = df["game_id"].values if has_games else None

    top1 = correct_top_k(df, 1)
    top3 = correct_top_k(df, 3)

    results = {
        "n_positions": int(len(df)),
        "n_games": int(df["game_id"].nunique()) if has_games else None,
        "ci_resamples": "games" if has_games else "moves (no game_id column)",
        "top1": _score(top1, games, n_resamples, seed),
        "top3": _score(top3, games, n_resamples, seed),
        "phases": {},
    }

    if has_ply:
        for name, in_phase in PHASES.items():
            mask = in_phase(df["ply"]).values
            results["phases"][name] = _score(
                top1[mask], games[mask] if has_games else None, n_resamples, seed)

    return results


def format_report(r):
    def line(label, s):
        if s["n"] == 0:
            return f"  {label:<12} {'-':>6}   (no positions)"
        lo, hi = s["ci"]
        return (f"  {label:<12} {s['acc'] * 100:5.1f}%   "
                f"95% CI {lo * 100:5.1f} - {hi * 100:5.1f}   n={s['n']}")

    games = f", {r['n_games']} games" if r["n_games"] is not None else ""
    out = [f"Evaluation: {r['n_positions']} positions{games}  "
           f"(CI resamples {r['ci_resamples']})",
           line("top-1", r["top1"]),
           line("top-3", r["top3"])]
    if r["phases"]:
        out.append("  top-1 by phase:")
        for name, s in r["phases"].items():
            out.append(line("  " + name, s))
    else:
        out.append("  (no 'ply' column, so no per-phase breakdown)")
    return "\n".join(out)