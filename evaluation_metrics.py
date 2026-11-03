import numpy as np
import pandas as pd
from scipy.stats import bootstrap

def compute_top_k_accuracy(df, k=1):
    #Calculates if actual human move is within the top-k predicted moves.
    if k == 1:
        return (df['actual_move'] == df['pred_1']).mean()
    else:
        top_k_cols = [f'pred_{i}' for i in range(1, k + 1)]
        return df.apply(lambda row: row['actual_move'] in row[top_k_cols].values, axis=1).mean()

def compute_phase_accuracies(df):
    #Calculates Top-1 accuracy broken down by game phase.
    # Opening: Plies 1–20 (Moves 1–10)
    opening_df = df[df['ply'] <= 20]
    # Middlegame: Plies 21–60 (Moves 11–30)
    middle_df = df[(df['ply'] > 20) & (df['ply'] <= 60)]
    # Endgame: Plies 61+ (Moves 31+)
    endgame_df = df[df['ply'] > 60]

    return {
        'opening_top1': compute_top_k_accuracy(opening_df, k=1) if len(opening_df) > 0 else 0.0,
        'middlegame_top1': compute_top_k_accuracy(middle_df, k=1) if len(middle_df) > 0 else 0.0,
        'endgame_top1': compute_top_k_accuracy(endgame_df, k=1) if len(endgame_df) > 0 else 0.0,
    }

def compute_bootstrap_ci(df, metric_fn, n_resamples=1000):
    #Calculates a 95% bootstrap confidence interval for a metric.
    data = (df['actual_move'] == df['pred_1']).values.astype(float)
    res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=n_resamples, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

def evaluate_predictions(df):
    #Master evaluation function required for Task P8.
    top1 = compute_top_k_accuracy(df, k=1)
    top3 = compute_top_k_accuracy(df, k=3)
    phase_acc = compute_phase_accuracies(df)
    ci_low, ci_high = compute_bootstrap_ci(df, compute_top_k_accuracy)

    results = {
        'top_1_accuracy': top1,
        'top_3_accuracy': top3,
        'phase_accuracy': phase_acc,
        'bootstrap_95_ci': (ci_low, ci_high)
    }
    return results