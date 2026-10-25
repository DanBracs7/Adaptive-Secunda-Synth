"""
Experiment E4: decomposition of the Baseline -> Slakh-FT improvement.

The fine-tune that produced Slakh-FT changed three things at once (more
training, more data, new codebook loss weights). The control runs of
notebook 06 change one factor at a time, so the improvement splits into
three consecutive steps:

    extra training    Baseline -> R1        same 704 tracks, Baseline weights
    new data          R1 -> R2              1229 tracks, Baseline weights
    new loss weights  R2 -> Slakh-FT        1229 tracks, new weights

The steps add up exactly to the total Baseline -> Slakh-FT. Every
checkpoint is evaluated on the same test segments, so each difference is
paired by segment; its uncertainty over the choice of test segments is a
percentile bootstrap over segments. The bootstrap does not cover the
randomness of training (one run per configuration).

The metric is the unweighted mean of the per-codebook cross-entropies,
which does not depend on the loss weights of a run.
"""

import numpy as np

from condition_ablation import bootstrap_mean_ci


STEPS = (
    ("extra training", "Baseline", "R1"),
    ("new data", "R1", "R2"),
    ("new loss weights", "R2", "Slakh-FT"),
)
TOTAL = ("total", "Baseline", "Slakh-FT")


def decomposition_table(segment_ce, steps=STEPS + (TOTAL,), num_resamples=10_000, seed=0):
    """
    segment_ce: {checkpoint: array [N, Q]} cross-entropy of every test
        segment and codebook (same segments, same order, for every checkpoint).
    Returns a list of rows, one per step, with the change of the unweighted
    cross-entropy (`delta`, negative = better), its bootstrap 95% CI, the
    fraction of segments that improve, and the mean change per codebook
    (`delta_q0` ... with CIs `q0_low`, `q0_high` ...).
    """
    shapes = {name: np.asarray(values).shape for name, values in segment_ce.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Every checkpoint needs the same segments and codebooks: {shapes}")

    rows = []
    for step, before, after in steps:
        difference = np.asarray(segment_ce[after], dtype=np.float64) - np.asarray(segment_ce[before], dtype=np.float64)
        delta, low, high = bootstrap_mean_ci(difference.mean(axis=1), num_resamples=num_resamples, seed=seed)
        row = {"step": step, "from": before, "to": after, "delta": delta, "ci_low": low, "ci_high": high,
               "improved_segments": float((difference.mean(axis=1) < 0).mean())}
        for q in range(difference.shape[1]):
            q_delta, q_low, q_high = bootstrap_mean_ci(difference[:, q], num_resamples=num_resamples, seed=seed)
            row.update({f"delta_q{q}": q_delta, f"q{q}_low": q_low, f"q{q}_high": q_high})
        rows.append(row)
    return rows
