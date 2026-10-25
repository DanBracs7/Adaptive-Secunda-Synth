"""
Tests for src/e4_decomposition.py (synthetic numbers, no model needed).

1. The three steps add up exactly to the total Baseline -> Slakh-FT, for
   the unweighted change and for every codebook.
2. A known shift is recovered: a step that lowers every segment by 0.1
   has delta -0.1, a zero-width CI and 100% improved segments; a step
   with noise only has a CI containing 0.
3. Checkpoints evaluated on different segments are refused.

Run from the repository root:
    python tests/test_e4_decomposition.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from e4_decomposition import decomposition_table  # noqa: E402

NUM_SEGMENTS, NUM_CODEBOOKS = 151, 8


def synthetic_ce(seed=0):
    rng = np.random.default_rng(seed)
    baseline = 5.0 + rng.normal(0.0, 0.5, size=(NUM_SEGMENTS, NUM_CODEBOOKS))
    r1 = baseline - 0.1                                          # exactly 0.1 better everywhere
    r2 = r1 + rng.normal(0.0, 0.05, size=r1.shape)               # noise only
    slakh_ft = r2 + np.linspace(0.05, -0.05, NUM_CODEBOOKS)      # trades coarse for fine codebooks
    return {"Baseline": baseline, "R1": r1, "R2": r2, "Slakh-FT": slakh_ft}


def test_steps_add_up_to_total():
    rows = {row["step"]: row for row in decomposition_table(synthetic_ce(), num_resamples=500)}
    steps = ("extra training", "new data", "new loss weights")
    assert abs(sum(rows[s]["delta"] for s in steps) - rows["total"]["delta"]) < 1e-12
    for q in range(NUM_CODEBOOKS):
        assert abs(sum(rows[s][f"delta_q{q}"] for s in steps) - rows["total"][f"delta_q{q}"]) < 1e-12


def test_known_shift_is_recovered():
    rows = {row["step"]: row for row in decomposition_table(synthetic_ce(), num_resamples=2000)}
    extra = rows["extra training"]
    assert abs(extra["delta"] + 0.1) < 1e-12
    assert abs(extra["ci_low"] + 0.1) < 1e-9 and abs(extra["ci_high"] + 0.1) < 1e-9   # constant change: zero-width CI
    assert extra["improved_segments"] == 1.0
    data = rows["new data"]
    assert data["ci_low"] < 0.0 < data["ci_high"]
    weights = rows["new loss weights"]
    assert weights["delta_q0"] > 0 > weights["delta_q7"]


def test_different_segments_are_refused():
    ce = synthetic_ce()
    ce["R2"] = ce["R2"][:-1]
    try:
        decomposition_table(ce, num_resamples=10)
    except ValueError:
        return
    raise AssertionError("ValueError not raised")


if __name__ == "__main__":
    for test in (test_steps_add_up_to_total, test_known_shift_is_recovered, test_different_segments_are_refused):
        test()
        print("PASS", test.__name__)
    print("All e4_decomposition tests passed.")
