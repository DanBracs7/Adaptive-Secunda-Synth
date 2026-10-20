"""
Tests for src/listening_test.py (no audio files needed).

1. select_pairs is deterministic, gives 4 + 8 + 8 pairs, and every prompt
   of a checkpoint appears in one block only.
2. prepare_pair removes the 1-s prompt, matches loudness in the matched
   blocks, keeps the level difference in block A, and never exceeds the
   peak limit.
3. summarize counts how often the requested clip was chosen and gives a
   Wilson interval and a binomial p-value.

Run from the repository root:
    python tests/test_listening_test.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import listening_test as lt  # noqa: E402


def test_pair_selection():
    pairs = lt.select_pairs()
    assert pairs == lt.select_pairs()
    assert [p["block"] for p in pairs].count("A") == 4
    assert [p["block"] for p in pairs].count("B") == 8
    assert [p["block"] for p in pairs].count("C") == 8
    for checkpoint in lt.CHECKPOINTS:
        prompts = [p["prompt"] for p in pairs if p["checkpoint"] == checkpoint]
        assert len(prompts) == len(set(prompts))
    assert all(p["high_condition"] == "tH_cH" for p in pairs if p["block"] == "C")


def test_prepare_pair():
    rng = np.random.default_rng(0)
    quiet = 0.02 * rng.standard_normal(10 * lt.SAMPLE_RATE)
    loud = 0.3 * rng.standard_normal(10 * lt.SAMPLE_RATE)

    low, high = lt.prepare_pair(quiet, loud, loudness_matched=True)
    assert len(low) == len(high) == 9 * lt.SAMPLE_RATE
    assert abs(lt.rms_db(low) - lt.rms_db(high)) < 0.01
    assert max(np.abs(low).max(), np.abs(high).max()) <= lt.PEAK_LIMIT + 1e-9

    low, high = lt.prepare_pair(quiet, loud, loudness_matched=False)
    original_difference = lt.rms_db(loud[lt.SAMPLE_RATE:]) - lt.rms_db(quiet[lt.SAMPLE_RATE:])
    assert abs((lt.rms_db(high) - lt.rms_db(low)) - original_difference) < 0.1
    assert max(np.abs(low).max(), np.abs(high).max()) <= lt.PEAK_LIMIT + 1e-9


def test_summarize():
    responses = [{"listener": f"L{i % 3}", "block": "B", "checkpoint": "OST-FT", "chose_high": "1" if i < 15 else "0"}
                 for i in range(20)]
    rows = {(r["block"], r["checkpoint"]): r for r in lt.summarize(responses)}
    overall = rows[("B", "all")]
    assert overall["answers"] == 20 and overall["chose_requested"] == 15
    assert overall["ci_low"] < 0.75 < overall["ci_high"]
    assert overall["p_value"] < 0.05 and overall["listeners"] == 3


if __name__ == "__main__":
    for test in (test_pair_selection, test_prepare_pair, test_summarize):
        test()
        print("PASS", test.__name__)
    print("All listening_test tests passed.")
