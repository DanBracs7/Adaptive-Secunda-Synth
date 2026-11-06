"""
Tests for src/report_figures.py, on the saved results in results/.

1. The realised fractions of Figure 1 (b) equal the gains of the E2
   analysis (control_adherence.factorial_effects), and every interval
   contains its estimate.
2. Both figures are written as PDF and PNG.

Run from the repository root:
    python tests/test_report_figures.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import control_adherence as ca  # noqa: E402
import report_figures  # noqa: E402


def test_realised_fractions_match_analysis():
    clips = pd.read_csv(PROJECT_ROOT / "results" / "control_adherence" / "clips.csv")
    table = report_figures.realised_fractions(clips)
    assert len(table) == 12
    assert (table["low"] <= table["fraction"]).all() and (table["fraction"] <= table["high"]).all()

    for checkpoint, group in clips.groupby("checkpoint"):
        gains = ca.factorial_effects(group).set_index(["requested", "measured"])["gain"]
        rows = table[table["checkpoint"] == checkpoint].set_index("effect")["fraction"]
        assert abs(rows["tension → tension"] - gains[("tension", "tension")]) < 1e-12
        assert abs(rows["combat → combat"] - gains[("combat_score", "combat_score")]) < 1e-12


def test_figures_are_written():
    with tempfile.TemporaryDirectory() as folder:
        report_figures.FIGURES = Path(folder)
        report_figures.figure1()
        report_figures.figure_a_e4()
        names = sorted(path.name for path in Path(folder).iterdir())
    assert names == ["fig1_control_adherence.pdf", "fig1_control_adherence.png",
                     "figA_e4_decomposition.pdf", "figA_e4_decomposition.png"], names


if __name__ == "__main__":
    for test in (test_realised_fractions_match_analysis, test_figures_are_written):
        test()
        print("PASS", test.__name__)
    print("All report_figures tests passed.")
