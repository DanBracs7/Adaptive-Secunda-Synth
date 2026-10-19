"""
Re-run the analysis of experiment E2 from its saved results, without a GPU.

notebooks/05_control_adherence.ipynb generates and measures the clips and
saves the measurements; a copy of them is kept in results/control_adherence/.
This script recomputes every table of the notebook from clips.csv and adds
the combat-score effect at each tension level (control_adherence.combat_effect_by_tension).

Run from the repository root:
    python src/analyze_control_adherence.py [results_folder]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import control_adherence as ca


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(results_dir):
    clips = pd.read_csv(results_dir / "clips.csv")
    pd.set_option("display.width", 160)
    number_format = lambda value: f"{value:.3f}"  # noqa: E731

    for checkpoint_name, group in clips.groupby("checkpoint", sort=False):
        print(f"══ {checkpoint_name} ({group['unit'].nunique()} prompt x seed units) ══")

        print("\nFactorial effects (high - low request, averaged over the other control):")
        print(ca.factorial_effects(group).to_string(index=False, float_format=number_format))

        print("\nCombat-score request at each tension level:")
        print(ca.combat_effect_by_tension(group).to_string(index=False, float_format=number_format))

        print("\nSteps at 5 s (post - pre):")
        print(ca.step_effects(group).to_string(index=False, float_format=number_format))

        factorial = group[group["condition"].isin(ca.FACTORIAL_CONDITIONS)]
        r_tension = np.corrcoef(factorial["requested_tension_full"], factorial["tension_full"])[0, 1]
        r_combat = np.corrcoef(factorial["requested_combat_full"], factorial["combat_full"])[0, 1]
        print(f"\nClip-level correlation requested vs measured: tension r={r_tension:.3f}, "
              f"combat score r={r_combat:.3f} (n={len(factorial)})\n")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "results" / "control_adherence"
    main(folder)
