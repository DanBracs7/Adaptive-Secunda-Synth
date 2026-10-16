"""
Check that configs/norm_stats.json can be reproduced from the per-track
feature tables in csv/.

The five combat-feature entries are recomputed exactly as in
notebooks/archive/new-slakh-token-normalized.ipynb (cell 4): 5th/95th
percentiles of the Slakh pool and of the OST pool (Skyrim + Witcher 3),
averaged.

The "tension_rms" entry was computed from raw audio frames, so it cannot be
checked from these tables. The script reports which of the versions found
in the project history (control_features.KNOWN_TENSION_RMS_STATS) is the
one stored in the file; notebooks/02_control_statistics_audit.ipynb finds
which version each tensor block was actually normalised with.

slakh_raw_features.csv contains the 800 train and 151 test Slakh tracks,
so the percentile pool included the test tracks. The script also reports
how much the combat score changes when they are excluded.

Run from the repository root:
    python src/verify_norm_stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from control_features import KNOWN_TENSION_RMS_STATS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "csv"
NORM_STATS_PATH = PROJECT_ROOT / "configs" / "norm_stats.json"

RAW_COLUMNS = [
    "track_id", "perc_ratio", "percussive_loudness",
    "note_shortness", "dissonance", "spectral_centroid_mean",
]

FEATURES = [
    "perc_ratio", "percussive_loudness", "note_shortness",
    "dissonance", "spectral_centroid_mean",
]

# Same weights as compute_causal_combat_score_track in the tokenizer.
COMBAT_WEIGHTS = {
    "perc_ratio": 0.30,
    "percussive_loudness": 0.25,
    "note_shortness": 0.10,
    "dissonance": 0.15,
    "spectral_centroid_mean": 0.20,
}


def balanced_percentiles(df_slakh, df_ost):
    stats = {}

    for feature in FEATURES:
        p_low_slakh, p_high_slakh = np.percentile(df_slakh[feature], [5, 95])
        p_low_ost, p_high_ost = np.percentile(df_ost[feature], [5, 95])
        stats[feature] = {
            "p_low": float(np.mean([p_low_slakh, p_low_ost])),
            "p_high": float(np.mean([p_high_slakh, p_high_ost])),
        }

    return stats


def track_level_combat_score(df, stats):
    score = 0.0

    for feature in FEATURES:
        p_low = stats[feature]["p_low"]
        p_high = stats[feature]["p_high"]
        normalized = np.clip(
            (df[feature] - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0
        )
        score = score + COMBAT_WEIGHTS[feature] * normalized

    return score


def main():
    df_slakh = pd.read_csv(CSV_DIR / "slakh_raw_features.csv")
    df_skyrim_raw = pd.read_csv(
        CSV_DIR / "combat_chill_profile_normalized_v2.csv"
    )[RAW_COLUMNS]
    df_witcher_raw = pd.read_csv(
        CSV_DIR / "combat_chill_profile_W3_raw.csv"
    )[RAW_COLUMNS]
    df_ost = pd.concat([df_skyrim_raw, df_witcher_raw], ignore_index=True)

    with open(NORM_STATS_PATH, "r", encoding="utf-8") as file:
        stored = json.load(file)

    recomputed = balanced_percentiles(df_slakh, df_ost)

    print(f"Slakh tracks: {len(df_slakh)}, OST tracks: {len(df_ost)}")

    for feature in FEATURES:
        for key in ("p_low", "p_high"):
            if not np.isclose(recomputed[feature][key], stored[feature][key],
                              rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"{feature}.{key}: stored {stored[feature][key]} "
                    f"!= recomputed {recomputed[feature][key]}"
                )

    print("PASS: combat-feature statistics match configs/norm_stats.json.")

    stored_tension = stored["tension_rms"]
    matching_versions = [
        name
        for name, stats in KNOWN_TENSION_RMS_STATS.items()
        if stats == stored_tension
    ]
    print(
        "tension_rms stored in configs/norm_stats.json:",
        stored_tension,
        "-> version",
        matching_versions or "unknown",
    )

    # Sensitivity: percentile pool without the Slakh test tracks.
    train_only = balanced_percentiles(
        df_slakh[df_slakh["split"] == "train"], df_ost
    )
    difference = np.max(np.abs(
        track_level_combat_score(df_slakh, stored)
        - track_level_combat_score(df_slakh, train_only)
    ))
    print(
        "Max change of the track-level combat score if the Slakh test "
        f"tracks are excluded from the pool: {difference:.4f}"
    )


if __name__ == "__main__":
    main()
