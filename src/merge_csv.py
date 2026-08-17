import pandas as pd
import numpy as np
import json

# Slakh raw features — use as-is
df_slakh = pd.read_csv("csv/slakh_raw_features.csv")

# Skyrim raw features — extract ONLY the raw columns
raw_cols = ["track_id", "perc_ratio", "percussive_loudness",
            "note_shortness", "dissonance", "spectral_centroid_mean"]

df_skyrim_raw = pd.read_csv("csv/combat_chill_profile_normalized_v2.csv")[raw_cols]
df_witcher_raw = pd.read_csv("csv/combat_chill_profile_W3_raw.csv")[raw_cols]

df_ost = pd.concat([df_skyrim_raw, df_witcher_raw], ignore_index=True)
print(f"Slakh: {len(df_slakh)} tracks, OST (Skyrim+Witcher3): {len(df_ost)} tracks")

features = ["perc_ratio", "percussive_loudness", "note_shortness",
            "dissonance", "spectral_centroid_mean"]
norm_stats = {}
for feat in features:
    p_low_slakh, p_high_slakh = np.percentile(df_slakh[feat], [5, 95])
    p_low_ost,   p_high_ost   = np.percentile(df_ost[feat],   [5, 95])
    norm_stats[feat] = {
        "p_low":  float(np.mean([p_low_slakh, p_low_ost])),
        "p_high": float(np.mean([p_high_slakh, p_high_ost])),
    }
    print(f"{feat}: Slakh=({p_low_slakh:.3f},{p_high_slakh:.3f})  "
          f"OST=({p_low_ost:.3f},{p_high_ost:.3f})  "
          f"Balanced=({norm_stats[feat]['p_low']:.3f},{norm_stats[feat]['p_high']:.3f})")

# ── Save the librosa-feature stats now ──
# NOTE: tension_rms still needs to be added later (Cell 3 from earlier —
# RMS sampled directly from waveforms, run separately since it doesn't
# come from this CSV at all)
with open("norm_stats_partial.json", "w") as f:
    json.dump(norm_stats, f, indent=2)

print("\nSaved norm_stats_partial.json — still needs tension_rms merged in before tokenizing.")