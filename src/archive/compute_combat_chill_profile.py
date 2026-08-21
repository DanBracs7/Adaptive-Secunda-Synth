import librosa
import numpy as np
from os import path
import pandas as pd
from pathlib import Path


def compute_percussive_ratio(y, sr):
    """
    Returns fraction of total energy that is percussive.
    High value → combat (strong beat-driven percussion)
    Low value  → chill (mostly harmonic, piano/strings)
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harmonic ** 2)
    perc_energy = np.sum(y_percussive ** 2)
    total = harm_energy + perc_energy + 1e-8
    return perc_energy / total

def compute_beat_alignment(y, sr):
    """
    Measures how strongly percussive onsets align with detected beats.
    High → tight, on-the-beat combat rhythm
    Low  → loose, ambient, non-metric chill
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, units='frames')
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    if len(beats) == 0:
        return 0.0

    beat_strengths = onset_env[np.clip(beats, 0, len(onset_env) - 1)]
    return float(np.mean(beat_strengths)) if len(beat_strengths) > 0 else 0.0


def compute_note_duration_stats(y, sr):
    """
    Returns mean inter-onset interval (IOI) in seconds.
    Short IOI → combat (rapid, short notes)
    Long IOI  → chill (sustained, long notes)
    """
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    if len(onsets) < 2:
        return None  # too sparse to measure
    iois = np.diff(onsets)
    return float(np.mean(iois))


def compute_dissonance_score(y, sr):
    """
    Rough dissonance proxy: correlation between adjacent chroma bins.
    High simultaneous energy in adjacent semitones → dissonant/tense.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # [12, T]
    # Sum of energy in semitone-adjacent pairs (proxy for clashing intervals)
    adjacent_energy = np.sum(chroma[:-1] * chroma[1:], axis=0)  # [T]
    total_energy = np.sum(chroma, axis=0) + 1e-8
    dissonance = adjacent_energy / total_energy
    return float(np.mean(dissonance))

def compute_percussive_loudness(y, sr):
    """
    Absolute RMS energy of the percussive component (not a ratio).
    High → heavy drums/hits (combat)
    Low  → gentle strums/plucks (bard/folk)
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return float(np.sqrt(np.mean(y_percussive ** 2)))

def compute_spectral_centroid_mean(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid))

def compute_combat_chill_profile(file_path, sr_target=22050):
    y, sr = librosa.load(file_path, sr=sr_target, mono=True)

    perc_ratio   = compute_percussive_ratio(y, sr)
    perc_loud    = compute_percussive_loudness(y, sr)
    beat_align   = compute_beat_alignment(y, sr)   # reference only
    ioi          = compute_note_duration_stats(y, sr)
    dissonance   = compute_dissonance_score(y, sr)
    centroid     = compute_spectral_centroid_mean(y, sr)

    note_shortness = float(np.clip(1.0 - (ioi / 1.5), 0.0, 1.0)) if ioi else 0.0

    return {
        "perc_ratio": perc_ratio,
        "percussive_loudness": perc_loud,
        "beat_align": beat_align,
        "mean_ioi_sec": ioi,
        "note_shortness": note_shortness,
        "dissonance": dissonance,
        "spectral_centroid_mean": centroid,
    }


def normalize_features_and_rescore(df):
    features = [
        "perc_ratio",
        "percussive_loudness",
        "note_shortness",
        "dissonance",
        "spectral_centroid_mean",
    ]

    df_norm = df.copy()
    for feat in features:
        p5 = df[feat].quantile(0.05)
        p95 = df[feat].quantile(0.95)
        df_norm[feat + "_norm"] = ((df[feat] - p5) / (p95 - p5 + 1e-8)).clip(0, 1)

    df_norm["combat_score"] = (
        0.30 * df_norm["perc_ratio_norm"] +
        0.25 * df_norm["percussive_loudness_norm"] +
        0.10 * df_norm["note_shortness_norm"] +
        0.15 * df_norm["dissonance_norm"] +
        0.20 * df_norm["spectral_centroid_mean_norm"]
    )

    df_norm["chill_score"] = 1.0 - df_norm["combat_score"]
    return df_norm


def find_audio_files_deduped(folder):
    all_files = find_audio_files(folder)
    by_stem = {}
    for f in all_files:
        key = f.stem.lower()
        if key not in by_stem or f.suffix.lower() == ".flac":
            by_stem[key] = f
    return sorted(by_stem.values())


DATA_DIR = Path("/content/drive/MyDrive/Robe/Skyrim soundtracks/jeremy-soule-the-elder-scrolls-v-skyrim-the-original-game-soundtrack")

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".ogg"}

def find_audio_files(folder: Path):
    return sorted([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])

# 1. Extract raw features (what you already did)
audio_files = find_audio_files_deduped(DATA_DIR)   # dedupe flac/mp3 first
print(f"Found {len(audio_files)} audio files")

# 1. Extract (re-run needed — percussive_loudness is new)
results = []
for i, fp in enumerate(audio_files):
    try:
        profile = compute_combat_chill_profile(fp)
        profile["track_id"] = fp.stem
        results.append(profile)
        print(f"[{i+1}/{len(audio_files)}] {fp.name} done")
    except Exception as e:
        print(f"[{i+1}/{len(audio_files)}] FAILED: {fp.name} → {e}")

df = pd.DataFrame(results)
df_fixed = normalize_features_and_rescore(df)



SAVE_DIR = Path("/content/drive/MyDrive/DeepLearningProject")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Save the raw (pre-normalization) features
df.to_csv(SAVE_DIR / "combat_chill_profile_raw.csv", index=False)

# Save the normalized, scored version
df_fixed.to_csv(SAVE_DIR / "combat_chill_profile_normalized.csv", index=False)

print(f"Saved raw features to: {SAVE_DIR / 'combat_chill_profile_raw.csv'}")
print(f"Saved normalized scores to: {SAVE_DIR / 'combat_chill_profile_normalized.csv'}")

# If df is still in memory from the run, just do:
df_fixed = normalize_features_and_rescore(df)
print(df_fixed.sort_values("combat_score", ascending=False)[["track_id", "combat_score", "chill_score"]].to_string(index=False))