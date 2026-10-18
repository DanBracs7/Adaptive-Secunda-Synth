"""
Regression test for src/control_features.py.

The REFERENCE_* code below is copied verbatim from
notebooks/archive/new-slakh-token-normalized.ipynb (cell 5); it reads the
global NORM_STATS, exactly as there. The refactored functions must give
identical results.

Also checks the property used by notebooks/02_control_statistics_audit:
because the combat window is causal, the scores of the first K hops
computed on the first K seconds equal those computed on the full track, and that
combat_score_of_segment (used on generated clips) equals one causal
window.

Run from the repository root:
    python tests/test_control_features.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import control_features  # noqa: E402

with open(PROJECT_ROOT / "configs" / "norm_stats.json", "r", encoding="utf-8") as file:
    NORM_STATS = json.load(file)

SAMPLE_RATE = 24_000
FRAME_RATE = 75


# ── Copied from new-slakh-token-normalized.ipynb ──
def REFERENCE_normalize(value, feature_name):
    stats = NORM_STATS[feature_name]
    return float(np.clip(
        (value - stats["p_low"]) /
        (stats["p_high"] - stats["p_low"] + 1e-8),
        0.0,
        1.0,
    ))


def REFERENCE_compute_combat_features_from_array(y, sr):
    import librosa
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    perc_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-8)
    perc_loudness = float(np.sqrt(np.mean(y_percussive ** 2)))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    ioi = float(np.mean(np.diff(onsets))) if len(onsets) >= 2 else 1.5
    note_shortness = float(np.clip(1.0 - ioi / 1.5, 0.0, 1.0))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    adjacent_energy = np.sum(chroma[:-1] * chroma[1:], axis=0)
    dissonance = float(np.mean(adjacent_energy / (np.sum(chroma, axis=0) + 1e-8)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return perc_ratio, perc_loudness, note_shortness, dissonance, centroid


def REFERENCE_compute_causal_combat_score_track(y, sr, window_sec=5.0, hop_sec=1.0):
    hop_len = int(hop_sec * sr)
    window_len = int(window_sec * sr)
    n_hops = max(1, int(np.ceil(len(y) / hop_len)))
    scores = []
    for i in range(n_hops):
        end = min((i + 1) * hop_len, len(y))
        start = max(0, end - window_len)
        y_window = y[start:end]
        if len(y_window) < int(0.5 * sr):
            scores.append(scores[-1] if scores else 0.3)
            continue
        perc_ratio, perc_loudness, note_shortness, dissonance, centroid = (
            REFERENCE_compute_combat_features_from_array(y_window, sr)
        )
        score = (
            0.30 * REFERENCE_normalize(perc_ratio, "perc_ratio") +
            0.25 * REFERENCE_normalize(perc_loudness, "percussive_loudness") +
            0.10 * REFERENCE_normalize(note_shortness, "note_shortness") +
            0.15 * REFERENCE_normalize(dissonance, "dissonance") +
            0.20 * REFERENCE_normalize(centroid, "spectral_centroid_mean")
        )
        scores.append(score)
    return np.asarray(scores, dtype=np.float32), hop_sec


def REFERENCE_align_causal_to_frames(window_scores, hop_sec, frame_rate, total_frames):
    hop_frames = max(1, round(hop_sec * frame_rate))
    aligned = np.empty(total_frames, dtype=np.float32)
    for i, score in enumerate(window_scores):
        start = i * hop_frames
        end = min(start + hop_frames, total_frames)
        if start >= total_frames:
            break
        aligned[start:end] = score
    filled_until = min(len(window_scores) * hop_frames, total_frames)
    if filled_until < total_frames:
        aligned[filled_until:] = window_scores[-1]
    return aligned


def REFERENCE_tension(wav_mono, num_frames):
    # Tension part of process_track(), unchanged.
    num_samples = wav_mono.shape[-1]
    frame_size = max(1, round(num_samples / num_frames))
    required_samples = num_frames * frame_size
    if num_samples < required_samples:
        wav_for_rms = torch.nn.functional.pad(wav_mono, (0, required_samples - num_samples))
    else:
        wav_for_rms = wav_mono[:required_samples]
    rms_frames = torch.sqrt(torch.mean(wav_for_rms.reshape(num_frames, frame_size) ** 2, dim=-1))
    tension = np.clip(
        (rms_frames.numpy() - NORM_STATS["tension_rms"]["p_low"]) /
        (NORM_STATS["tension_rms"]["p_high"] - NORM_STATS["tension_rms"]["p_low"] + 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    return tension


# ── Helpers ──
def make_test_audio(seconds=8.3, seed=0):
    """Noise bursts plus tones, peak-normalised to 0.95 like the tokenizer."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    y = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 277 * t)
    for onset in rng.uniform(0, seconds - 0.2, size=int(4 * seconds)):
        start = int(onset * SAMPLE_RATE)
        length = int(0.08 * SAMPLE_RATE)
        y[start:start + length] += rng.standard_normal(length) * np.exp(-np.arange(length) / 600)
    y = y / np.abs(y).max() * 0.95
    return y.astype(np.float32)


# ── Tests ──
def test_combat_score_matches_tokenizer():
    y = make_test_audio()
    expected, expected_hop = REFERENCE_compute_causal_combat_score_track(y, SAMPLE_RATE)
    observed, observed_hop = control_features.compute_causal_combat_score_track(
        y, SAMPLE_RATE, NORM_STATS
    )
    assert expected_hop == observed_hop
    assert np.array_equal(expected, observed), np.max(np.abs(expected - observed))

    with_components, _, components = control_features.compute_causal_combat_score_track(
        y, SAMPLE_RATE, NORM_STATS, return_components=True
    )
    assert np.array_equal(expected, with_components)
    assert components.shape == (len(expected), len(control_features.COMBAT_FEATURES))

    num_frames = int(np.ceil(len(y) / 320))
    assert np.array_equal(
        REFERENCE_align_causal_to_frames(expected, expected_hop, FRAME_RATE, num_frames),
        control_features.align_causal_to_frames(observed, observed_hop, FRAME_RATE, num_frames),
    )


def test_causal_truncation_gives_same_first_scores():
    y = make_test_audio(seconds=8.3, seed=1)
    full, _ = control_features.compute_causal_combat_score_track(y, SAMPLE_RATE, NORM_STATS)
    seconds = 5
    truncated, _ = control_features.compute_causal_combat_score_track(
        y[:seconds * SAMPLE_RATE], SAMPLE_RATE, NORM_STATS
    )
    assert len(truncated) == seconds
    assert np.array_equal(full[:seconds], truncated)


def test_tension_matches_tokenizer():
    wav_mono = torch.from_numpy(make_test_audio(seconds=12.7, seed=2))
    num_frames = int(np.ceil(wav_mono.shape[-1] / 320))
    expected = REFERENCE_tension(wav_mono, num_frames)
    observed = control_features.normalize_tension(
        control_features.frame_rms(wav_mono, num_frames), NORM_STATS["tension_rms"]
    )
    assert np.array_equal(expected, observed)


def test_fit_recovers_shared_and_minmax_statistics():
    wav_mono = torch.from_numpy(make_test_audio(seconds=12.7, seed=3))
    num_frames = int(np.ceil(wav_mono.shape[-1] / 320))

    rms = control_features.frame_rms(wav_mono, num_frames)
    for name, stats in control_features.KNOWN_TENSION_RMS_STATS.items():
        fit = control_features.fit_tension_stats(rms, control_features.normalize_tension(rms, stats))
        assert control_features.closest_known_tension_stats(fit)[0] == name
        assert fit["max_abs_residual"] < 1e-6

    # Early Colab pipeline: legacy frames, per-track min-max normalisation.
    legacy_rms = control_features.frame_rms_legacy(wav_mono, num_frames)
    assert legacy_rms.shape[-1] == num_frames
    minmax_tension = (legacy_rms - legacy_rms.min()) / (legacy_rms.max() - legacy_rms.min() + 1e-8)
    fit = control_features.fit_tension_stats(legacy_rms, minmax_tension)
    assert abs(fit["p_low"] - float(legacy_rms.min())) < 1e-6
    assert abs(fit["p_high"] - float(legacy_rms.max())) < 1e-6


def test_segment_score_equals_one_causal_window():
    # A 5-second segment: the last hop of the causal track uses exactly
    # the whole segment as its window, so the two scores must be equal.
    # The causal track stores its scores as float32, so the segment score
    # (a Python float) is compared after the same conversion.
    y = make_test_audio(seconds=5.0, seed=5)
    scores, _ = control_features.compute_causal_combat_score_track(y, SAMPLE_RATE, NORM_STATS)
    segment_score = control_features.combat_score_of_segment(y, SAMPLE_RATE, NORM_STATS)
    assert np.float32(segment_score) == scores[-1], (segment_score, scores[-1])


def test_soundfile_fallback_reads_same_samples_as_torchaudio():
    import tempfile

    import soundfile

    rng = np.random.default_rng(4)
    stereo = (rng.uniform(-0.8, 0.8, size=(44100, 2))).astype(np.float32)

    with tempfile.TemporaryDirectory() as directory:
        for subtype in ("PCM_16", "PCM_24"):
            path = Path(directory) / f"test_{subtype}.flac"
            soundfile.write(path, stereo, 44100, subtype=subtype)

            # Force the fallback: "import torchaudio" raises ImportError.
            saved = sys.modules.get("torchaudio")
            sys.modules["torchaudio"] = None
            try:
                fallback, fallback_rate = control_features.read_audio_file(path)
            finally:
                if saved is None:
                    del sys.modules["torchaudio"]
                else:
                    sys.modules["torchaudio"] = saved
            assert "soundfile" in control_features.audio_backends_used
            assert fallback.shape == (2, 44100) and fallback_rate == 44100

            try:
                import torchaudio
                reference, reference_rate = torchaudio.load(path)
            except (ImportError, OSError, RuntimeError):
                print(f"  torchaudio cannot decode here; fallback only checked ({subtype})")
                continue
            assert reference_rate == fallback_rate
            assert torch.max(torch.abs(reference - fallback)).item() <= 1e-6, subtype


if __name__ == "__main__":
    tests = [
        test_combat_score_matches_tokenizer,
        test_causal_truncation_gives_same_first_scores,
        test_tension_matches_tokenizer,
        test_fit_recovers_shared_and_minmax_statistics,
        test_segment_score_equals_one_causal_window,
        test_soundfile_fallback_reads_same_samples_as_torchaudio,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All control_features tests passed.")
