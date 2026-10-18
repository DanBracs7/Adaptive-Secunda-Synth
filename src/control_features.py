"""
Control signals computed exactly as in
notebooks/archive/new-slakh-token-normalized.ipynb (process_track), so that
controls measured on generated audio use the same definition as the
training conditions.

Tension: frame RMS normalised with norm_stats["tension_rms"].
Combat score: weighted sum of five normalised features over a causal
5-second window with a 1-second hop, held for 75 frames per hop.

The functions keep the tokenizer code unchanged; the only difference is
that norm_stats is passed as a parameter instead of read from a global.
tests/test_control_features.py checks them against verbatim copies.
"""

import json

import numpy as np
import torch


SAMPLE_RATE = 24_000
FRAME_RATE = 75
PEAK_TARGET = 0.95

# Values of norm_stats["tension_rms"] used to encode the tensors. The
# tokenizer notebook re-estimates them in every run from randomly
# subsampled RMS frames (unseeded torch.randperm in collect_rms_samples),
# so each run produced slightly different values.
KNOWN_TENSION_RMS_STATS = {
    # configs/norm_stats.json, commit 6e53de4
    "v1": {"p_low": 0.010234275157563394, "p_high": 0.21666884757578356},
    # configs/norm_stats.json, commit 649994f (current file)
    "v2": {"p_low": 0.007976212748326363, "p_high": 0.22036580294370645},
    # printed by new-slakh-token-normalized.ipynb, run of 2026-09-05,
    # which encoded train_1070.pt to train_1394.pt
    "v3": {"p_low": 0.00793589388485998, "p_high": 0.21743767708539957},
    # v4 and v5 were never saved to a file or printed. They were recovered
    # by notebooks/02_control_statistics_audit.ipynb (Part B) by fitting the
    # stored tension against the recomputed frame RMS, and are rounded to
    # 6 decimals as printed there (the fit itself is accurate to ~1e-8).
    # v4: Slakh train_400.pt to train_750.pt (every 25th tensor checked)
    "v4": {"p_low": 0.009767, "p_high": 0.216359},
    # v5: Slakh train_800.pt to train_1050.pt (every 25th tensor checked)
    "v5": {"p_low": 0.007392, "p_high": 0.219332},
}


def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Decoders used by read_audio_file in this session, reported by notebook 02.
audio_backends_used = set()


def read_audio_file(file_path):
    """
    Return (waveform [channels, N] float32, sample_rate).

    Uses torchaudio.load, as the tokenizer did. Recent torchaudio versions
    need the torchcodec package to decode; when it is unavailable the file
    is read with soundfile instead. For lossless files (FLAC, WAV) both
    decoders return the same samples (tests/test_control_features.py).
    """
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(file_path)
        audio_backends_used.add("torchaudio")
        return waveform, sample_rate
    except (ImportError, OSError, RuntimeError):
        import soundfile

        data, sample_rate = soundfile.read(
            str(file_path),
            dtype="float32",
            always_2d=True,
        )
        audio_backends_used.add("soundfile")
        return torch.from_numpy(data.T.copy()), sample_rate


def load_waveform_like_tokenizer(file_path):
    """
    Load an audio file with the same steps as process_track(): mono,
    EnCodec conversion to 24 kHz, peak normalisation to 0.95.
    Returns wav_mono with shape [N].
    """
    from encodec.utils import convert_audio

    waveform, source_sample_rate = read_audio_file(file_path)

    # Explicit mono conversion before EnCodec
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = convert_audio(
        waveform,
        source_sample_rate,
        SAMPLE_RATE,
        1,
    )

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * PEAK_TARGET

    return waveform[0].cpu()


def frame_rms(wav_mono, num_frames):
    """
    Per-frame RMS aligned to `num_frames` codec frames, as in
    process_track(): the frame size is derived from the data and the
    waveform is padded or trimmed to num_frames * frame_size samples.
    """
    num_samples = wav_mono.shape[-1]

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    frame_size = max(1, round(num_samples / num_frames))

    required_samples = num_frames * frame_size
    if num_samples < required_samples:
        wav_for_rms = torch.nn.functional.pad(
            wav_mono,
            (0, required_samples - num_samples),
        )
    else:
        wav_for_rms = wav_mono[:required_samples]

    rms_frames = torch.sqrt(
        torch.mean(
            wav_for_rms.reshape(num_frames, frame_size) ** 2,
            dim=-1,
        )
    )

    return rms_frames


def normalize_tension(rms_frames, tension_stats):
    """Map frame RMS to tension in [0, 1] with given p_low / p_high."""
    rms = torch.as_tensor(rms_frames).numpy()

    tension = np.clip(
        (rms - tension_stats["p_low"]) /
        (tension_stats["p_high"] - tension_stats["p_low"] + 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)

    return tension


def fit_tension_stats(rms_frames, tension, margin=1e-4):
    """
    Recover the p_low / p_high used to normalise a stored tension curve.

    On frames that were not clipped, tension is a linear function of the
    frame RMS: rms = p_low + tension * (p_high - p_low + 1e-8). A
    least-squares line through those frames gives both statistics. The
    residual is ~1e-7 when the RMS was recomputed with the same pipeline;
    a large residual means the pipeline (or the audio file) differs.
    """
    rms = np.asarray(torch.as_tensor(rms_frames).numpy(), dtype=np.float64)
    tension = np.asarray(torch.as_tensor(tension).numpy(), dtype=np.float64)

    if rms.shape != tension.shape:
        raise ValueError(
            f"rms {rms.shape} and tension {tension.shape} differ in length."
        )

    unclipped = (tension > margin) & (tension < 1.0 - margin)

    if unclipped.sum() < 10:
        return None

    slope, intercept = np.polyfit(tension[unclipped], rms[unclipped], 1)
    residual = rms[unclipped] - (intercept + slope * tension[unclipped])

    return {
        "p_low": float(intercept),
        "p_high": float(intercept + slope - 1e-8),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "unclipped_frames": int(unclipped.sum()),
        "total_frames": int(len(tension)),
    }


def closest_known_tension_stats(fitted_stats):
    """Name of the KNOWN_TENSION_RMS_STATS entry closest to a fit."""
    best_name = None
    best_distance = float("inf")

    for name, stats in KNOWN_TENSION_RMS_STATS.items():
        distance = max(
            abs(stats["p_low"] - fitted_stats["p_low"]),
            abs(stats["p_high"] - fitted_stats["p_high"]),
        )
        if distance < best_distance:
            best_name, best_distance = name, distance

    return best_name, best_distance


def frame_rms_legacy(wav_mono, num_frames):
    """
    Frame RMS of the early Colab tokenizer
    (notebooks/archive/colab_ost_features_early_tokenizer.ipynb, cell 2):
    floor frame size, padding to a multiple of it, unfold, then truncation
    to the token length. Used only to recognise tensors produced by that
    pipeline; the floor frame size is 319 instead of 320 samples for most
    tracks, so these frames drift away from the codec frames.
    """
    num_samples = wav_mono.shape[-1]
    frame_size = num_samples // num_frames if num_frames > 0 else 1

    pad_len = (-num_samples) % frame_size
    wav_padded = (
        torch.nn.functional.pad(wav_mono, (0, pad_len))
        if pad_len > 0 else wav_mono
    )

    rms_frames = torch.sqrt(
        torch.mean(wav_padded.unfold(-1, frame_size, frame_size) ** 2, dim=-1)
    )

    return rms_frames[:num_frames]


# ── Combat score (causal rolling window) ──

COMBAT_FEATURES = (
    "perc_ratio",
    "percussive_loudness",
    "note_shortness",
    "dissonance",
    "spectral_centroid_mean",
)

COMBAT_WEIGHTS = {
    "perc_ratio": 0.30,
    "percussive_loudness": 0.25,
    "note_shortness": 0.10,
    "dissonance": 0.15,
    "spectral_centroid_mean": 0.20,
}


def normalize(value, feature_name, norm_stats):
    stats = norm_stats[feature_name]
    return float(np.clip(
        (value - stats["p_low"]) /
        (stats["p_high"] - stats["p_low"] + 1e-8),
        0.0,
        1.0,
    ))


def compute_combat_features_from_array(y, sr):
    import librosa

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    perc_ratio = percussive_energy / (
        harmonic_energy + percussive_energy + 1e-8
    )

    perc_loudness = float(np.sqrt(np.mean(y_percussive ** 2)))

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    ioi = float(np.mean(np.diff(onsets))) if len(onsets) >= 2 else 1.5
    note_shortness = float(np.clip(1.0 - ioi / 1.5, 0.0, 1.0))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    adjacent_energy = np.sum(chroma[:-1] * chroma[1:], axis=0)
    dissonance = float(np.mean(
        adjacent_energy / (np.sum(chroma, axis=0) + 1e-8)
    ))

    centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    ))

    return perc_ratio, perc_loudness, note_shortness, dissonance, centroid


def compute_causal_combat_score_track(
    y,
    sr,
    norm_stats,
    window_sec=5.0,
    hop_sec=1.0,
    return_components=False,
):
    """
    Each score at time t uses only audio in [t - 5 seconds, t].
    No centered/future window leakage.

    With return_components=True it also returns the raw (unnormalised)
    features of every hop, shape [n_hops, 5] in COMBAT_FEATURES order;
    hops that reuse the previous score (window shorter than 0.5 s) are NaN.
    """
    hop_len = int(hop_sec * sr)
    window_len = int(window_sec * sr)
    n_hops = max(1, int(np.ceil(len(y) / hop_len)))

    scores = []
    components = []

    for i in range(n_hops):
        end = min((i + 1) * hop_len, len(y))
        start = max(0, end - window_len)
        y_window = y[start:end]

        if len(y_window) < int(0.5 * sr):
            scores.append(scores[-1] if scores else 0.3)
            components.append([np.nan] * len(COMBAT_FEATURES))
            continue

        perc_ratio, perc_loudness, note_shortness, dissonance, centroid = (
            compute_combat_features_from_array(y_window, sr)
        )

        score = (
            0.30 * normalize(perc_ratio, "perc_ratio", norm_stats) +
            0.25 * normalize(perc_loudness, "percussive_loudness", norm_stats) +
            0.10 * normalize(note_shortness, "note_shortness", norm_stats) +
            0.15 * normalize(dissonance, "dissonance", norm_stats) +
            0.20 * normalize(centroid, "spectral_centroid_mean", norm_stats)
        )
        scores.append(score)
        components.append(
            [perc_ratio, perc_loudness, note_shortness, dissonance, centroid]
        )

    scores = np.asarray(scores, dtype=np.float32)

    if return_components:
        return scores, hop_sec, np.asarray(components, dtype=np.float64)

    return scores, hop_sec


def combat_score_of_segment(y, sr, norm_stats, return_components=False):
    """
    Combat score of one audio segment, computed exactly like one window of
    compute_causal_combat_score_track (same features, normalisation and
    weights), with the whole segment as the window.

    Used on generated clips, where the windows of interest are chosen by
    the experiment (e.g. before and after a requested change) instead of
    the tokenizer's rolling 5-second window.
    """
    if len(y) < int(0.5 * sr):
        raise ValueError("The segment must be at least 0.5 s long.")

    perc_ratio, perc_loudness, note_shortness, dissonance, centroid = (
        compute_combat_features_from_array(y, sr)
    )

    score = (
        0.30 * normalize(perc_ratio, "perc_ratio", norm_stats) +
        0.25 * normalize(perc_loudness, "percussive_loudness", norm_stats) +
        0.10 * normalize(note_shortness, "note_shortness", norm_stats) +
        0.15 * normalize(dissonance, "dissonance", norm_stats) +
        0.20 * normalize(centroid, "spectral_centroid_mean", norm_stats)
    )

    if return_components:
        return score, [perc_ratio, perc_loudness, note_shortness, dissonance, centroid]

    return score


def align_causal_to_frames(window_scores, hop_sec, frame_rate, total_frames):
    """
    Causal sample-and-hold alignment:
    one rolling score lasts until the next score becomes available.
    """
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
