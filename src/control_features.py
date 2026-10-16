"""
Control signals computed exactly as in
notebooks/archive/new-slakh-token-normalized.ipynb (process_track), so that
controls measured on generated audio use the same definition as the
training conditions.

This module currently contains the tension part. The combat-score
functions are added in Phase 2.
"""

import json

import numpy as np
import torch


SAMPLE_RATE = 24_000
FRAME_RATE = 75
PEAK_TARGET = 0.95

# Values of norm_stats["tension_rms"] found in the project history. The
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
}


def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_waveform_like_tokenizer(file_path):
    """
    Load an audio file with the same steps as process_track(): mono,
    EnCodec conversion to 24 kHz, peak normalisation to 0.95.
    Returns wav_mono with shape [N].
    """
    import torchaudio
    from encodec.utils import convert_audio

    waveform, source_sample_rate = torchaudio.load(file_path)

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
