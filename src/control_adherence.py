"""
Experiment E2: does the generated audio follow the requested controls?

E1 showed that the model uses the conditions to predict real audio. E2
asks the question of the title directly: when the controls are changed at
generation time, does the generated music change as requested? Tension
and combat score are measured on the generated audio with the same
definitions used to build the training conditions (src/control_features.py)
and compared with the requested values.

Design (paired): for every prompt and seed, all conditions are generated
with the same random numbers (generation.generate_codes, shared_noise=True),
so differences between them are caused by the controls only.

  * 2 x 2 factorial, constant controls after the prompt:
    tension {low, high} x combat_score {low, high}. Because E1 found that
    the model relies almost only on tension, the two controls must be
    varied independently to tell their effects apart.
  * Two steps at 5 s: tension low -> high (combat_score fixed at its
    median) and combat_score low -> high (tension fixed at its median),
    to measure whether and how fast the output follows a change.

Measurements on each decoded clip (no peak normalisation):
  * tension: frame RMS mapped with the training statistics, per frame;
  * combat score: one combat-score window over each analysis span
    (control_features.combat_score_of_segment).
Analysis spans (frames at 75 Hz, prompt = first second):
  full = 1-10 s, pre = 1-5 s, post = 5.5-10 s (0.5 s left for the change).
"""

import numpy as np
import torch

from condition_ablation import bootstrap_mean_ci
from control_features import (
    combat_score_of_segment,
    frame_rms,
    normalize_tension,
)
from generation import constant_control, step_control, with_prompt_controls


FRAME_RATE = 75
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE  # 320

TOTAL_FRAMES = 750          # 10 s, the training segment length
PROMPT_FRAMES = 75          # 1 s of real audio
STEP_FRAME = 375            # 5 s
SETTLE_FRAMES = 38          # 0.5 s left to the model to follow a step

SPANS = {
    "full": (PROMPT_FRAMES, TOTAL_FRAMES),
    "pre": (PROMPT_FRAMES, STEP_FRAME),
    "post": (STEP_FRAME + SETTLE_FRAMES, TOTAL_FRAMES),
}

# Control levels: 10th / 50th / 90th percentiles of the Slakh train+val
# conditions (notebook 02, Part A). The same levels are used for every
# checkpoint so that the results are comparable.
LEVELS = {
    "tension": {"low": 0.14, "mid": 0.41, "high": 0.77},
    "combat_score": {"low": 0.21, "mid": 0.60, "high": 0.85},
}

FACTORIAL_CONDITIONS = ("tL_cL", "tL_cH", "tH_cL", "tH_cH")
STEP_CONDITIONS = ("tension_step", "combat_step")
CONDITIONS = FACTORIAL_CONDITIONS + STEP_CONDITIONS


def requested_controls(condition, prompt_tension, prompt_combat, total_frames=TOTAL_FRAMES):
    """
    Requested (tension, combat_score) tracks [T] for a condition; the
    prompt frames keep the prompt's own controls.
    """
    tension_levels, combat_levels = LEVELS["tension"], LEVELS["combat_score"]

    if condition in FACTORIAL_CONDITIONS:
        tension_level = tension_levels["low" if condition[1] == "L" else "high"]
        combat_level = combat_levels["low" if condition[4] == "L" else "high"]
        tension = constant_control(tension_level, total_frames)
        combat = constant_control(combat_level, total_frames)
    elif condition == "tension_step":
        tension = step_control(tension_levels["low"], tension_levels["high"], total_frames, STEP_FRAME)
        combat = constant_control(combat_levels["mid"], total_frames)
    elif condition == "combat_step":
        tension = constant_control(tension_levels["mid"], total_frames)
        combat = step_control(combat_levels["low"], combat_levels["high"], total_frames, STEP_FRAME)
    else:
        raise ValueError(f"Unknown condition '{condition}'. Known: {CONDITIONS}")

    return with_prompt_controls(tension, prompt_tension), with_prompt_controls(combat, prompt_combat)


def measured_tension_curve(waveform, tension_stats, total_frames=TOTAL_FRAMES):
    """Per-frame tension of a decoded clip [N] (frame RMS, training normalisation)."""
    waveform = torch.as_tensor(waveform).float()
    return normalize_tension(frame_rms(waveform, total_frames), tension_stats)


def measure_clip(waveform, norm_stats, tension_stats, spans=SPANS):
    """
    Tension and combat score of one decoded clip [N] on every span.
    Returns (record dict, tension curve [T]).
    """
    audio = np.asarray(torch.as_tensor(waveform).float().cpu(), dtype=np.float32)
    curve = measured_tension_curve(audio, tension_stats)
    record = {}

    for name, (start, end) in spans.items():
        record[f"tension_{name}"] = float(curve[start:end].mean())
        segment = audio[start * SAMPLES_PER_FRAME:end * SAMPLES_PER_FRAME]
        record[f"combat_{name}"] = float(combat_score_of_segment(segment, SAMPLE_RATE, norm_stats))

    return record, curve


def requested_span_means(tension, combat_score, spans=SPANS):
    record = {}
    tension = np.asarray(tension, dtype=np.float64)
    combat_score = np.asarray(combat_score, dtype=np.float64)
    for name, (start, end) in spans.items():
        record[f"requested_tension_{name}"] = float(tension[start:end].mean())
        record[f"requested_combat_{name}"] = float(combat_score[start:end].mean())
    return record


def paired_effect(differences, requested_difference=None, seed=0):
    """
    Mean of paired differences over units (prompt x seed) with a bootstrap
    95% CI, the fraction of units with a positive difference and, if the
    requested difference is given, the gain (1 = the measured change equals
    the requested change).
    """
    differences = np.asarray(differences, dtype=np.float64)
    mean, low, high = bootstrap_mean_ci(differences, seed=seed)
    result = {
        "units": int(len(differences)),
        "effect": mean,
        "ci_low": low,
        "ci_high": high,
        "positive_units": float(np.mean(differences > 0)),
    }
    if requested_difference is not None:
        result["gain"] = mean / requested_difference
    return result


def factorial_effects(clips):
    """
    clips: DataFrame with one row per clip and columns unit, condition,
    tension_full, combat_full. Returns one row per (requested control ->
    measured control) effect, averaged over the levels of the other control.
    """
    import pandas as pd

    table = clips[clips["condition"].isin(FACTORIAL_CONDITIONS)].pivot(
        index="unit", columns="condition", values=["tension_full", "combat_full"]
    )
    requested_tension_difference = LEVELS["tension"]["high"] - LEVELS["tension"]["low"]
    requested_combat_difference = LEVELS["combat_score"]["high"] - LEVELS["combat_score"]["low"]

    rows = []
    for measured, column in (("tension", "tension_full"), ("combat_score", "combat_full")):
        values = table[column]
        tension_effect = ((values["tH_cL"] - values["tL_cL"]) + (values["tH_cH"] - values["tL_cH"])) / 2
        combat_effect = ((values["tL_cH"] - values["tL_cL"]) + (values["tH_cH"] - values["tH_cL"])) / 2
        rows.append({
            "requested": "tension", "measured": measured,
            **paired_effect(tension_effect, requested_tension_difference if measured == "tension" else None),
        })
        rows.append({
            "requested": "combat_score", "measured": measured,
            **paired_effect(combat_effect, requested_combat_difference if measured == "combat_score" else None),
        })

    return pd.DataFrame(rows)


def step_effects(clips):
    """
    Post-step minus pre-step change of both measured controls, for the two
    step conditions. clips: one row per clip with columns unit, condition,
    tension_pre/post, combat_pre/post.
    """
    import pandas as pd

    requested = {
        "tension_step": ("tension", LEVELS["tension"]["high"] - LEVELS["tension"]["low"]),
        "combat_step": ("combat_score", LEVELS["combat_score"]["high"] - LEVELS["combat_score"]["low"]),
    }
    rows = []
    for condition, (stepped_control, requested_difference) in requested.items():
        subset = clips[clips["condition"] == condition].set_index("unit")
        for measured, prefix in (("tension", "tension"), ("combat_score", "combat")):
            differences = subset[f"{prefix}_post"] - subset[f"{prefix}_pre"]
            rows.append({
                "condition": condition, "measured": measured,
                **paired_effect(differences, requested_difference if measured == stepped_control else None),
            })
    return pd.DataFrame(rows)


def step_latency_seconds(curve, step_frame=STEP_FRAME, pre=SPANS["pre"], post=SPANS["post"], min_rise=0.05):
    """
    Time after the step at which a tension curve (averaged over units)
    first reaches the midpoint between its pre-step and post-step means.
    No smoothing is applied, so the estimate is not shifted in time.
    Returns None if the curve rises by less than min_rise (no step to time).
    """
    curve = np.asarray(curve, dtype=np.float64)
    pre_mean = curve[pre[0]:pre[1]].mean()
    post_mean = curve[post[0]:post[1]].mean()
    if post_mean - pre_mean < min_rise:
        return None
    midpoint = (pre_mean + post_mean) / 2
    after = np.nonzero(curve[step_frame:] >= midpoint)[0]
    if len(after) == 0:
        return None
    return float(after[0] / FRAME_RATE)
