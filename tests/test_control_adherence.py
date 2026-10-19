"""
Tests for src/control_adherence.py.

1. requested_controls builds the factorial and step tracks with the
   chosen levels and keeps the prompt's own controls on the prompt frames.
2. measure_clip measures a louder second half as higher post-step tension.
3. factorial_effects recovers known effects from synthetic measurements:
   a tension gain of 0.5, no combat effect and no cross-talk.
4. step_effects recovers a known step response.
5. step_latency_seconds finds the delay of a delayed step.
6. combat_effect_by_tension recovers a combat effect present only at
   high tension.

Run from the repository root:
    python tests/test_control_adherence.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import control_adherence as ca  # noqa: E402
from control_features import KNOWN_TENSION_RMS_STATS  # noqa: E402

with open(PROJECT_ROOT / "configs" / "norm_stats.json", "r", encoding="utf-8") as file:
    NORM_STATS = json.load(file)


def test_requested_controls():
    prompt_tension = torch.full((ca.PROMPT_FRAMES,), 0.33)
    prompt_combat = torch.full((ca.PROMPT_FRAMES,), 0.44)

    tension, combat = ca.requested_controls("tH_cL", prompt_tension, prompt_combat)
    assert tension.shape == (ca.TOTAL_FRAMES,)
    assert torch.all(tension[:ca.PROMPT_FRAMES] == 0.33) and torch.all(combat[:ca.PROMPT_FRAMES] == 0.44)
    assert torch.allclose(tension[ca.PROMPT_FRAMES:], torch.tensor(ca.LEVELS["tension"]["high"]))
    assert torch.allclose(combat[ca.PROMPT_FRAMES:], torch.tensor(ca.LEVELS["combat_score"]["low"]))

    tension, combat = ca.requested_controls("combat_step", prompt_tension, prompt_combat)
    assert torch.allclose(tension[ca.PROMPT_FRAMES:], torch.tensor(ca.LEVELS["tension"]["mid"]))
    assert torch.allclose(combat[ca.PROMPT_FRAMES:ca.STEP_FRAME], torch.tensor(ca.LEVELS["combat_score"]["low"]))
    assert torch.allclose(combat[ca.STEP_FRAME:], torch.tensor(ca.LEVELS["combat_score"]["high"]))


def test_measure_clip_sees_a_louder_second_half():
    rng = np.random.default_rng(0)
    n = ca.TOTAL_FRAMES * ca.SAMPLES_PER_FRAME
    audio = rng.standard_normal(n).astype(np.float32) * 0.02
    audio[ca.STEP_FRAME * ca.SAMPLES_PER_FRAME:] *= 8
    record, curve = ca.measure_clip(audio, NORM_STATS, KNOWN_TENSION_RMS_STATS["v1"])
    assert curve.shape == (ca.TOTAL_FRAMES,)
    assert record["tension_post"] > record["tension_pre"] + 0.3
    for name in ca.SPANS:
        assert 0.0 <= record[f"combat_{name}"] <= 1.0


def synthetic_clips(tension_gain=0.5, combat_gain=0.0, units=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(units):
        offset_t, offset_c = rng.uniform(0, 0.2, size=2)
        for condition in ca.FACTORIAL_CONDITIONS:
            requested_t = ca.LEVELS["tension"]["low" if condition[1] == "L" else "high"]
            requested_c = ca.LEVELS["combat_score"]["low" if condition[4] == "L" else "high"]
            rows.append({
                "unit": unit, "condition": condition,
                "tension_full": offset_t + tension_gain * requested_t,
                "combat_full": offset_c + combat_gain * requested_c,
            })
        for condition in ca.STEP_CONDITIONS:
            stepped_tension = condition == "tension_step"
            rows.append({
                "unit": unit, "condition": condition,
                "tension_pre": offset_t + tension_gain * (ca.LEVELS["tension"]["low"] if stepped_tension else 0.41),
                "tension_post": offset_t + tension_gain * (ca.LEVELS["tension"]["high"] if stepped_tension else 0.41),
                "combat_pre": offset_c, "combat_post": offset_c,
            })
    return pd.DataFrame(rows)


def test_factorial_effects_recover_known_effects():
    effects = ca.factorial_effects(synthetic_clips()).set_index(["requested", "measured"])
    assert abs(effects.loc[("tension", "tension"), "gain"] - 0.5) < 1e-9
    assert abs(effects.loc[("combat_score", "combat_score"), "effect"]) < 1e-9
    assert abs(effects.loc[("tension", "combat_score"), "effect"]) < 1e-9
    assert abs(effects.loc[("combat_score", "tension"), "effect"]) < 1e-9
    assert effects.loc[("tension", "tension"), "positive_units"] == 1.0


def test_step_effects_recover_known_step():
    effects = ca.step_effects(synthetic_clips()).set_index(["condition", "measured"])
    assert abs(effects.loc[("tension_step", "tension"), "gain"] - 0.5) < 1e-9
    assert abs(effects.loc[("combat_step", "combat_score"), "effect"]) < 1e-9


def test_step_latency():
    curve = np.full(ca.TOTAL_FRAMES, 0.2)
    curve[ca.STEP_FRAME + 30:] = 0.8          # rises 0.4 s after the step
    assert abs(ca.step_latency_seconds(curve) - 30 / ca.FRAME_RATE) < 1e-9
    assert ca.step_latency_seconds(np.full(ca.TOTAL_FRAMES, 0.5)) is None
    flat_with_noise = 0.5 + 1e-6 * np.random.default_rng(0).standard_normal(ca.TOTAL_FRAMES)
    assert ca.step_latency_seconds(flat_with_noise) is None


def test_combat_effect_by_tension():
    rows = []
    for unit in range(5):
        for condition in ca.FACTORIAL_CONDITIONS:
            high_tension, high_combat = condition[1] == "H", condition[4] == "H"
            # Combat request raises the measured combat score by 0.2, only at high tension.
            rows.append({"unit": unit, "condition": condition,
                         "combat_full": 0.3 + 0.01 * unit + (0.2 if high_tension and high_combat else 0.0)})
    effects = ca.combat_effect_by_tension(pd.DataFrame(rows)).set_index("comparison")
    assert abs(effects.loc["combat effect at low tension", "effect"]) < 1e-9
    assert abs(effects.loc["combat effect at high tension", "effect"] - 0.2) < 1e-9
    assert abs(effects.loc["interaction (high - low tension)", "effect"] - 0.2) < 1e-9


if __name__ == "__main__":
    tests = [
        test_requested_controls,
        test_measure_clip_sees_a_louder_second_half,
        test_factorial_effects_recover_known_effects,
        test_step_effects_recover_known_step,
        test_step_latency,
        test_combat_effect_by_tension,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All control_adherence tests passed.")
