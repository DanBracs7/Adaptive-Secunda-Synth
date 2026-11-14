"""
Tests for src/demo_clips.py (no tensors, checkpoint or codec needed).

1. control_track keeps the prompt's own values, ramps linearly to the
   target and stays at the target afterwards; with target=None it stays
   at the prompt's final level.
2. The pilot passages exist in configs/demo_prompts.csv, half chill and
   half fight.
3. Every sweep setting changes only options of generate_clip, "base" is
   the pilot, and the per-codebook temperatures have one value per
   codebook.
4. By default a bank clip and its prompt fit in one 10-s window, and the
   bank covers all passages of configs/demo_prompts.csv.
5. Loops: crossfade_concat has the expected length and leaves the audio
   outside the crossfades unchanged; write_loop joins the takes from
   their WAV files; --labels and --tag select the passages and the folder.

Run from the repository root:
    python tests/test_demo_clips.py
"""

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import demo_clips  # noqa: E402


def test_control_track():
    prompt_values = torch.full((225,), 0.6)
    track = demo_clips.control_track(prompt_values, target=0.2, total_frames=1350, ramp_frames=75)
    assert track.shape == (1350,)
    assert torch.allclose(track[:225], prompt_values)
    assert abs(float(track[225]) - 0.6) < 1e-6                  # ramp starts at the prompt level
    assert torch.all(track[225:300][1:] <= track[225:300][:-1])  # decreasing ramp
    assert torch.allclose(track[300:], torch.tensor(0.2))


def test_control_track_holds_prompt_level():
    prompt_values = torch.cat([torch.full((150,), 0.2), torch.full((75,), 0.7)])
    track = demo_clips.control_track(prompt_values, target=None, total_frames=1350, ramp_frames=75)
    assert torch.allclose(track[:225], prompt_values)
    assert torch.allclose(track[225:], torch.tensor(0.7))       # mean of the last 0.5 s


def test_pilot_passages():
    passages = demo_clips.load_passages(demo_clips.PILOT_PASSAGES)
    labels = [p["label"] for p in passages]
    assert labels.count("chill") == 3 and labels.count("fight") == 3


def test_sweep_settings():
    options = set(inspect.signature(demo_clips.generate_clip).parameters)
    assert demo_clips.SWEEP_SETTINGS["base"] == {}
    for name, setting in demo_clips.SWEEP_SETTINGS.items():
        assert set(setting) <= options, name
        assert len(setting) <= 1, name                            # one change from base
        temperature = setting.get("temperature", 1.0)
        if not isinstance(temperature, (int, float)):
            assert len(temperature) == demo_clips.NUM_CODEBOOKS, name


def test_bank_defaults():
    prompt_seconds, generated_seconds = demo_clips.DEFAULT_SECONDS["bank"]
    assert prompt_seconds + generated_seconds <= 10.0
    labels = [p["label"] for p in demo_clips.load_all_passages()]
    assert len(labels) == 36 and set(labels) == {"chill", "fight"}


def test_crossfade_concat():
    a, b, c = np.full(1000, 0.5), np.full(800, -0.3), np.full(600, 0.1)
    joined = demo_clips.crossfade_concat([a, b, c], crossfade_samples=100)
    assert len(joined) == 1000 + 800 + 600 - 2 * 100
    assert np.allclose(joined[:900], 0.5)                          # before the first crossfade
    assert np.allclose(joined[1000:1600], -0.3)                    # between the crossfades
    assert np.allclose(joined[-500:], 0.1)                         # after the last crossfade
    assert np.allclose(demo_clips.crossfade_concat([a], 100), a)   # one take: unchanged


def test_write_loop():
    rate = demo_clips.SAMPLE_RATE
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        for take in range(3):
            soundfile.write(str(folder / f"fight_x_0m00_t{take + 1}_generated.wav"),
                            np.full(3 * rate, 0.1 * (take + 1)), rate, subtype="PCM_16")
        fields = demo_clips.write_loop(folder, "fight_x_0m00", [f"fight_x_0m00_t{t}" for t in (1, 2, 3)])
        loop, loop_rate = soundfile.read(str(folder / "fight_x_0m00_loop.wav"))
    crossfade = int(demo_clips.LOOP_CROSSFADE_SECONDS * rate)
    assert loop_rate == rate and len(loop) == 3 * 3 * rate - 2 * crossfade
    assert fields["generated_file"] == "fight_x_0m00_loop.wav"
    assert fields["with_prompt_file"] == "fight_x_0m00_t1_with_prompt.wav"
    assert fields["takes"] == 3 and fields["clipped_fraction"] == 0.0


def test_bank_selection():
    fights = demo_clips.bank_passages(["fight"])
    assert fights and all(p["label"] == "fight" for p in fights)
    assert len(demo_clips.bank_passages()) == len(demo_clips.load_all_passages())
    assert demo_clips.bank_folder("base").name == "base"
    assert demo_clips.bank_folder("base", "3s").name == "base_3s"


if __name__ == "__main__":
    for test in (test_control_track, test_control_track_holds_prompt_level, test_pilot_passages,
                 test_sweep_settings, test_bank_defaults, test_crossfade_concat, test_write_loop,
                 test_bank_selection):
        test()
        print("PASS", test.__name__)
    print("All demo_clips tests passed.")
