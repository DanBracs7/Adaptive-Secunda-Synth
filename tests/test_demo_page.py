"""
Tests for src/demo_page.py (synthetic WAV files, no model needed).

1. Only the clips marked keep = 1 are used, a loop is found through
   loops.csv, and only generated audio is embedded (never a with-prompt
   file).
2. A selection without a clip of one label is refused.
3. The E2 section embeds one clip per condition and prompt, and is left
   out when no prompt is given.

Run from the repository root:
    python tests/test_demo_page.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import demo_page  # noqa: E402

RATE = demo_page.SAMPLE_RATE


def write_wav(path, seconds=1.0, frequency=220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    soundfile.write(str(path), 0.2 * np.sin(2 * np.pi * frequency * t), RATE, subtype="PCM_16")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_bank(root):
    root = Path(root)
    (root / "base").mkdir()
    (root / "base_3s").mkdir()
    clips = [("chill_a_0m00_t1", "chill", "A"), ("chill_b_0m10_t1", "chill", "B"), ("fight_c_0m00_t1", "fight", "C")]
    for clip, _, _ in clips:
        write_wav(root / "base" / f"{clip}_generated.wav")
        write_wav(root / "base" / f"{clip}_with_prompt.wav", seconds=2.0)
    write_csv(root / "base" / "clips.csv", [
        {"setting": "base", "clip": clip, "label": label, "title": title, "prompt_start": "0:00", "temperature": "1.0"}
        for clip, label, title in clips])
    write_wav(root / "base_3s" / "fight_d_0m00_loop.wav", seconds=2.0)
    write_csv(root / "base_3s" / "loops.csv", [
        {"setting": "base", "clip": "fight_d_0m00_loop", "label": "fight", "title": "D", "prompt_start": "0:00",
         "temperature": "1.0", "generated_file": "fight_d_0m00_loop.wav",
         "with_prompt_file": "fight_d_0m00_t1_with_prompt.wav", "takes": "4"}])
    selection = [
        {"setting": "base", "clip": "chill_a_0m00_t1", "label": "chill", "title": "A", "prompt_start": "0:00", "keep": "1", "note": ""},
        {"setting": "base", "clip": "chill_b_0m10_t1", "label": "chill", "title": "B", "prompt_start": "0:10", "keep": "0", "note": "noisy"},
        {"setting": "base", "clip": "fight_c_0m00_t1", "label": "fight", "title": "C", "prompt_start": "0:00", "keep": "1", "note": ""},
        {"setting": "base_3s", "clip": "fight_d_0m00_loop", "label": "fight", "title": "D", "prompt_start": "0:00", "keep": "1", "note": ""},
    ]
    write_csv(root / "demo_selection.csv", selection)
    return root


def test_pools_use_kept_generated_audio_only():
    with tempfile.TemporaryDirectory() as root:
        root = make_bank(root)
        pools, audio = demo_page.collect_pools(demo_page.load_selection(root / "demo_selection.csv"), root)
    assert [c["title"] for c in pools["chill"]] == ["A"]
    assert [c["title"] for c in pools["fight"]] == ["C", "D"]
    assert set(audio) == {"bank/base/chill_a_0m00_t1_generated.wav", "bank/base/fight_c_0m00_t1_generated.wav",
                          "bank/base_3s/fight_d_0m00_loop.wav"}
    assert not any("with_prompt" in key for key in audio)
    assert all(len(value) > 100 for value in audio.values())        # non-empty MP3, base64


def test_missing_label_is_refused():
    with tempfile.TemporaryDirectory() as root:
        root = make_bank(root)
        selection = [row for row in demo_page.load_selection(root / "demo_selection.csv") if row["label"] == "fight"]
        try:
            demo_page.collect_pools(selection, root)
        except ValueError:
            return
    raise AssertionError("ValueError not raised")


def test_experiment_section():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        for prompt in (3, 4):
            for condition, _ in demo_page.EXPERIMENT_CONDITIONS:
                write_wav(demo_page.clip_path(folder, "Slakh-FT", prompt, 0, condition))
        examples, audio = demo_page.collect_experiment((3, 4), folder)
        (folder / "bank").mkdir()
        bank = make_bank(folder / "bank")
        pools, pool_audio = demo_page.collect_pools(demo_page.load_selection(bank / "demo_selection.csv"), bank)
    assert len(examples) == 2 and len(audio) == 2 * len(demo_page.EXPERIMENT_CONDITIONS)
    page = demo_page.build_page(pools, examples, {**pool_audio, **audio})
    assert "Same prompt, different requests" in page and page.count("<audio controls") == 6
    assert json.loads(page.split("const POOLS = ", 1)[1].split(";\n", 1)[0]).keys() == {"chill", "fight"}
    assert demo_page.REPOSITORY_URL in page
    page_without = demo_page.build_page(pools, [], pool_audio)
    assert "Same prompt, different requests" not in page_without and "<audio controls" not in page_without


if __name__ == "__main__":
    for test in (test_pools_use_kept_generated_audio_only, test_missing_label_is_refused, test_experiment_section):
        test()
        print("PASS", test.__name__)
    print("All demo_page tests passed.")
