"""
Tests for src/demo_review.py (synthetic bank, no audio needed).

1. read_bank merges the clips of every folder by passage (clip id without
   the take or loop suffix), keeps the temperature of each folder, and
   orders chill before fight, then by title and prompt start.
2. The page links both WAV files of every clip by relative path, has one
   entry per (folder, clip) in the data used for the export, and marks a
   clip missing from one folder.
3. A folder with loops.csv shows its loops (loop file + first take with
   prompt) next to the single takes of the same passage.
4. A bank folder without clips.csv or loops.csv is refused.

Run from the repository root:
    python tests/test_demo_review.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import demo_review  # noqa: E402

FIELDS = ["setting", "clip", "label", "title", "prompt_start", "temperature", "rms_dbfs", "clipped_fraction"]


def write_bank(root):
    rows = {
        "base": [
            ("fight_one_they_fear_0m10_t1", "fight", "One They Fear", "0:10"),
            ("chill_secunda_0m46_t1", "chill", "Secunda", "0:46"),
            ("chill_secunda_0m10_t1", "chill", "Secunda", "0:10"),
        ],
        "fine_t07": [
            ("chill_secunda_0m10_t1", "chill", "Secunda", "0:10"),
            ("fight_one_they_fear_0m10_t1", "fight", "One They Fear", "0:10"),
        ],
    }
    temperatures = {"base": "1.0", "fine_t07": "1 0.7 0.7 0.7 0.7 0.7 0.7 0.7"}
    for setting, clips in rows.items():
        folder = Path(root) / setting
        folder.mkdir(parents=True)
        with open(folder / "clips.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for clip, label, title, start in clips:
                writer.writerow({"setting": setting, "clip": clip, "label": label, "title": title,
                                 "prompt_start": start, "temperature": temperatures[setting],
                                 "rms_dbfs": "-22.0", "clipped_fraction": "0.0"})


def test_read_bank():
    with tempfile.TemporaryDirectory() as root:
        write_bank(root)
        settings, passages = demo_review.read_bank(root)
    assert settings == {"base": "1.0", "fine_t07": "1 0.7 0.7 0.7 0.7 0.7 0.7 0.7"}
    assert [p["id"] for p in passages] == ["chill_secunda_0m10", "chill_secunda_0m46", "fight_one_they_fear_0m10"]
    assert set(passages[0]["per_setting"]) == {"base", "fine_t07"}
    assert set(passages[1]["per_setting"]) == {"base"}
    assert passages[0]["per_setting"]["base"][0]["clip"] == "chill_secunda_0m10_t1"


def test_page_links_every_clip():
    with tempfile.TemporaryDirectory() as root:
        write_bank(root)
        page = demo_review.build_page(*demo_review.read_bank(root))
    for setting, clip in (("base", "chill_secunda_0m10_t1"), ("fine_t07", "fight_one_they_fear_0m10_t1")):
        assert f'data-src="{setting}/{clip}_generated.wav"' in page
        assert f'data-src="{setting}/{clip}_with_prompt.wav"' in page
    clips = json.loads(page.split("const CLIPS = ", 1)[1].split(";\n", 1)[0])
    assert len(clips) == 5                                  # 3 base + 2 fine_t07
    assert len({c["key"] for c in clips}) == 5
    assert "fine_t07: clip mancante" in page                # chill_secunda_0m46_t1 only in base


def test_loops_folder():
    with tempfile.TemporaryDirectory() as root:
        write_bank(root)
        folder = Path(root) / "base_3s"
        folder.mkdir()
        fields = ["setting", "clip", "label", "title", "prompt_start", "temperature", "generated_file",
                  "with_prompt_file", "takes", "loop_seconds", "rms_dbfs", "clipped_fraction"]
        with open(folder / "loops.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"setting": "base", "clip": "fight_one_they_fear_0m10_loop", "label": "fight",
                             "title": "One They Fear", "prompt_start": "0:10", "temperature": "1.0",
                             "generated_file": "fight_one_they_fear_0m10_loop.wav",
                             "with_prompt_file": "fight_one_they_fear_0m10_t1_with_prompt.wav",
                             "takes": "4", "loop_seconds": "11.25", "rms_dbfs": "-20.0", "clipped_fraction": "0.0"})
        settings, passages = demo_review.read_bank(root)
        page = demo_review.build_page(settings, passages)
    assert list(settings) == ["base", "base_3s", "fine_t07"]
    fight = [p for p in passages if p["id"] == "fight_one_they_fear_0m10"][0]
    assert set(fight["per_setting"]) == {"base", "base_3s", "fine_t07"}
    assert 'data-src="base_3s/fight_one_they_fear_0m10_loop.wav"' in page
    assert 'data-src="base_3s/fight_one_they_fear_0m10_t1_with_prompt.wav"' in page
    assert "loop (4 take, 11.25 s)" in page


def test_missing_bank_is_refused():
    with tempfile.TemporaryDirectory() as root:
        try:
            demo_review.read_bank(root)
        except FileNotFoundError:
            return
    raise AssertionError("FileNotFoundError not raised")


if __name__ == "__main__":
    for test in (test_read_bank, test_page_links_every_clip, test_loops_folder, test_missing_bank_is_refused):
        test()
        print("PASS", test.__name__)
    print("All demo_review tests passed.")
