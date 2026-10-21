"""
Tests for src/prompt_candidates.py (no tensors or codec needed).

1. parse_time converts 'm:ss'.
2. window_means equals a direct mean over each window.
3. rank_windows keeps at most one window per track, respects the
   tension filters and ranks calm windows as chill and intense ones as
   fight.
4. Every passage in configs/demo_prompts.csv names an OST tensor of the
   manifest and lies inside the track.

Run from the repository root:
    python tests/test_prompt_candidates.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import prompt_candidates as pc  # noqa: E402


def test_parse_time():
    assert pc.parse_time("0:29") == 29
    assert pc.parse_time("3:59") == 239
    assert pc.parse_time("1:05") == 65


def test_window_means():
    values = np.random.default_rng(0).random(500)
    starts, means = pc.window_means(values, window=50, hop=20, skip=30)
    for start, mean in zip(starts, means):
        assert abs(mean - values[start:start + 50].mean()) < 1e-12
    assert starts[0] == 30 and starts[-1] + 50 + 30 <= 500


def make_track(name, tension_level, combat_level, frames=3000):
    return {"file": name, "track_id": name, "split": "train",
            "tension": np.full(frames, tension_level), "combat_score": np.full(frames, combat_level)}


def test_rank_windows():
    tracks = [
        make_track("calm.pt", 0.3, 0.1),
        make_track("battle.pt", 0.8, 0.9),
        make_track("silent.pt", 0.0, 0.1),   # filtered: near-silent
        make_track("mid.pt", 0.6, 0.4),
    ]
    ranked = pc.rank_windows(tracks)
    chill_files = [e["file"] for e in ranked["chill"]]
    fight_files = [e["file"] for e in ranked["fight"]]
    assert chill_files == ["calm.pt"]
    assert fight_files[0] == "battle.pt" and "mid.pt" in fight_files
    assert "silent.pt" not in chill_files + fight_files
    assert len(fight_files) == len(set(fight_files))


def test_config_passages_are_valid():
    with open(PROJECT_ROOT / "manifests" / "all_tracks.csv", newline="", encoding="utf-8") as file:
        frames = {Path(r["path"]).name: int(r["num_frames"]) for r in csv.DictReader(file)}
    with open(pc.CONFIG_PATH, newline="", encoding="utf-8") as file:
        passages = list(csv.DictReader(file))
    assert passages
    for passage in passages:
        assert passage["label"] in ("chill", "fight"), passage
        assert passage["tensor_file"] in frames, passage
        start, end = pc.parse_time(passage["start"]), pc.parse_time(passage["end"])
        assert start < end <= frames[passage["tensor_file"]] / pc.FRAME_RATE, passage


if __name__ == "__main__":
    for test in (test_parse_time, test_window_means, test_rank_windows, test_config_passages_are_valid):
        test()
        print("PASS", test.__name__)
    print("All prompt_candidates tests passed.")
