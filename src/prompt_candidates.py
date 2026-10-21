"""
Prompt selection for the demo (Phase 4).

The generated music takes its character (instruments, atmosphere) from
the prompt, while tension and combat score set its intensity (notebooks
05 and 06). The demo therefore uses prompts that match the game state:
calm passages for "chill" and battle passages for "fight".

  ost    The OST passages were chosen by listening (configs/demo_prompts.csv).
         For each passage this command reports the stored tension and
         combat score, i.e. how the training conditions describe the
         human choice, and decodes it with all 32 codebooks so that the
         cut points can be checked exactly as the model sees them.
  slakh  The Slakh tracks are too many to listen to. This command ranks
         10-second windows by their stored conditions and decodes the
         calmest and the most intense candidates (one window per track),
         so that only a short list has to be auditioned.

Run from the repository root (CPU is enough; EnCodec downloads its
weights on first use):
    python src/prompt_candidates.py ost
    python src/prompt_candidates.py slakh
WAV files go to outputs/prompt_candidates/ (not versioned); the tables
go to results/prompt_selection/.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from paths import get_path, source_roots_for_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_prompts.csv"
AUDIO_OUT = PROJECT_ROOT / "outputs" / "prompt_candidates"
TABLE_OUT = PROJECT_ROOT / "results" / "prompt_selection"

FRAME_RATE = 75
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
SKIP_EDGE_SECONDS = 10.0       # ignore intros and endings
CANDIDATES_PER_LABEL = 15

# Window filters on the stored conditions (scale [0, 1]).
CHILL_TENSION_RANGE = (0.15, 0.50)   # calm but not near-silent
FIGHT_MIN_TENSION = 0.50
MAX_SILENT_FRACTION = 0.10           # frames with tension < 0.02


def parse_time(text):
    """'m:ss' -> seconds."""
    minutes, seconds = str(text).strip().split(":")
    return int(minutes) * 60 + float(seconds)


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def window_means(values, window, hop, skip):
    """Mean of `values` over windows [start, start + window) for start = skip, skip + hop, ..."""
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    starts = np.arange(skip, len(values) - window - skip + 1, hop)
    means = (cumulative[starts + window] - cumulative[starts]) / window
    return starts, means


def rank_windows(tracks, per_label=CANDIDATES_PER_LABEL):
    """
    tracks: list of dicts with keys file, track_id, split, tension, combat_score.
    Returns {"chill": [...], "fight": [...]}, at most one window per track,
    best first. A window entry has file, track_id, split, start_frame,
    tension_mean, combat_mean, score.
    """
    window = int(WINDOW_SECONDS * FRAME_RATE)
    hop = int(HOP_SECONDS * FRAME_RATE)
    skip = int(SKIP_EDGE_SECONDS * FRAME_RATE)
    best = {"chill": {}, "fight": {}}

    for track in tracks:
        tension = np.asarray(track["tension"], dtype=np.float64)
        combat = np.asarray(track["combat_score"], dtype=np.float64)
        starts, tension_means = window_means(tension, window, hop, skip)
        _, combat_means = window_means(combat, window, hop, skip)
        _, silent = window_means((tension < 0.02).astype(np.float64), window, hop, skip)

        for start, t_mean, c_mean, silent_fraction in zip(starts, tension_means, combat_means, silent):
            if silent_fraction > MAX_SILENT_FRACTION:
                continue
            candidates = []
            if CHILL_TENSION_RANGE[0] <= t_mean <= CHILL_TENSION_RANGE[1]:
                candidates.append(("chill", (1.0 - c_mean) - 0.5 * t_mean))
            if t_mean >= FIGHT_MIN_TENSION:
                candidates.append(("fight", c_mean + 0.5 * t_mean))
            for label, score in candidates:
                current = best[label].get(track["file"])
                if current is None or score > current["score"]:
                    best[label][track["file"]] = {
                        "file": track["file"], "track_id": track["track_id"], "split": track["split"],
                        "start_frame": int(start), "tension_mean": float(t_mean),
                        "combat_mean": float(c_mean), "score": float(score),
                    }

    return {label: sorted(entries.values(), key=lambda e: -e["score"])[:per_label]
            for label, entries in best.items()}


def decode_segment(codec, tokens, start_frame, end_frame):
    """Decode a segment with all 32 codebooks (near the original audio quality)."""
    from generation import decode_codes

    codes = tokens[:, start_frame:end_frame].long().unsqueeze(0)
    return decode_codes(codec, codes)[0]


def make_codec():
    from generation import load_codec

    codec = load_codec(torch.device("cpu"))
    codec.set_target_bandwidth(24.0)   # 32 codebooks
    return codec


def write_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_ost():
    from generation import save_wav

    with open(CONFIG_PATH, newline="", encoding="utf-8") as file:
        passages = list(csv.DictReader(file))

    codec = make_codec()
    out_dir = AUDIO_OUT / "ost"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for passage in passages:
        data = load_tensor(get_path(passage["source"]) / passage["tensor_file"])
        start = int(round(parse_time(passage["start"]) * FRAME_RATE))
        end = min(int(round(parse_time(passage["end"]) * FRAME_RATE)), data["tokens"].shape[-1])
        tension = data["tension"][start:end].float().numpy()
        combat = data["combat_score"][start:end].float().numpy()

        name = f"{passage['label']}_{Path(passage['tensor_file']).stem}_{passage['start'].replace(':', 'm')}.wav"
        save_wav(out_dir / name, decode_segment(codec, data["tokens"], start, end))

        rows.append({**passage, "start_frame": start, "end_frame": end,
                     "tension_mean": round(float(tension.mean()), 4),
                     "combat_mean": round(float(combat.mean()), 4), "wav": name})
        print(f"{passage['label']:6s} {passage['title'][:28]:28s} {passage['start']:>5s}-{passage['end']:<5s} "
              f"tension {tension.mean():.2f}  combat {combat.mean():.2f}")

    write_table(rows, TABLE_OUT / "ost_passages.csv")
    for label in ("chill", "fight"):
        selected = [r for r in rows if r["label"] == label]
        print(f"{label}: {len(selected)} passages, mean stored tension "
              f"{np.mean([r['tension_mean'] for r in selected]):.2f}, combat "
              f"{np.mean([r['combat_mean'] for r in selected]):.2f}")
    print(f"WAV -> {out_dir}\nTable -> {TABLE_OUT / 'ost_passages.csv'}")


def command_slakh():
    from generation import save_wav

    with open(get_path("manifest"), newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file)
                if r["source"] == "slakh" and r["split"] in ("train", "val", "test")]

    tracks = []
    for index, row in enumerate(rows, start=1):
        path = source_roots_for_split(row["split"])["slakh"] / Path(row["path"]).name
        data = load_tensor(path)
        tracks.append({"file": Path(row["path"]).name, "track_id": row["track_id"], "split": row["split"],
                       "path": path, "tension": data["tension"].float().numpy(),
                       "combat_score": data["combat_score"].float().numpy()})
        if index % 200 == 0:
            print(f"read {index}/{len(rows)} tensors")

    ranked = rank_windows(tracks)
    codec = make_codec()
    window = int(WINDOW_SECONDS * FRAME_RATE)
    table = []

    for label, entries in ranked.items():
        out_dir = AUDIO_OUT / "slakh" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, entry in enumerate(entries, start=1):
            path = next(t["path"] for t in tracks if t["file"] == entry["file"])
            tokens = load_tensor(path)["tokens"]
            start_seconds = entry["start_frame"] / FRAME_RATE
            name = f"{rank:02d}_{Path(entry['file']).stem}_{int(start_seconds)}s.wav"
            save_wav(out_dir / name, decode_segment(codec, tokens, entry["start_frame"], entry["start_frame"] + window))
            table.append({"label": label, "rank": rank, **{k: entry[k] for k in ("file", "track_id", "split")},
                          "start": f"{int(start_seconds // 60)}:{int(start_seconds % 60):02d}",
                          "tension_mean": round(entry["tension_mean"], 4),
                          "combat_mean": round(entry["combat_mean"], 4), "wav": name, "keep": ""})
        print(f"{label}: {len(entries)} candidates -> {out_dir}")

    write_table(table, TABLE_OUT / "slakh_candidates.csv")
    print(f"Table -> {TABLE_OUT / 'slakh_candidates.csv'} (mark the ones to keep in the 'keep' column)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("ost", "slakh"))
    args = parser.parse_args(argv)
    command_ost() if args.command == "ost" else command_slakh()


if __name__ == "__main__":
    sys.exit(main())
