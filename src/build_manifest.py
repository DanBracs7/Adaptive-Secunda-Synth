import csv
import json
import random
from pathlib import Path

import torch


# ── Project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Point this at wherever the new dataset version is mounted/extracted.
SLAKH_TRAIN_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "slakh_final_1070_tensors" /
    "train"
)

SLAKH_TEST_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "slakh_final_1070_tensors" /
    "test"
)

SKYRIM_DIR = PROJECT_ROOT / "datasets" / "skyrim"
WITCHER_DIR = PROJECT_ROOT / "datasets" / "witcher3"

EXCLUSIONS_PATH = (
    PROJECT_ROOT / "configs" / "excluded_tensors.json"
)

OUTPUT_PATH = PROJECT_ROOT / "manifests" / "all_tracks.csv"

SPLIT_SEED = 42

# Only sources that need a random train/val split.
# Slakh's held-out "test" folder is assigned split="test" directly,
# bypassing this dict entirely.
VALIDATION_RATIOS = {
    "slakh": 0.12,
}

VALIDATION_FIXED_COUNTS = {
    "skyrim": 6,
    "witcher3": 6,
    "slakh": 35,
}


def resolve_validation_count(source_name, pool_size):
    if source_name in VALIDATION_FIXED_COUNTS:
        return VALIDATION_FIXED_COUNTS[source_name]

    ratio = VALIDATION_RATIOS[source_name]
    return max(1, round(pool_size * ratio))

def load_exclusions():
    if not EXCLUSIONS_PATH.exists():
        raise FileNotFoundError(
            f"Exclusion config not found: {EXCLUSIONS_PATH}"
        )

    with open(EXCLUSIONS_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    excluded_files = {}

    for item in config.get("excluded_files", []):
        relative_path = item["path"].replace("\\", "/")
        excluded_files[relative_path] = item["reason"]

    return excluded_files


def discover_tensor_files(source_name, directory):
    if not directory.exists():
        raise FileNotFoundError(
            f"{source_name} folder does not exist: {directory}"
        )

    tensor_files = sorted(directory.glob("*.pt"))

    if not tensor_files:
        raise FileNotFoundError(
            f"No .pt tensors found for {source_name}: {directory}"
        )

    return tensor_files


def get_relative_manifest_path(source_name, pt_path):
    return f"{source_name}/{pt_path.name}"


def read_tensor_metadata(pt_path):
    data = torch.load(pt_path, map_location="cpu")

    required_keys = {
        "track_id", "duration_sec", "tokens", "tension",
        "combat_score", "sample_rate", "frame_rate", "bandwidth_kbps",
    }

    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise KeyError(
            f"{pt_path.name} is missing keys: {sorted(missing_keys)}"
        )

    tokens = data["tokens"]
    tension = data["tension"]
    combat_score = data["combat_score"]

    if tokens.ndim != 2:
        raise ValueError(
            f"{pt_path.name}: expected tokens [n_q, T], "
            f"got {tuple(tokens.shape)}"
        )

    if tokens.shape[-1] != len(tension):
        raise ValueError(
            f"{pt_path.name}: tokens/tension mismatch: "
            f"{tokens.shape[-1]} vs {len(tension)}"
        )

    if tokens.shape[-1] != len(combat_score):
        raise ValueError(
            f"{pt_path.name}: tokens/combat mismatch: "
            f"{tokens.shape[-1]} vs {len(combat_score)}"
        )

    return {
        "track_id": str(data["track_id"]),
        "duration_sec": float(data["duration_sec"]),
        "num_codebooks": int(tokens.shape[0]),
        "num_frames": int(tokens.shape[-1]),
        "sample_rate": int(data["sample_rate"]),
        "frame_rate": int(data["frame_rate"]),
        "bandwidth_kbps": float(data["bandwidth_kbps"]),
    }


def stage_for_source(source_name):
    if source_name == "slakh":
        return "pretrain"

    if source_name in {"skyrim", "witcher3"}:
        return "finetune"

    raise ValueError(f"Unknown source: {source_name}")




def assign_splits(rows):
    rows_by_source = {}

    for row in rows:
        if row["split"] != "unassigned":
            continue

        rows_by_source.setdefault(row["source"], []).append(row)

    for source_name, source_rows in rows_by_source.items():
        validation_count = resolve_validation_count(
            source_name, len(source_rows)
        )

        if validation_count >= len(source_rows):
            raise ValueError(
                f"{source_name}: validation count "
                f"{validation_count} must be smaller than "
                f"number of usable tracks {len(source_rows)}."
            )

        rng = random.Random(SPLIT_SEED)
        shuffled_rows = source_rows.copy()
        rng.shuffle(shuffled_rows)

        validation_rows = shuffled_rows[:validation_count]
        validation_paths = {row["path"] for row in validation_rows}

        for row in source_rows:
            row["split"] = (
                "val" if row["path"] in validation_paths else "train"
            )

def build_rows():
    excluded_files = load_exclusions()

    # sources needing a random train/val split
    train_pool_sources = {
        "slakh": SLAKH_TRAIN_DIR,
        "skyrim": SKYRIM_DIR,
        "witcher3": WITCHER_DIR,
    }

    # sources whose tracks are ALWAYS held out as test, never split
    fixed_test_sources = {
        "slakh": SLAKH_TEST_DIR,
    }

    rows = []

    def make_row(source_name, pt_path, forced_split=None):
        manifest_path = get_relative_manifest_path(source_name, pt_path)
        metadata = read_tensor_metadata(pt_path)
        exclusion_reason = excluded_files.get(manifest_path)

        if exclusion_reason:
            split = "excluded"
        elif forced_split is not None:
            split = forced_split
        else:
            split = "unassigned"

        return {
            "path": manifest_path,
            "source": source_name,
            "track_id": metadata["track_id"],
            "duration_sec": metadata["duration_sec"],
            "num_codebooks": metadata["num_codebooks"],
            "num_frames": metadata["num_frames"],
            "sample_rate": metadata["sample_rate"],
            "frame_rate": metadata["frame_rate"],
            "bandwidth_kbps": metadata["bandwidth_kbps"],
            "split": split,
            "stage": (
                "excluded" if exclusion_reason
                else stage_for_source(source_name)
            ),
            "exclusion_reason": exclusion_reason or "",
        }

    for source_name, directory in train_pool_sources.items():
        for pt_path in discover_tensor_files(source_name, directory):
            rows.append(make_row(source_name, pt_path))

    for source_name, directory in fixed_test_sources.items():
        for pt_path in discover_tensor_files(source_name, directory):
            # Note: reuses the "slakh" source name so the dataset class
            # still resolves it via SOURCE_ROOTS["slakh"] at load time,
            # but the manifest path prefix distinguishes physical folder.
            rows.append(
                make_row(source_name, pt_path, forced_split="test")
            )

    unknown_exclusions = set(excluded_files) - {
        row["path"] for row in rows
    }

    if unknown_exclusions:
        raise ValueError(
            "Exclusion paths were not found in discovered tensors: "
            f"{sorted(unknown_exclusions)}"
        )

    assign_splits(rows)

    return rows


def validate_final_rows(rows):
    observed_counts = {}

    for row in rows:
        key = (row["source"], row["split"])
        observed_counts[key] = observed_counts.get(key, 0) + 1

    print("\nManifest counts:")
    for key in sorted(observed_counts):
        print(f"{key[0]:<9} {key[1]:<9}: {observed_counts[key]}")

    total_rows = len(rows)
    total_usable = sum(
        count for (_, split), count in observed_counts.items()
        if split != "excluded"
    )

    print(f"\nTotal rows: {total_rows}")
    print(f"Total usable (non-excluded): {total_usable}")

    # Sanity checks instead of hard-coded exact counts.
    for source_name in VALIDATION_FIXED_COUNTS:
        expected_val = VALIDATION_FIXED_COUNTS[source_name]
        observed_val = observed_counts.get((source_name, "val"), 0)
        if observed_val != expected_val:
            raise ValueError(
                f"{source_name}: expected {expected_val} val tracks, "
                f"got {observed_val}."
            )

    # Slakh's val count is now ratio-based; just print it, don't hard-assert an exact number.
    print(
        f"slakh val count ({VALIDATION_RATIOS['slakh']} ratio):",
        observed_counts.get(("slakh", "val"), 0),
    )
            

    for row in rows:
        if row["split"] == "excluded":
            continue

        if row["num_codebooks"] != 32:
            raise ValueError(
                f"{row['path']} has {row['num_codebooks']} "
                "codebooks; expected 32."
            )

        if row["frame_rate"] != 75:
            raise ValueError(
                f"{row['path']} has {row['frame_rate']} "
                "fps; expected 75."
            )

        if row["bandwidth_kbps"] != 24.0:
            raise ValueError(
                f"{row['path']} has {row['bandwidth_kbps']} "
                "kbps; expected 24.0."
            )


def write_manifest(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path", "source", "track_id", "duration_sec", "num_codebooks",
        "num_frames", "sample_rate", "frame_rate", "bandwidth_kbps",
        "split", "stage", "exclusion_reason",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved manifest: {OUTPUT_PATH}")
    print(f"Total rows: {len(rows)}")


def main():
    rows = build_rows()
    validate_final_rows(rows)
    write_manifest(rows)


if __name__ == "__main__":
    main()