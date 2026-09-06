import csv
import json
import random
from pathlib import Path

import torch


# ── Project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# New combined dataset (1,395 Slakh train tensors)
SLAKH_TRAIN_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "slakh_final_1395_tensors" /
    "train"
)

SLAKH_TEST_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "slakh_final_1395_tensors" /
    "test"
)

SKYRIM_DIR = PROJECT_ROOT / "datasets" / "skyrim"
WITCHER_DIR = PROJECT_ROOT / "datasets" / "witcher3"

EXCLUSIONS_PATH = (
    PROJECT_ROOT / "configs" / "excluded_tensors.json"
)

OUTPUT_PATH = PROJECT_ROOT / "manifests" / "all_tracks.csv"

# Previous manifest that already has the correct 1,035/35 Slakh split
PREVIOUS_MANIFEST_PATH = (
    PROJECT_ROOT /
    "manifests" /
    "all_tracks_800.csv"
)

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
}

SLAKH_NEW_VALIDATION_COUNT = 35

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


def load_previous_slakh_splits():
    """
    Load split assignments (train/val) for Slakh tracks from the previous manifest.
    Returns a dict: {path: split} for Slakh rows with split in {train, val}.
    Returns None if the previous manifest does not exist.
    """
    if not PREVIOUS_MANIFEST_PATH.exists():
        return None

    previous_rows = []
    with open(PREVIOUS_MANIFEST_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["source"] != "slakh":
                continue
            if row["split"] in ("train", "val"):
                previous_rows.append(row)

    return {row["path"]: row["split"] for row in previous_rows}


def assign_splits(rows):
    """
    Assign train/val splits to unassigned rows.

    Old Slakh assignments from all_tracks_800.csv are already frozen.
    New Slakh tracks are split separately so each added block contributes
    35 validation tracks:
      - train_800.pt to train_1069.pt
      - train_1070.pt to train_1394.pt
    """
    rows_by_source = {}

    for row in rows:
        if row["split"] != "unassigned":
            continue

        rows_by_source.setdefault(row["source"], []).append(row)

    for source_name, source_rows in rows_by_source.items():
        if source_name == "slakh":
            slakh_rows_800_to_1069 = []
            slakh_rows_1070_to_1394 = []

            for row in source_rows:
                file_name = Path(row["path"]).name
                track_number = int(file_name.replace("train_", "").replace(".pt", ""))

                if 800 <= track_number <= 1069:
                    slakh_rows_800_to_1069.append(row)
                elif 1070 <= track_number <= 1394:
                    slakh_rows_1070_to_1394.append(row)
                else:
                    raise ValueError(
                        f"Unexpected unassigned Slakh track: {row['path']}"
                    )

            slakh_groups = {
                "800_to_1069": slakh_rows_800_to_1069,
                "1070_to_1394": slakh_rows_1070_to_1394,
            }

            for group_name, group_rows in slakh_groups.items():
                if len(group_rows) < SLAKH_NEW_VALIDATION_COUNT:
                    raise ValueError(
                        f"Slakh group {group_name} has only {len(group_rows)} tracks; "
                        f"cannot reserve {SLAKH_NEW_VALIDATION_COUNT} for validation."
                    )

                rng = random.Random(f"{SPLIT_SEED}_{group_name}")
                shuffled_rows = group_rows.copy()
                rng.shuffle(shuffled_rows)

                validation_paths = {
                    row["path"]
                    for row in shuffled_rows[:SLAKH_NEW_VALIDATION_COUNT]
                }

                for row in group_rows:
                    row["split"] = (
                        "val" if row["path"] in validation_paths else "train"
                    )

            continue

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

        validation_paths = {
            row["path"]
            for row in shuffled_rows[:validation_count]
        }

        for row in source_rows:
            row["split"] = (
                "val" if row["path"] in validation_paths else "train"
            )

def build_rows():
    excluded_files = load_exclusions()

    # Load previous Slakh splits to freeze them
    previous_slakh_splits = load_previous_slakh_splits()

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

        row = {
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

        # Freeze old Slakh train/val splits from the previous manifest
        if (
            source_name == "slakh" and
            previous_slakh_splits is not None and
            manifest_path in previous_slakh_splits
        ):
            row["split"] = previous_slakh_splits[manifest_path]

        return row

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
    # Skyrim and Witcher validation counts are fixed totals.
    for source_name in ("skyrim", "witcher3"):
        expected_val = VALIDATION_FIXED_COUNTS[source_name]
        observed_val = observed_counts.get((source_name, "val"), 0)

        if observed_val != expected_val:
            raise ValueError(
                f"{source_name}: expected {expected_val} val tracks, "
                f"got {observed_val}."
            )

        # Slakh validation tracks consist of:
    # - 96 frozen validation tracks from all_tracks_800.csv
    # - 35 new validation tracks from train_800.pt to train_1069.pt
    # - 35 new validation tracks from train_1070.pt to train_1394.pt
    expected_slakh_val = 166
    observed_slakh_val = observed_counts.get(("slakh", "val"), 0)

    if observed_slakh_val != expected_slakh_val:
        raise ValueError(
            f"slakh: expected {expected_slakh_val} total validation tracks, "
            f"got {observed_slakh_val}."
        )

    print(
        "slakh validation tracks:",
        observed_slakh_val,
        "(96 frozen from all_tracks_800.csv + "
        "35 from train_800.pt–train_1069.pt + "
        "35 from train_1070.pt–train_1394.pt)",
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