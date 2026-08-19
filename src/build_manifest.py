import csv
import json
import random
from pathlib import Path

import torch


# ── Project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SLAKH_DIR = (
    PROJECT_ROOT /
    "datasets" /
    "slakh_final_388tensors" /
    "tensors_final" /
    "train"
)

SKYRIM_DIR = PROJECT_ROOT / "datasets" / "skyrim"
WITCHER_DIR = PROJECT_ROOT / "datasets" / "witcher3"

EXCLUSIONS_PATH = (
    PROJECT_ROOT /
    "configs" /
    "excluded_tensors.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT /
    "manifests" /
    "all_tracks.csv"
)

# ── Deterministic split configuration ─────────────────────────────────
SPLIT_SEED = 42

VALIDATION_COUNTS = {
    "slakh": 48,
    "skyrim": 4,
    "witcher3": 4,
}


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
        "track_id",
        "duration_sec",
        "tokens",
        "tension",
        "combat_score",
        "sample_rate",
        "frame_rate",
        "bandwidth_kbps",
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
        if row["split"] == "excluded":
            continue

        rows_by_source.setdefault(row["source"], []).append(row)

    for source_name, source_rows in rows_by_source.items():
        validation_count = VALIDATION_COUNTS[source_name]

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

        validation_paths = {
            row["path"]
            for row in validation_rows
        }

        for row in source_rows:
            row["split"] = (
                "val"
                if row["path"] in validation_paths
                else "train"
            )


def build_rows():
    excluded_files = load_exclusions()

    sources = {
        "slakh": SLAKH_DIR,
        "skyrim": SKYRIM_DIR,
        "witcher3": WITCHER_DIR,
    }

    rows = []

    for source_name, directory in sources.items():
        tensor_files = discover_tensor_files(
            source_name,
            directory,
        )

        for pt_path in tensor_files:
            manifest_path = get_relative_manifest_path(
                source_name,
                pt_path,
            )

            metadata = read_tensor_metadata(pt_path)

            exclusion_reason = excluded_files.get(manifest_path)

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
                "split": (
                    "excluded"
                    if exclusion_reason
                    else "unassigned"
                ),
                "stage": (
                    "excluded"
                    if exclusion_reason
                    else stage_for_source(source_name)
                ),
                "exclusion_reason": exclusion_reason or "",
            }

            rows.append(row)

    unknown_exclusions = set(excluded_files) - {
        row["path"]
        for row in rows
    }

    if unknown_exclusions:
        raise ValueError(
            "Exclusion paths were not found in discovered tensors: "
            f"{sorted(unknown_exclusions)}"
        )

    assign_splits(rows)

    return rows


def validate_final_rows(rows):
    expected_counts = {
        ("slakh", "train"): 340,
        ("slakh", "val"): 48,
        ("skyrim", "train"): 34,
        ("skyrim", "val"): 4,
        ("skyrim", "excluded"): 1,
        ("witcher3", "train"): 31,
        ("witcher3", "val"): 4,
    }

    observed_counts = {}

    for row in rows:
        key = (row["source"], row["split"])
        observed_counts[key] = observed_counts.get(key, 0) + 1

    print("\nManifest counts:")
    for key in sorted(observed_counts):
        print(f"{key[0]:<9} {key[1]:<9}: {observed_counts[key]}")

    if observed_counts != expected_counts:
        raise ValueError(
            "\nUnexpected manifest counts.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {observed_counts}"
        )

    for row in rows:
        if row["split"] == "excluded":
            continue

        if row["num_codebooks"] != 32:
            raise ValueError(
                f"{row['path']} has "
                f"{row['num_codebooks']} codebooks; expected 32."
            )

        if row["frame_rate"] != 75:
            raise ValueError(
                f"{row['path']} has "
                f"{row['frame_rate']} fps; expected 75."
            )

        if row["bandwidth_kbps"] != 24.0:
            raise ValueError(
                f"{row['path']} has "
                f"{row['bandwidth_kbps']} kbps; expected 24.0."
            )


def write_manifest(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "source",
        "track_id",
        "duration_sec",
        "num_codebooks",
        "num_frames",
        "sample_rate",
        "frame_rate",
        "bandwidth_kbps",
        "split",
        "stage",
        "exclusion_reason",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

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