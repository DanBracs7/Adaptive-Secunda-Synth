import csv
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Tensor folders come from src/paths.py, the same configuration used by
# the notebooks (local defaults: datasets/slakh_final_1395_tensors/...).
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import get_path, source_roots_for_split  # noqa: E402


MANIFEST_PATH = get_path("manifest")


def resolve_tensor_path(relative_path, source, split):
    # Slakh test tensors keep source="slakh" but live in their own folder,
    # so the root depends on the split as well as on the source.
    filename = Path(relative_path).name
    return source_roots_for_split(split)[source] / filename


def check_split_disjointness(usable_rows):
    """Every track_id must belong to exactly one split."""
    splits_per_track = {}

    for row in usable_rows:
        key = (row["source"], row["track_id"])
        splits_per_track.setdefault(key, set()).add(row["split"])

    shared = {
        key: splits
        for key, splits in splits_per_track.items()
        if len(splits) > 1
    }

    if shared:
        raise ValueError(
            f"{len(shared)} track_ids appear in more than one split, "
            f"e.g. {list(shared.items())[:5]}"
        )

    print("PASS: train/val/test are disjoint at track_id level.")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    print(f"Manifest rows: {len(rows)}")

    usable_rows = [
        row for row in rows
        if row["split"] != "excluded"
    ]

    excluded_rows = [
        row for row in rows
        if row["split"] == "excluded"
    ]

    print(f"Usable rows: {len(usable_rows)}")
    print(f"Excluded rows: {len(excluded_rows)}")

    for row in excluded_rows:
        print(
            f"Excluded: {row['path']} | "
            f"reason={row['exclusion_reason']}"
        )

    counts = {}
    for row in usable_rows:
        key = (row["source"], row["split"])
        counts[key] = counts.get(key, 0) + 1

    print("\nUsable rows per source and split:")
    for (source, split), count in sorted(counts.items()):
        print(f"  {source:<9} {split:<6}: {count}")

    check_split_disjointness(usable_rows)

    for row in usable_rows:
        source = row["source"]
        tensor_path = resolve_tensor_path(
            row["path"],
            source,
            row["split"],
        )

        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Missing tensor for manifest row: "
                f"{row['path']}\nExpected: {tensor_path}"
            )

        data = torch.load(tensor_path, map_location="cpu")

        tokens = data["tokens"]
        tension = data["tension"]
        combat = data["combat_score"]

        if tokens.shape[-1] != len(tension):
            raise ValueError(
                f"{tensor_path}: token/tension mismatch"
            )

        if tokens.shape[-1] != len(combat):
            raise ValueError(
                f"{tensor_path}: token/combat mismatch"
            )

        if tokens.shape[0] != 32:
            raise ValueError(
                f"{tensor_path}: expected 32 codebooks, "
                f"got {tokens.shape[0]}"
            )

    print("All usable manifest paths and tensor alignments passed.")


if __name__ == "__main__":
    main()
