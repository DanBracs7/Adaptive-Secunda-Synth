import csv
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "all_tracks.csv"

SOURCE_ROOTS = {
    "slakh": (
        PROJECT_ROOT
        / "datasets"
        / "slakh_final_388tensors"
        / "tensors_final"
        / "train"
    ),
    "skyrim": PROJECT_ROOT / "datasets" / "skyrim",
    "witcher3": PROJECT_ROOT / "datasets" / "witcher3",
}


def resolve_tensor_path(relative_path, source):
    filename = Path(relative_path).name
    return SOURCE_ROOTS[source] / filename


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    rows = []

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

    for row in usable_rows:
        source = row["source"]
        tensor_path = resolve_tensor_path(
            row["path"],
            source,
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