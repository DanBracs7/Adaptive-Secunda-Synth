import hashlib
from collections import defaultdict
from pathlib import Path

import torch


# Change these paths if required.
DATASETS = {
    "skyrim": Path("datasets/skyrim"),
    "witcher3": Path("datasets/witcher3"),
}


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.numpy().tobytes()
    ).hexdigest()


def file_signature(pt_path):
    data = torch.load(pt_path, map_location="cpu")

    required_keys = {"tokens", "tension", "combat_score"}

    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(
            f"{pt_path.name} is missing keys: {sorted(missing)}"
        )

    return (
        tensor_hash(data["tokens"]),
        tensor_hash(data["tension"]),
        tensor_hash(data["combat_score"]),
    )


def main():
    for dataset_name, data_dir in DATASETS.items():
        print(f"\n{'=' * 90}")
        print(f"Checking exact tensor duplicates: {dataset_name}")
        print(f"Folder: {data_dir}")
        print(f"{'=' * 90}")

        if not data_dir.exists():
            print("Folder does not exist; skipping.")
            continue

        groups = defaultdict(list)

        for pt_path in sorted(data_dir.glob("*.pt")):
            signature = file_signature(pt_path)
            groups[signature].append(pt_path)

        duplicate_groups = [
            files
            for files in groups.values()
            if len(files) > 1
        ]

        print(f"Tensor files checked: {sum(len(x) for x in groups.values())}")

        if not duplicate_groups:
            print("No exact duplicates found.")
            continue

        print(f"Exact duplicate groups found: {len(duplicate_groups)}")

        for index, files in enumerate(duplicate_groups, start=1):
            print(f"\nDuplicate group {index}:")

            for pt_path in files:
                data = torch.load(pt_path, map_location="cpu")

                print(
                    f"  {pt_path.name} | "
                    f"track_id={data.get('track_id', 'UNKNOWN')} | "
                    f"duration={float(data.get('duration_sec', 0.0)):.2f}s"
                )


if __name__ == "__main__":
    main()