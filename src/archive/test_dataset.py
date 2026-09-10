from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_dataset import SecundaAudioDataset


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


def test_dataset(split, stage):
    dataset = SecundaAudioDataset(
        manifest_path=MANIFEST_PATH,
        source_roots=SOURCE_ROOTS,
        split=split,
        stage=stage,
        segment_seconds=5.0,
        frame_rate=75,
        fixed_start=False,
    )

    print(
        f"\nDataset split={split}, stage={stage}"
    )
    print(f"Number of tracks: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("tokens shape:", tuple(batch["tokens"].shape))
    print("tension shape:", tuple(batch["tension"].shape))
    print(
        "combat score shape:",
        tuple(batch["combat_score"].shape),
    )
    print("sources:", batch["source"])
    print("track IDs:", batch["track_id"])
    print("start frames:", batch["start_frame"])

    assert batch["tokens"].shape == (2, 32, 375)
    assert batch["tension"].shape == (2, 375)
    assert batch["combat_score"].shape == (2, 375)


def main():
    test_dataset("train", "pretrain")
    test_dataset("train", "finetune")
    test_dataset("val", "pretrain")
    test_dataset("val", "finetune")

    print("\nAll dataset loader tests passed.")


if __name__ == "__main__":
    main()