import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from frame_codec import NUM_CODEBOOKS


class SecundaAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        source_roots,
        split="train",
        stage=None,
        segment_seconds=5.0,
        frame_rate=75,
        fixed_start=False,
        seed=42,
        samples_per_track=1,
    ):
        self.manifest_path = Path(manifest_path)

        self.source_roots = {
            key: Path(value)
            for key, value in source_roots.items()
        }

        self.segment_frames = int(
            round(segment_seconds * frame_rate)
        )

        if self.segment_frames < 2:
            raise ValueError(
                "segment_seconds must produce at least 2 frames."
            )

        if samples_per_track < 1:
            raise ValueError(
                "samples_per_track must be at least 1."
            )

        self.fixed_start = fixed_start
        self.seed = seed
        self.epoch = 0
        self.samples_per_track = samples_per_track

        with open(
            self.manifest_path,
            newline="",
            encoding="utf-8",
        ) as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row
            for row in all_rows
            if row["split"] == split
            and row["split"] != "excluded"
            and (
                stage is None
                or row["stage"] == stage
            )
        ]

        if not self.rows:
            raise ValueError(
                f"No rows found for split={split}, stage={stage}"
            )

        self._validate_rows()

    def _validate_rows(self):
        for row in self.rows:
            source = row["source"]

            if source not in self.source_roots:
                raise KeyError(
                    f"No root configured for source={source}"
                )

            filename = Path(row["path"]).name
            tensor_path = self.source_roots[source] / filename

            if not tensor_path.exists():
                raise FileNotFoundError(tensor_path)

            if int(row["num_codebooks"]) != 32:
                raise ValueError(
                    f"{tensor_path}: expected 32 codebooks"
                )

            if int(row["frame_rate"]) != 75:
                raise ValueError(
                    f"{tensor_path}: expected 75 fps"
                )

            if float(row["bandwidth_kbps"]) != 24.0:
                raise ValueError(
                    f"{tensor_path}: expected 24 kbps"
                )

            if int(row["num_frames"]) < self.segment_frames:
                raise ValueError(
                    f"{tensor_path}: has only "
                    f"{row['num_frames']} frames, but needs "
                    f"{self.segment_frames}"
                )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows) * self.samples_per_track

    def _path_for_row(self, row):
        source = row["source"]
        filename = Path(row["path"]).name
        return self.source_roots[source] / filename

    def _choose_start(self,row_index,crop_index,total_frames,):
        max_start = total_frames - self.segment_frames
    
        if max_start <= 0:
            return 0
    
        if self.fixed_start:
            # Validation:
            # same interior crop for this validation track every epoch.
            # It is deterministic, but not forced to begin at frame 0.
            generator = random.Random(
                self.seed
                + 99_999_937
                + row_index * 10_000
                + crop_index
            )
        else:
            # Training:
            # deterministic but changes with epoch.
            generator = random.Random(
                self.seed
                + self.epoch * 1_000_000
                + row_index * 10_000
                + crop_index
            )
    
        return generator.randint(0, max_start)



    def __getitem__(self, index):
        row_index = index // self.samples_per_track
        crop_index = index % self.samples_per_track

        row = self.rows[row_index]
        tensor_path = self._path_for_row(row)

        data = torch.load(
            tensor_path,
            map_location="cpu",
        )

        tokens = data["tokens"].long()
        tension = data["tension"].float()
        combat_score = data["combat_score"].float()

        if tokens.ndim != 2:
            raise ValueError(
                f"{tensor_path}: tokens must be [n_q, T]"
            )

        total_frames = tokens.shape[-1]

        if total_frames != len(tension):
            raise ValueError(
                f"{tensor_path}: tokens/tension mismatch"
            )

        if total_frames != len(combat_score):
            raise ValueError(
                f"{tensor_path}: tokens/combat mismatch"
            )

        start = self._choose_start(
            row_index=row_index,
            crop_index=crop_index,
            total_frames=total_frames,
        )

        end = start + self.segment_frames

        return {
            "tokens": tokens[:NUM_CODEBOOKS, start:end],
            "tension": tension[start:end],
            "combat_score": combat_score[start:end],
            "source": row["source"],
            "track_id": row["track_id"],
            "path": row["path"],
            "start_frame": start,
            "crop_index": crop_index,
        }