"""
Tests for src/training_loop.py, on a small model and a synthetic dataset
(CPU, a few seconds).

1. A run stopped in the middle of an epoch and resumed from last.pt ends
   with exactly the same weights and validation history as an
   uninterrupted run (same data order, same dropout, same optimizer state).
2. ResumableEpochSampler continues the same order after start_index and
   changes order between epochs.
3. The learning rate stored in the history follows lr_at_step.
4. final.pt can be loaded with training_utils.load_checkpoint_model.
5. filter_rows keeps only the requested manifest rows.

Run from the repository root:
    python tests/test_training_loop.py
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frame_codec import CODEBOOK_SIZE, NUM_CODEBOOKS, CodebookSerializer  # noqa: E402
from model import SecundaTransformer  # noqa: E402
from training_loop import ResumableEpochSampler, build_optimizer, filter_rows, train  # noqa: E402
from training_utils import load_checkpoint_model, lr_at_step  # noqa: E402

SEGMENT_FRAMES = 24
MODEL_CONFIG = {"embedding_dim": 32, "num_layers": 2, "num_heads": 4, "feedforward_dim": 64,
                "dropout": 0.1, "max_sequence_length": SEGMENT_FRAMES + NUM_CODEBOOKS - 1}


class SyntheticDataset(Dataset):
    """Same interface as SecundaAudioDataset: rows, set_epoch, crops that depend on the epoch."""

    def __init__(self, num_tracks=5, samples_per_track=4):
        generator = torch.Generator().manual_seed(0)
        self.tracks = [
            {"tokens": torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, 60), generator=generator),
             "tension": torch.rand(60, generator=generator),
             "combat_score": torch.rand(60, generator=generator)}
            for _ in range(num_tracks)
        ]
        self.rows = [{"path": f"slakh/train_{i:03d}.pt", "track_id": f"T{i}"} for i in range(num_tracks)]
        self.samples_per_track = samples_per_track
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, index):
        track = self.tracks[index // self.samples_per_track]
        start = (index * 7 + self.epoch * 3) % (60 - SEGMENT_FRAMES)
        end = start + SEGMENT_FRAMES
        return {"tokens": track["tokens"][:, start:end], "tension": track["tension"][start:end],
                "combat_score": track["combat_score"][start:end], "track_id": f"T{index}"}


def make_config(max_steps=14):
    return {
        **MODEL_CONFIG,
        "batch_size": 4, "num_workers": 0, "peak_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 3,
        "max_steps": max_steps, "codebook_loss_weights": [4.0, 3.2, 2.7, 2.2, 2.0, 2.0, 2.0, 2.0],
        "grad_clip_norm": 1.0, "seed": 42, "save_every_steps": 3, "log_every_steps": 1000,
    }


def fresh_run(seed=0):
    torch.manual_seed(seed)
    model = SecundaTransformer(**MODEL_CONFIG)
    optimizer = build_optimizer(model, peak_lr=1e-3, weight_decay=0.01)
    return model, optimizer


def run(output_dir, resume_path=None, stop_at_step=None, init_seed=0):
    model, optimizer = fresh_run(init_seed)
    val_loader = DataLoader(SyntheticDataset(num_tracks=2, samples_per_track=1), batch_size=2)
    state = train(model, optimizer, SyntheticDataset(), val_loader, CodebookSerializer(), torch.device("cpu"),
                  make_config(), output_dir, resume_path=resume_path, stop_at_step=stop_at_step,
                  log=lambda message: None)
    return model, state


def test_resume_gives_identical_weights():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        model_a, state_a = run(directory / "a")
        assert state_a["finished"] and state_a["global_step"] == 14

        # 20 samples / batch 4 = 5 steps per epoch: step 7 is inside epoch 2.
        _, partial = run(directory / "b", stop_at_step=7)
        assert not partial["finished"] and partial["global_step"] == 7

        torch.manual_seed(999)  # the resumed run must restore its own RNG state
        model_b, state_b = run(directory / "b", resume_path=directory / "b" / "last.pt", init_seed=123)

        assert state_b["finished"] and state_b["global_step"] == 14
        for name, tensor in model_a.state_dict().items():
            assert torch.equal(tensor, model_b.state_dict()[name]), name
        assert state_a["history"]["val_loss"] == state_b["history"]["val_loss"]


def test_sampler_resumes_the_same_order():
    sampler = ResumableEpochSampler(20, seed=42)
    sampler.set_epoch(3)
    full_order = list(iter(sampler))
    sampler.set_epoch(3, start_index=8)
    assert list(iter(sampler)) == full_order[8:]
    sampler.set_epoch(4)
    assert list(iter(sampler)) != full_order
    assert sorted(full_order) == list(range(20))


def test_learning_rate_follows_schedule():
    with tempfile.TemporaryDirectory() as directory:
        _, state = run(Path(directory))
        config = make_config()
        for step, learning_rate in zip(state["history"]["global_step"], state["history"]["learning_rate"]):
            expected = lr_at_step(step, config["peak_lr"], config["min_lr"], config["warmup_steps"], config["max_steps"])
            assert abs(learning_rate - expected) < 1e-12


def test_final_checkpoint_loads_for_evaluation():
    with tempfile.TemporaryDirectory() as directory:
        model, _ = run(Path(directory))
        loaded, checkpoint = load_checkpoint_model(Path(directory) / "final.pt", torch.device("cpu"))
        assert checkpoint["finished"]
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, loaded.state_dict()[name]), name


def test_filter_rows():
    dataset = filter_rows(SyntheticDataset(), lambda row: int(row["path"][-6:-3]) <= 2)
    assert [row["track_id"] for row in dataset.rows] == ["T0", "T1", "T2"]


if __name__ == "__main__":
    tests = [
        test_resume_gives_identical_weights,
        test_sampler_resumes_the_same_order,
        test_learning_rate_follows_schedule,
        test_final_checkpoint_loads_for_evaluation,
        test_filter_rows,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All training_loop tests passed.")
