"""
Regression test for src/training_utils.py.

The REFERENCE_* functions below are copied verbatim from the notebooks of
the first submission (they read global variables, exactly as there). The
test checks that the refactored functions in src/training_utils.py give
identical results, so moving the code did not change any number.

Run from the repository root:
    python tests/test_training_utils.py
or
    python -m pytest tests/
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frame_codec import (  # noqa: E402
    NUM_CODEBOOKS,
    PAD_TOKEN_ID,
    CodebookSerializer,
    prepare_next_frame_batch,
)
from model import SecundaTransformer  # noqa: E402
import training_utils  # noqa: E402


# ── Globals read by the reference (notebook) functions ──
device = torch.device("cpu")
serializer = CodebookSerializer()
CODEBOOK_LOSS_WEIGHTS = torch.tensor(
    [4.0, 3.2, 2.7, 2.2, 2.0, 2.0, 2.0, 2.0],
    dtype=torch.float32,
    device=device,
)
PEAK_LR = 2.0e-4
MIN_LR = 3.0e-5
WARMUP_STEPS = 4_000
MAX_STEPS = 35_220


# ── Copied from slakh-baseline-training-8kb.ipynb ──
def REFERENCE_lr_at_step(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * step / max(1, WARMUP_STEPS)
    progress = min(max(step - WARMUP_STEPS, 0), MAX_STEPS - WARMUP_STEPS)
    decay_ratio = progress / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + (PEAK_LR - MIN_LR) * cosine


def REFERENCE_prepare_batch_for_model(batch):
    prepared = prepare_next_frame_batch(batch=batch, serializer=serializer)
    input_tension = prepared["target_tension"][:, 0, :]
    input_combat = prepared["target_combat"][:, 0, :]
    return {
        "input_codes": prepared["input_codes"].to(device, non_blocking=True),
        "target_codes": prepared["target_codes"].to(device, non_blocking=True),
        "input_tension": input_tension.to(device, non_blocking=True),
        "input_combat": input_combat.to(device, non_blocking=True),
        "target_tension": prepared["target_tension"].to(device, non_blocking=True),
        "target_combat": prepared["target_combat"].to(device, non_blocking=True),
    }


def REFERENCE_multi_codebook_metrics(logits, target_codes):
    per_codebook_loss = []
    per_codebook_accuracy = []
    codebook_losses = []
    for codebook_index in range(NUM_CODEBOOKS):
        codebook_logits = logits[:, codebook_index, :, :]
        codebook_targets = target_codes[:, codebook_index, :]
        valid_mask = codebook_targets != PAD_TOKEN_ID
        if valid_mask.sum() == 0:
            nan_tensor = torch.tensor(float("nan"), device=logits.device)
            per_codebook_loss.append(nan_tensor)
            per_codebook_accuracy.append(nan_tensor)
            codebook_losses.append(nan_tensor)
            continue
        codebook_loss = F.cross_entropy(
            codebook_logits[valid_mask], codebook_targets[valid_mask]
        )
        predictions = codebook_logits.argmax(dim=-1)
        codebook_accuracy = (
            predictions[valid_mask] == codebook_targets[valid_mask]
        ).float().mean()
        per_codebook_loss.append(codebook_loss)
        per_codebook_accuracy.append(codebook_accuracy)
        codebook_losses.append(codebook_loss)
    per_codebook_loss = torch.stack(per_codebook_loss)
    per_codebook_accuracy = torch.stack(per_codebook_accuracy)
    codebook_losses = torch.stack(codebook_losses)
    weights = CODEBOOK_LOSS_WEIGHTS[:NUM_CODEBOOKS].to(codebook_losses.device)
    valid_losses = ~torch.isnan(codebook_losses)
    if valid_losses.sum() == 0:
        total_loss = torch.tensor(float("nan"), device=logits.device)
    else:
        w = weights[valid_losses]
        losses = codebook_losses[valid_losses]
        total_loss = (losses * w).sum() / w.sum()
    return total_loss, per_codebook_loss, per_codebook_accuracy


# ── Copied from evaluation_test_slakh.ipynb ──
def REFERENCE_compute_batch_metrics(batch, model):
    model_batch = REFERENCE_prepare_batch_for_model(batch)
    logits = model(
        input_codes=model_batch["input_codes"],
        input_tension=model_batch["input_tension"],
        input_combat=model_batch["input_combat"],
        target_tension=model_batch["target_tension"],
        target_combat=model_batch["target_combat"],
    )
    return REFERENCE_multi_codebook_metrics(
        logits=logits, target_codes=model_batch["target_codes"]
    )


@torch.no_grad()
def REFERENCE_evaluate_on_loader(model, loader):
    model.eval()
    total_loss = 0.0
    total_per_codebook_loss = torch.zeros(NUM_CODEBOOKS, device=device)
    total_per_codebook_accuracy = torch.zeros(NUM_CODEBOOKS, device=device)
    batch_count = 0
    for batch in loader:
        loss, per_codebook_loss, per_codebook_accuracy = (
            REFERENCE_compute_batch_metrics(batch, model)
        )
        total_loss += loss.detach().item()
        total_per_codebook_loss += per_codebook_loss.detach()
        total_per_codebook_accuracy += per_codebook_accuracy.detach()
        batch_count += 1
    return {
        "loss": total_loss / batch_count,
        "per_codebook_loss": (total_per_codebook_loss / batch_count).cpu().tolist(),
        "per_codebook_accuracy": (total_per_codebook_accuracy / batch_count).cpu().tolist(),
    }


# ── Helpers ──
SEGMENT_FRAMES = 40


def make_batch(batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "tokens": torch.randint(
            0, 1024, (batch_size, NUM_CODEBOOKS, SEGMENT_FRAMES),
            generator=generator,
        ),
        "tension": torch.rand(batch_size, SEGMENT_FRAMES, generator=generator),
        "combat_score": torch.rand(batch_size, SEGMENT_FRAMES, generator=generator),
    }


def make_small_model(seed=0):
    torch.manual_seed(seed)
    model = SecundaTransformer(
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
        max_sequence_length=SEGMENT_FRAMES + NUM_CODEBOOKS - 1,
    )
    model.eval()
    return model


# ── Tests ──
def test_lr_at_step_matches_notebook():
    for step in [0, 1, 100, 3_999, 4_000, 4_001, 20_000, 35_219, 35_220, 50_000]:
        expected = REFERENCE_lr_at_step(step)
        observed = training_utils.lr_at_step(
            step,
            peak_lr=PEAK_LR,
            min_lr=MIN_LR,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
        )
        assert observed == expected, (step, observed, expected)


def test_prepare_batch_for_model_matches_notebook():
    batch = make_batch(batch_size=3, seed=1)
    expected = REFERENCE_prepare_batch_for_model(batch)
    observed = training_utils.prepare_batch_for_model(
        batch, serializer=serializer, device=device
    )
    assert expected.keys() == observed.keys()
    for key in expected:
        assert torch.equal(expected[key], observed[key]), key


def test_multi_codebook_metrics_matches_notebook():
    model = make_small_model()
    model_batch = REFERENCE_prepare_batch_for_model(make_batch(batch_size=3, seed=2))
    with torch.no_grad():
        logits = model(
            input_codes=model_batch["input_codes"],
            input_tension=model_batch["input_tension"],
            input_combat=model_batch["input_combat"],
            target_tension=model_batch["target_tension"],
            target_combat=model_batch["target_combat"],
        )
    expected = REFERENCE_multi_codebook_metrics(logits, model_batch["target_codes"])
    # Weights passed both as a tensor and as a plain list.
    for weights in (CODEBOOK_LOSS_WEIGHTS, CODEBOOK_LOSS_WEIGHTS.tolist()):
        observed = training_utils.multi_codebook_metrics(
            logits, model_batch["target_codes"], codebook_loss_weights=weights
        )
        for expected_value, observed_value in zip(expected, observed):
            assert torch.equal(expected_value, observed_value)


def test_evaluate_on_loader_matches_notebook():
    model = make_small_model()
    # Uneven last batch, as in the real test split (151 = 6 x 24 + 7).
    loader = [make_batch(4, seed=3), make_batch(4, seed=4), make_batch(2, seed=5)]
    expected = REFERENCE_evaluate_on_loader(model, loader)
    observed = training_utils.evaluate_on_loader(
        model,
        loader,
        serializer=serializer,
        device=device,
        codebook_loss_weights=CODEBOOK_LOSS_WEIGHTS,
    )
    assert expected == observed


def test_load_checkpoint_model_round_trip(tmp_path=None):
    import tempfile

    model = make_small_model(seed=7)
    config = {
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_dim": 64,
        "dropout": 0.0,
        "max_sequence_length": SEGMENT_FRAMES + NUM_CODEBOOKS - 1,
    }
    model_batch = REFERENCE_prepare_batch_for_model(make_batch(batch_size=2, seed=8))

    with tempfile.TemporaryDirectory() as directory:
        for drop_max_sequence_length in (False, True):
            saved_config = dict(config)
            if drop_max_sequence_length:
                # Older checkpoints: length inferred from the weights.
                del saved_config["max_sequence_length"]
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": saved_config}, path)

            loaded_model, checkpoint = training_utils.load_checkpoint_model(path, device)
            assert not loaded_model.training
            with torch.no_grad():
                inputs = dict(
                    input_codes=model_batch["input_codes"],
                    input_tension=model_batch["input_tension"],
                    input_combat=model_batch["input_combat"],
                    target_tension=model_batch["target_tension"],
                    target_combat=model_batch["target_combat"],
                )
                assert torch.equal(model(**inputs), loaded_model(**inputs))


if __name__ == "__main__":
    tests = [
        test_lr_at_step_matches_notebook,
        test_prepare_batch_for_model_matches_notebook,
        test_multi_codebook_metrics_matches_notebook,
        test_evaluate_on_loader_matches_notebook,
        test_load_checkpoint_model_round_trip,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All training_utils tests passed.")
