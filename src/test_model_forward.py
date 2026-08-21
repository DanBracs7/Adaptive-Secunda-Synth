from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_dataset import ConditionalAudioDataset
from frame_codec import (
    CodebookSerializer,
    multi_codebook_cross_entropy,
    prepare_next_frame_batch,
)
from model import DelayedCodebookTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT /
    "manifests" /
    "all_tracks.csv"
)

SOURCE_ROOTS = {
    "slakh": (
        PROJECT_ROOT /
        "datasets" /
        "slakh_final_388tensors" /
        "tensors_final" /
        "train"
    ),
    "skyrim": PROJECT_ROOT / "datasets" / "skyrim",
    "witcher3": PROJECT_ROOT / "datasets" / "witcher3",
}

BATCH_SIZE = 2
SEGMENT_SECONDS = 5.0
FRAME_RATE = 75


def get_prepared_batch():
    dataset = ConditionalAudioDataset(
        manifest_path=MANIFEST_PATH,
        source_roots=SOURCE_ROOTS,
        split="train",
        stage="pretrain",
        segment_seconds=SEGMENT_SECONDS,
        frame_rate=FRAME_RATE,
        fixed_start=True,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    return prepare_next_frame_batch(
        batch=batch,
        serializer=CodebookSerializer(),
    )


def main():
    torch.manual_seed(42)

    prepared = get_prepared_batch()

    model = DelayedCodebookTransformer(
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=512,
        dropout=0.10,
        max_sequence_length=800,
    )

    input_tension = prepared["target_tension"][:, 0, :]
    input_combat = prepared["target_combat"][:, 0, :]

    logits = model(
        input_codes=prepared["input_codes"],
        input_tension=input_tension,
        input_combat=input_combat,
        target_tension=prepared["target_tension"],
        target_combat=prepared["target_combat"],
    )

    expected_shape = (
        BATCH_SIZE,
        32,
        405,
        1025,
    )

    assert tuple(logits.shape) == expected_shape
    assert torch.isfinite(logits).all()

    loss = multi_codebook_cross_entropy(
        logits=logits,
        target_codes=prepared["target_codes"],
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()

    parameters_with_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not parameters_with_gradients:
        raise RuntimeError(
            "No gradients were produced."
        )

    finite_gradients = all(
        torch.isfinite(parameter.grad).all()
        for parameter in parameters_with_gradients
    )

    if not finite_gradients:
        raise RuntimeError(
            "At least one gradient contains NaN or infinity."
        )

    nonzero_gradient_count = sum(
        int(parameter.grad.abs().sum() > 0)
        for parameter in parameters_with_gradients
    )

    if nonzero_gradient_count == 0:
        raise RuntimeError(
            "All gradients are zero."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model-forward test passed.")
    print("input codes:", tuple(prepared["input_codes"].shape))
    print("target codes:", tuple(prepared["target_codes"].shape))
    print("logits:", tuple(logits.shape))
    print(f"loss: {loss.detach().item():.4f}")
    print(f"model parameters: {parameter_count:,}")
    print(
        "parameters with non-zero gradients:",
        f"{nonzero_gradient_count}/{len(parameters_with_gradients)}",
    )


if __name__ == "__main__":
    main()