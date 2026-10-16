"""
Training and evaluation helpers shared by all notebooks.

lr_at_step, prepare_batch_for_model, multi_codebook_metrics,
compute_batch_metrics and evaluate_on_loader were previously redefined
inside each notebook (slakh-baseline-training-8kb,
finetune-of-slakh-baseline-training-8kb, evaluation_test_slakh). They are
moved here with unchanged behaviour: the only difference is that values
the notebooks read from global variables (model, serializer, device,
CODEBOOK_LOSS_WEIGHTS, PEAK_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS) are now
explicit parameters. tests/test_training_utils.py checks that the results
are identical to the notebook versions.

multi_codebook_metrics is the loss that produced every reported number:
the mean cross-entropy of each codebook (PAD targets excluded), combined
with a weighted average over codebooks.
"""

import math

import torch
import torch.nn.functional as F

from frame_codec import (
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    PAD_TOKEN_ID,
    prepare_next_frame_batch,
)
from model import SecundaTransformer


def lr_at_step(step, peak_lr, min_lr, warmup_steps, max_steps):
    # Warmup
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)

    # Cosine decay from peak_lr to min_lr over [warmup_steps..max_steps]
    progress = min(max(step - warmup_steps, 0), max_steps - warmup_steps)
    decay_ratio = progress / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (peak_lr - min_lr) * cosine


def prepare_batch_for_model(batch, serializer, device):
    prepared = prepare_next_frame_batch(
        batch=batch,
        serializer=serializer,
    )

    # Codebook 0 target at delayed step s maps to original frame s + 1.
    # This shared condition tells the Transformer the desired next
    # primary-codebook audio state.
    input_tension = prepared["target_tension"][
        :,
        0,
        :,
    ]

    input_combat = prepared["target_combat"][
        :,
        0,
        :,
    ]

    return {
        "input_codes": prepared["input_codes"].to(
            device,
            non_blocking=True,
        ),
        "target_codes": prepared["target_codes"].to(
            device,
            non_blocking=True,
        ),
        "input_tension": input_tension.to(
            device,
            non_blocking=True,
        ),
        "input_combat": input_combat.to(
            device,
            non_blocking=True,
        ),
        "target_tension": prepared["target_tension"].to(
            device,
            non_blocking=True,
        ),
        "target_combat": prepared["target_combat"].to(
            device,
            non_blocking=True,
        ),
    }


def multi_codebook_metrics(
    logits,
    target_codes,
    codebook_loss_weights,
):
    """
    Returns:
        total_loss: scalar tensor (weighted)
        per_codebook_loss: [NUM_CODEBOOKS] tensor (unweighted, for logging)
        per_codebook_accuracy: [NUM_CODEBOOKS] tensor

    PAD targets are excluded from every metric.
    """
    if logits.shape[1] != NUM_CODEBOOKS:
        raise ValueError(
            f"Expected logits with {NUM_CODEBOOKS} codebooks, "
            f"got shape {tuple(logits.shape)}."
        )

    # Keep the original unweighted per-codebook losses for logging.
    per_codebook_loss = []
    per_codebook_accuracy = []

    codebook_losses = []  # will hold unweighted losses as tensors

    for codebook_index in range(NUM_CODEBOOKS):
        codebook_logits = logits[
            :,
            codebook_index,
            :,
            :,
        ]

        codebook_targets = target_codes[
            :,
            codebook_index,
            :,
        ]

        valid_mask = codebook_targets != PAD_TOKEN_ID

        if valid_mask.sum() == 0:
            nan_tensor = torch.tensor(
                float("nan"),
                device=logits.device,
            )
            per_codebook_loss.append(nan_tensor)
            per_codebook_accuracy.append(nan_tensor)
            codebook_losses.append(nan_tensor)
            continue

        codebook_loss = F.cross_entropy(
            codebook_logits[valid_mask],
            codebook_targets[valid_mask],
        )

        predictions = codebook_logits.argmax(dim=-1)

        codebook_accuracy = (
            predictions[valid_mask] ==
            codebook_targets[valid_mask]
        ).float().mean()

        per_codebook_loss.append(codebook_loss)
        per_codebook_accuracy.append(codebook_accuracy)
        codebook_losses.append(codebook_loss)

    # Stack unweighted losses for logging
    per_codebook_loss = torch.stack(per_codebook_loss)
    per_codebook_accuracy = torch.stack(per_codebook_accuracy)
    codebook_losses = torch.stack(codebook_losses)  # [Q]

    # Apply codebook weights to compute the optimized total_loss
    weights = torch.as_tensor(
        codebook_loss_weights,
        dtype=torch.float32,
        device=codebook_losses.device,
    )[:NUM_CODEBOOKS]

    # Ignore NaN codebooks (if any) in the weighted average
    valid_losses = ~torch.isnan(codebook_losses)
    if valid_losses.sum() == 0:
        total_loss = torch.tensor(
            float("nan"),
            device=logits.device,
        )
    else:
        w = weights[valid_losses]
        losses = codebook_losses[valid_losses]
        total_loss = (losses * w).sum() / w.sum()

    return total_loss, per_codebook_loss, per_codebook_accuracy


def compute_batch_metrics(
    batch,
    model,
    serializer,
    device,
    codebook_loss_weights,
):
    model_batch = prepare_batch_for_model(
        batch=batch,
        serializer=serializer,
        device=device,
    )

    logits = model(
        input_codes=model_batch["input_codes"],
        input_tension=model_batch["input_tension"],
        input_combat=model_batch["input_combat"],
        target_tension=model_batch["target_tension"],
        target_combat=model_batch["target_combat"],
    )

    return multi_codebook_metrics(
        logits=logits,
        target_codes=model_batch["target_codes"],
        codebook_loss_weights=codebook_loss_weights,
    )


@torch.no_grad()
def evaluate_on_loader(
    model,
    loader,
    serializer,
    device,
    codebook_loss_weights,
):
    """
    Same reduction as evaluation_test_slakh.ipynb: every metric is the
    mean of the per-batch values (the last, smaller batch counts as one
    batch). Kept unchanged so that Table 1 is reproduced exactly.
    """
    model.eval()

    total_loss = 0.0
    total_per_codebook_loss = torch.zeros(NUM_CODEBOOKS, device=device)
    total_per_codebook_accuracy = torch.zeros(NUM_CODEBOOKS, device=device)
    batch_count = 0

    for batch in loader:
        loss, per_codebook_loss, per_codebook_accuracy = compute_batch_metrics(
            batch=batch,
            model=model,
            serializer=serializer,
            device=device,
            codebook_loss_weights=codebook_loss_weights,
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


REQUIRED_CONFIG_KEYS = (
    "embedding_dim",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "dropout",
)


def load_checkpoint_model(checkpoint_path, device):
    """
    Rebuild SecundaTransformer from a training checkpoint saved by the
    training notebooks. Returns (model, checkpoint); the model is in eval
    mode.

    Hyperparameters come from checkpoint["config"] and are cross-checked
    against the shapes stored in the state_dict. When the config has no
    "max_sequence_length" (older checkpoints), it is read from the
    positional-embedding shape.
    """
    # weights_only=False: these are our own checkpoints and they also
    # store optimizer state, history and config, as in
    # evaluation_test_slakh.ipynb.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    missing_keys = [
        key for key in REQUIRED_CONFIG_KEYS
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            f"{checkpoint_path}: config is missing {missing_keys}"
        )

    position_rows, stored_embedding_dim = (
        state_dict["position_embedding.weight"].shape
    )

    stored_num_layers = len({
        key.split(".")[2]
        for key in state_dict
        if key.startswith("transformer.layers.")
    })

    if stored_embedding_dim != config["embedding_dim"]:
        raise ValueError(
            f"{checkpoint_path}: config embedding_dim="
            f"{config['embedding_dim']} but weights have "
            f"{stored_embedding_dim}."
        )

    if stored_num_layers != config["num_layers"]:
        raise ValueError(
            f"{checkpoint_path}: config num_layers="
            f"{config['num_layers']} but weights have "
            f"{stored_num_layers}."
        )

    max_sequence_length = config.get(
        "max_sequence_length",
        position_rows,
    )

    if max_sequence_length != position_rows:
        raise ValueError(
            f"{checkpoint_path}: config max_sequence_length="
            f"{max_sequence_length} but weights have {position_rows}."
        )

    model = SecundaTransformer(
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_sequence_length=max_sequence_length,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, checkpoint
