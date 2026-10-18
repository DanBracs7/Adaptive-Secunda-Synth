"""
Experiment E1: does the model use its conditions?

The professor's main objection: all reported metrics are teacher-forced
cross-entropies, which would look similar even if the FiLM conditioning
were disconnected. E1 tests exactly this. The same test segments are
evaluated with the real conditions and with conditions that carry less or
wrong information; if the model uses the conditions, the cross-entropy of
the real tokens must increase when they are replaced.

The conditions are replaced at frame level, before the delay pattern is
applied (prepare_batch_for_model), so every variant goes through the same
code path as training.

Variants:
    real             conditions stored in the tensor (Table 1)
    zero             tension = combat_score = 0 everywhere. 0 is a valid
                     value (minimum intensity), not "no condition": the
                     model has no null condition.
    constant         a constant value for the whole segment: the median of
                     the Slakh train+val conditions (notebook 02, Part A).
                     Same level for every segment, no temporal information.
    shuffled_time    the segment's own values in a random temporal order
                     (same permutation for tension and combat_score): right
                     distribution, wrong timing.
    swapped          the conditions of another segment of the batch (half a
                     batch away, so that with several crops per track it
                     comes from a different track): realistic curves that
                     belong to different audio.
    tension_swapped  only tension from the other segment, real combat_score
    combat_swapped   only combat_score from the other segment, real tension
"""

import numpy as np
import torch
import torch.nn.functional as F

from frame_codec import NUM_CODEBOOKS, PAD_TOKEN_ID


ABLATIONS = (
    "real",
    "zero",
    "constant",
    "shuffled_time",
    "swapped",
    "tension_swapped",
    "combat_swapped",
)

# Median of the Slakh train+val conditions (notebook 02, Part A).
MEDIAN_CONDITIONS = {"tension": 0.41, "combat_score": 0.60}


def ablate_conditions(tension, combat_score, mode, generator=None, constant=MEDIAN_CONDITIONS):
    """
    tension, combat_score: [B, T] real conditions of a batch.
    Returns new (tension, combat_score) tensors for the given variant.
    generator is used only by "shuffled_time".
    """
    if mode not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{mode}'. Known: {ABLATIONS}")

    if tension.shape != combat_score.shape or tension.ndim != 2:
        raise ValueError("tension and combat_score must both have shape [B, T].")

    if mode == "real":
        return tension.clone(), combat_score.clone()

    if mode == "zero":
        return torch.zeros_like(tension), torch.zeros_like(combat_score)

    if mode == "constant":
        return (
            torch.full_like(tension, constant["tension"]),
            torch.full_like(combat_score, constant["combat_score"]),
        )

    if mode == "shuffled_time":
        batch_size, time_steps = tension.shape
        noise = torch.rand(batch_size, time_steps, generator=generator)
        order = torch.argsort(noise, dim=1).to(tension.device)
        return tension.gather(1, order), combat_score.gather(1, order)

    # The remaining variants take the conditions of the segment half a
    # batch away (segment i gets the conditions of segment i + B // 2).
    batch_size = tension.shape[0]
    if batch_size < 2:
        raise ValueError(f"'{mode}' needs at least two segments in the batch.")

    shift = batch_size // 2
    other_tension = tension.roll(shifts=-shift, dims=0)
    other_combat = combat_score.roll(shifts=-shift, dims=0)

    if mode == "swapped":
        return other_tension.clone(), other_combat.clone()
    if mode == "tension_swapped":
        return other_tension.clone(), combat_score.clone()
    return tension.clone(), other_combat.clone()  # combat_swapped


def per_segment_cross_entropy(logits, target_codes):
    """
    logits: [B, Q, S, V], target_codes: [B, Q, S]
    Returns (mean_ce [B, Q], token_count [B, Q]): the cross-entropy of
    every segment and codebook averaged over its non-PAD targets, i.e.
    the per-segment version of the loss in training_utils.multi_codebook_metrics.
    """
    batch_size, num_codebooks, sequence_length, vocab_size = logits.shape

    if num_codebooks != NUM_CODEBOOKS:
        raise ValueError(f"Expected {NUM_CODEBOOKS} codebooks, got {num_codebooks}.")

    token_ce = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        target_codes.reshape(-1).clamp(max=vocab_size - 1),
        reduction="none",
    ).reshape(batch_size, num_codebooks, sequence_length)

    valid = target_codes != PAD_TOKEN_ID
    token_count = valid.sum(dim=-1)
    mean_ce = (token_ce * valid).sum(dim=-1) / token_count.clamp(min=1)

    return mean_ce, token_count


def bootstrap_mean_ci(values, num_resamples=10_000, confidence=0.95, seed=0):
    """Mean of `values` and a percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(num_resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(low), float(high)
