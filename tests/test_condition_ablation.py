"""
Tests for src/condition_ablation.py.

1. Every variant keeps the shapes; "real" returns the inputs unchanged.
2. "shuffled_time" only reorders each segment's values, with the same
   order for tension and combat_score.
3. The swapped variants take the conditions of the segment half a batch
   away (B // 2), never the segment's own.
4. per_segment_cross_entropy agrees with training_utils.multi_codebook_metrics:
   for a single segment the per-codebook values are identical.
5. bootstrap_mean_ci returns the mean and an interval that contains it.

Run from the repository root:
    python tests/test_condition_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from condition_ablation import (  # noqa: E402
    ABLATIONS,
    MEDIAN_CONDITIONS,
    ablate_conditions,
    bootstrap_mean_ci,
    per_segment_cross_entropy,
)
from frame_codec import CODEBOOK_SIZE, NUM_CODEBOOKS, CodebookSerializer  # noqa: E402
from model import SecundaTransformer  # noqa: E402
from training_utils import multi_codebook_metrics, prepare_batch_for_model  # noqa: E402


def make_conditions(batch_size=3, time_steps=50, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, time_steps, generator=generator), torch.rand(batch_size, time_steps, generator=generator)


def test_shapes_and_real():
    tension, combat = make_conditions()
    for mode in ABLATIONS:
        new_tension, new_combat = ablate_conditions(tension, combat, mode, generator=torch.Generator().manual_seed(1))
        assert new_tension.shape == tension.shape and new_combat.shape == combat.shape, mode
    real_tension, real_combat = ablate_conditions(tension, combat, "real")
    assert torch.equal(real_tension, tension) and torch.equal(real_combat, combat)

    zero_tension, zero_combat = ablate_conditions(tension, combat, "zero")
    assert zero_tension.abs().sum() == 0 and zero_combat.abs().sum() == 0

    constant_tension, constant_combat = ablate_conditions(tension, combat, "constant")
    assert torch.all(constant_tension == MEDIAN_CONDITIONS["tension"])
    assert torch.all(constant_combat == MEDIAN_CONDITIONS["combat_score"])


def test_shuffled_time_is_a_joint_permutation():
    tension, combat = make_conditions()
    new_tension, new_combat = ablate_conditions(tension, combat, "shuffled_time", generator=torch.Generator().manual_seed(1))
    assert not torch.equal(new_tension, tension)
    for row in range(tension.shape[0]):
        assert torch.equal(torch.sort(new_tension[row]).values, torch.sort(tension[row]).values)
        # Same permutation: every (tension, combat) pair is preserved.
        original_pairs = sorted(zip(tension[row].tolist(), combat[row].tolist()))
        new_pairs = sorted(zip(new_tension[row].tolist(), new_combat[row].tolist()))
        assert original_pairs == new_pairs


def test_swapped_variants_take_another_segment():
    tension, combat = make_conditions(batch_size=8)
    swapped_tension, swapped_combat = ablate_conditions(tension, combat, "swapped")
    for row in range(8):
        assert torch.equal(swapped_tension[row], tension[(row + 4) % 8])
        assert torch.equal(swapped_combat[row], combat[(row + 4) % 8])

    only_tension, kept_combat = ablate_conditions(tension, combat, "tension_swapped")
    assert torch.equal(only_tension[0], tension[4]) and torch.equal(kept_combat, combat)

    kept_tension, only_combat = ablate_conditions(tension, combat, "combat_swapped")
    assert torch.equal(kept_tension, tension) and torch.equal(only_combat[0], combat[4])

    # Odd batch (the last, smaller batch of a loader): still never itself.
    tension, combat = make_conditions(batch_size=7)
    swapped_tension, _ = ablate_conditions(tension, combat, "swapped")
    assert all(not torch.equal(swapped_tension[row], tension[row]) for row in range(7))


def test_per_segment_ce_matches_training_loss():
    torch.manual_seed(0)
    model = SecundaTransformer(embedding_dim=32, num_layers=2, num_heads=4, feedforward_dim=64,
                               dropout=0.1, max_sequence_length=60).eval()
    generator = torch.Generator().manual_seed(2)
    batch = {
        "tokens": torch.randint(0, CODEBOOK_SIZE, (2, NUM_CODEBOOKS, 40), generator=generator),
        "tension": torch.rand(2, 40, generator=generator),
        "combat_score": torch.rand(2, 40, generator=generator),
    }
    prepared = prepare_batch_for_model(batch, CodebookSerializer(), torch.device("cpu"))
    with torch.no_grad():
        logits = model(
            input_codes=prepared["input_codes"],
            input_tension=prepared["input_tension"],
            input_combat=prepared["input_combat"],
            target_tension=prepared["target_tension"],
            target_combat=prepared["target_combat"],
        )

    mean_ce, token_count = per_segment_cross_entropy(logits, prepared["target_codes"])
    assert mean_ce.shape == (2, NUM_CODEBOOKS)
    # Delayed position 0 is never a target, so codebook 0 has T - 1 targets
    # (its frame 0 is only input) and the delayed codebooks have all T.
    assert torch.all(token_count[:, 0] == 40 - 1) and torch.all(token_count[:, 1:] == 40)

    for segment in range(2):
        _, per_codebook, _ = multi_codebook_metrics(
            logits[segment:segment + 1], prepared["target_codes"][segment:segment + 1], [1.0] * NUM_CODEBOOKS
        )
        assert torch.allclose(mean_ce[segment], per_codebook, atol=1e-5), (mean_ce[segment], per_codebook)


def test_bootstrap_interval_contains_mean():
    values = np.random.default_rng(0).normal(0.2, 1.0, size=200)
    mean, low, high = bootstrap_mean_ci(values, num_resamples=2000)
    assert low < mean < high
    assert abs(mean - values.mean()) < 1e-12


if __name__ == "__main__":
    tests = [
        test_shapes_and_real,
        test_shuffled_time_is_a_joint_permutation,
        test_swapped_variants_take_another_segment,
        test_per_segment_ce_matches_training_loss,
        test_bootstrap_interval_contains_mean,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All condition_ablation tests passed.")
