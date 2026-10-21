"""
Tests for src/generation.py, on a small randomly initialised model (no
checkpoint or GPU needed).

1. The delay pattern round-trips (undelay(delay(codes)) == codes).
2. The generation conditions are identical to the ones built in training
   by training_utils.prepare_batch_for_model.
3. Oracle consistency: if the loop writes the real codes instead of
   sampling, the logits of every step equal those of one teacher-forced
   forward pass, and the filled buffer equals serializer.delay(codes).
   This checks that every cell is predicted from the same context and
   conditions as in training.
4. generate_codes keeps the prompt, never outputs PAD, and is
   reproducible with a seed.
5. Gumbel-max sampling follows the top-k softmax distribution and never
   returns PAD or an ID outside the top k.
6. With shared_noise=True (paired design) identical controls give
   identical samples, and a sample does not depend on the other batch
   elements.
7. The key/value cache gives the same logits as re-running the whole
   prefix (test 3 is run with and without the cache) and the same codes.
8. Invalid inputs are refused (controls outside [0, 1], empty prompt,
   segment longer than the positional embedding).

Run from the repository root:
    python tests/test_generation.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frame_codec import CODEBOOK_SIZE, NUM_CODEBOOKS, PAD_TOKEN_ID, CodebookSerializer  # noqa: E402
from generation import (  # noqa: E402
    _run_delayed_loop,
    build_generation_conditions,
    generate_codes,
    sample_top_k,
    ramp_control,
    step_control,
    with_prompt_controls,
)
from model import SecundaTransformer  # noqa: E402
from training_utils import prepare_batch_for_model  # noqa: E402

TOTAL_FRAMES = 40
PROMPT_FRAMES = 6


def make_model(seed=0, max_sequence_length=TOTAL_FRAMES + NUM_CODEBOOKS - 2):
    torch.manual_seed(seed)
    model = SecundaTransformer(
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.1,
        max_sequence_length=max_sequence_length,
    )
    model.eval()
    return model


def make_batch(seed=1, batch_size=2, total_frames=TOTAL_FRAMES):
    generator = torch.Generator().manual_seed(seed)
    return {
        "tokens": torch.randint(0, CODEBOOK_SIZE, (batch_size, NUM_CODEBOOKS, total_frames), generator=generator),
        "tension": torch.rand(batch_size, total_frames, generator=generator),
        "combat_score": torch.rand(batch_size, total_frames, generator=generator),
    }


def test_delay_round_trip():
    serializer = CodebookSerializer()
    codes = make_batch()["tokens"]
    assert torch.equal(serializer.undelay(serializer.delay(codes)), codes)


def test_conditions_match_training():
    batch = make_batch()
    training = prepare_batch_for_model(batch, CodebookSerializer(), torch.device("cpu"))
    generation = build_generation_conditions(batch["tension"], batch["combat_score"])

    for key in ("input_tension", "input_combat", "target_tension", "target_combat"):
        assert generation[key].shape == training[key].shape, key
        assert torch.equal(generation[key], training[key]), key


def test_stepwise_logits_match_teacher_forcing(use_cache=True):
    model = make_model()
    batch = make_batch(batch_size=1)
    serializer = CodebookSerializer()

    # One teacher-forced forward pass, as in training and evaluation.
    training = prepare_batch_for_model(batch, serializer, torch.device("cpu"))
    with torch.no_grad():
        teacher_forced_logits = model(
            input_codes=training["input_codes"],
            input_tension=training["input_tension"],
            input_combat=training["input_combat"],
            target_tension=training["target_tension"],
            target_combat=training["target_combat"],
        )

    real_delayed = serializer.delay(batch["tokens"])
    max_difference = 0.0
    sampled_cells = torch.zeros_like(real_delayed, dtype=torch.bool)

    def write_real_codes(step, logits, frame_index, to_sample):
        nonlocal max_difference
        # Input position step - 1 predicts delayed position step.
        reference = teacher_forced_logits[:, :, step - 1, :]
        difference = (logits - reference)[:, to_sample].abs().max().item()
        max_difference = max(max_difference, difference)
        sampled_cells[:, to_sample, step] = True
        return real_delayed[:, :, step]

    prompt = batch["tokens"][:, :, :PROMPT_FRAMES]
    with torch.no_grad():
        filled = _run_delayed_loop(
            model, prompt, batch["tension"], batch["combat_score"], write_real_codes, use_cache=use_cache
        )

    assert max_difference < 1e-4, max_difference
    assert torch.equal(filled, real_delayed)

    # Exactly the non-PAD cells after the prompt were generated, once each.
    expected = real_delayed != PAD_TOKEN_ID
    for q in range(NUM_CODEBOOKS):
        expected[:, q, q:q + PROMPT_FRAMES] = False
    assert torch.equal(sampled_cells, expected)


def test_generate_codes_keeps_prompt_and_is_reproducible():
    model = make_model()
    batch = make_batch(batch_size=2)
    prompt = batch["tokens"][:, :, :PROMPT_FRAMES]

    tension = torch.stack([
        with_prompt_controls(ramp_control(0.2, 0.8, TOTAL_FRAMES), batch["tension"][0, :PROMPT_FRAMES]),
        with_prompt_controls(step_control(0.2, 0.8, TOTAL_FRAMES, 20), batch["tension"][1, :PROMPT_FRAMES]),
    ])
    combat_score = batch["combat_score"]

    first = generate_codes(model, prompt, tension, combat_score, seed=7)
    second = generate_codes(model, prompt, tension, combat_score, seed=7)
    other_seed = generate_codes(model, prompt, tension, combat_score, seed=8)

    assert first.shape == (2, NUM_CODEBOOKS, TOTAL_FRAMES)
    assert torch.equal(first[:, :, :PROMPT_FRAMES], prompt)
    assert int(first.min()) >= 0 and int(first.max()) < CODEBOOK_SIZE
    assert torch.equal(first, second)
    assert not torch.equal(first, other_seed)


def test_sampling_matches_top_k_distribution():
    vocab_size = PAD_TOKEN_ID + 1
    logits = torch.full((vocab_size,), -10.0)
    logits[[3, 7, 11, 500]] = torch.tensor([2.0, 1.0, 0.5, -1.0])
    logits[PAD_TOKEN_ID] = 5.0  # largest logit, must never be sampled

    draws = 200_000
    generator = torch.Generator().manual_seed(0)
    samples = sample_top_k(logits.expand(draws, -1), temperature=1.0, top_k=3, generator=generator)

    expected_ids = torch.tensor([3, 7, 11])
    expected = torch.softmax(logits[expected_ids], dim=0)
    observed = torch.stack([(samples == index).float().mean() for index in expected_ids])

    assert set(samples.unique().tolist()) <= set(expected_ids.tolist())
    assert torch.allclose(observed, expected, atol=0.005), (observed, expected)


def test_shared_noise_gives_paired_samples():
    model = make_model()
    batch = make_batch(batch_size=1)
    prompt = batch["tokens"][:, :, :PROMPT_FRAMES]
    low = with_prompt_controls(torch.full((TOTAL_FRAMES,), 0.1), batch["tension"][0, :PROMPT_FRAMES])
    high = with_prompt_controls(torch.full((TOTAL_FRAMES,), 0.9), batch["tension"][0, :PROMPT_FRAMES])
    combat_score = batch["combat_score"]

    # Same controls twice in one batch: identical rows.
    same = generate_codes(
        model, prompt.expand(2, -1, -1), torch.stack([low, low]), combat_score.expand(2, -1),
        seed=3, shared_noise=True,
    )
    assert torch.equal(same[0], same[1])

    # A sample does not depend on the other elements of the batch.
    paired = generate_codes(
        model, prompt.expand(2, -1, -1), torch.stack([low, high]), combat_score.expand(2, -1),
        seed=3, shared_noise=True,
    )
    alone = generate_codes(model, prompt, low[None], combat_score, seed=3, shared_noise=True)
    assert torch.equal(paired[0], alone[0])
    assert not torch.equal(paired[0], paired[1])

    # Without shared noise the rows are independent samples.
    independent = generate_codes(
        model, prompt.expand(2, -1, -1), torch.stack([low, low]), combat_score.expand(2, -1),
        seed=3, shared_noise=False,
    )
    assert not torch.equal(independent[0], independent[1])


def test_stepwise_logits_without_cache():
    test_stepwise_logits_match_teacher_forcing(use_cache=False)


def test_cache_gives_the_same_codes():
    model = make_model()
    batch = make_batch(batch_size=2)
    prompt = batch["tokens"][:, :, :PROMPT_FRAMES]
    cached = generate_codes(model, prompt, batch["tension"], batch["combat_score"], seed=5, use_cache=True)
    recomputed = generate_codes(model, prompt, batch["tension"], batch["combat_score"], seed=5, use_cache=False)
    assert torch.equal(cached, recomputed)


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("ValueError not raised")


def test_invalid_inputs_are_refused():
    model = make_model()
    batch = make_batch(batch_size=1)
    prompt = batch["tokens"][:, :, :PROMPT_FRAMES]
    tension, combat_score = batch["tension"], batch["combat_score"]

    # 0-9 scale of the first-submission script.
    expect_value_error(lambda: generate_codes(model, prompt, tension * 9, combat_score))
    # Empty prompt.
    expect_value_error(lambda: generate_codes(model, prompt[:, :, :0], tension, combat_score))
    # Longer than the positional embedding allows.
    long_controls = torch.rand(1, TOTAL_FRAMES + 1)
    expect_value_error(lambda: generate_codes(model, prompt, long_controls, long_controls))


if __name__ == "__main__":
    tests = [
        test_delay_round_trip,
        test_conditions_match_training,
        test_stepwise_logits_match_teacher_forcing,
        test_stepwise_logits_without_cache,
        test_cache_gives_the_same_codes,
        test_generate_codes_keeps_prompt_and_is_reproducible,
        test_sampling_matches_top_k_distribution,
        test_shared_noise_gives_paired_samples,
        test_invalid_inputs_are_refused,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All generation tests passed.")
