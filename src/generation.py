"""
Autoregressive generation with SecundaTransformer.

Replaces src/archive/generation_test_8cb.py, which had four problems:
  1. it sampled one frame of all 8 codebooks per step, ignoring the delay
     pattern the model was trained with;
  2. its control profiles used a 0-9 scale instead of the [0, 1] scale of
     the training conditions;
  3. its conditions were shifted by one frame and not delayed per codebook;
  4. it started from an empty prompt (a context never seen in training).

Here generation runs in the delayed space, exactly mirroring training
(frame_codec.prepare_next_frame_batch + training_utils.prepare_batch_for_model):

  * The model reads the delayed sequence D[:, :, :s] and, at its last
    position, predicts delayed position s of every codebook. Codebook q at
    delayed position s holds original frame f = s - q.
  * At step s only the cells with prompt_frames <= f < total_frames are
    sampled. Cells with f < 0 or f >= total_frames stay PAD (as in
    training), cells with f < prompt_frames keep the prompt codes.
  * Conditions are delayed per codebook with delay_conditions_by_codebook
    and shifted to the target positions; the shared input condition is the
    codebook-0 target condition, as in prepare_batch_for_model.
  * The delayed codes are converted back with CodebookSerializer.undelay.

In training, delayed position 0 is never a target (inputs are D[:, :, :-1],
targets D[:, :, 1:]), so the model cannot produce the first frame of
codebook 0 from nothing: a prompt of at least one real frame is required.

Each step reuses the keys and values of the previous positions
(KeyValueCache), so only the newest position is processed; this gives the
same logits as re-running the whole prefix and is several times faster.

tests/test_generation.py checks the conditions against
prepare_batch_for_model and the step-by-step logits (with and without the
cache) against a single teacher-forced forward pass.
"""

import time

import numpy as np
import torch

from frame_codec import (
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    PAD_TOKEN_ID,
    CodebookSerializer,
    delay_conditions_by_codebook,
)


SAMPLE_RATE = 24_000
FRAME_RATE = 75

# Sampling settings of MusicGen (Copet et al., 2023).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 250


# ── Control tracks ──

def check_control_range(name, values):
    """Training conditions lie in [0, 1]; refuse anything else."""
    values = torch.as_tensor(values)
    minimum = float(values.min())
    maximum = float(values.max())

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1] like the training conditions, "
            f"got min={minimum:.3f}, max={maximum:.3f}."
        )


def constant_control(value, total_frames):
    return torch.full((total_frames,), float(value), dtype=torch.float32)


def ramp_control(start_value, end_value, total_frames, ramp_start_frame=0):
    """Constant start_value until ramp_start_frame, then a linear ramp."""
    control = torch.full((total_frames,), float(start_value), dtype=torch.float32)
    ramp_frames = total_frames - ramp_start_frame
    control[ramp_start_frame:] = torch.linspace(
        float(start_value), float(end_value), ramp_frames
    )
    return control


def step_control(low_value, high_value, total_frames, step_frame):
    """low_value before step_frame, high_value from step_frame on."""
    control = torch.full((total_frames,), float(low_value), dtype=torch.float32)
    control[step_frame:] = float(high_value)
    return control


def with_prompt_controls(control, prompt_control):
    """
    Replace the first frames of a requested control track with the
    controls stored for the prompt, so that the prompt codes and their
    conditions agree (as they always did in training).
    """
    control = torch.as_tensor(control, dtype=torch.float32).clone()
    prompt_control = torch.as_tensor(prompt_control, dtype=torch.float32)
    control[:prompt_control.shape[-1]] = prompt_control
    return control


# ── Conditions ──

def build_generation_conditions(tension, combat_score, num_codebooks=NUM_CODEBOOKS):
    """
    Conditions for every delayed input position, built exactly as in
    training.

    tension, combat_score: [B, T] in [0, 1]
    Returns a dict with
        input_tension, input_combat:   [B, T + Q - 2]
        target_tension, target_combat: [B, Q, T + Q - 2]
    Position j of these tensors conditions the prediction of delayed
    position j + 1.
    """
    delayed_tension, delayed_combat, _ = delay_conditions_by_codebook(
        tension, combat_score, num_codebooks
    )

    target_tension = delayed_tension[:, :, 1:]
    target_combat = delayed_combat[:, :, 1:]

    return {
        "input_tension": target_tension[:, 0, :],
        "input_combat": target_combat[:, 0, :],
        "target_tension": target_tension,
        "target_combat": target_combat,
    }


# ── Sampling ──

def gumbel_noise(shape, device, generator=None):
    uniform = torch.rand(shape, device=device, generator=generator)
    uniform = uniform.clamp(min=1e-10, max=1.0 - 1e-7)
    return -torch.log(-torch.log(uniform))


def sample_top_k(logits, temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, noise=None, generator=None):
    """
    logits: [..., V] -> sampled IDs [...]. The PAD ID is never sampled.

    Sampling uses the Gumbel-max trick: argmax(logits / T + Gumbel noise)
    is an exact sample from softmax(logits / T) restricted to the top-k
    IDs. Its advantage over torch.multinomial is that the random numbers
    are explicit: `noise` can be shared by several batch elements (see
    generate_codes, shared_noise=True).

    temperature: a number (0 = greedy), or one value > 0 per codebook for
    logits [..., Q, V] (e.g. a lower temperature for Q1-Q7 than for Q0).
    """
    logits = logits.float().clone()
    logits[..., PAD_TOKEN_ID] = float("-inf")

    if isinstance(temperature, (int, float)):
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
    else:
        temperature = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
        if temperature.ndim != 1 or temperature.shape[0] != logits.shape[-2] or bool((temperature <= 0).any()):
            raise ValueError("A per-codebook temperature needs one value > 0 per codebook.")
        logits = logits / temperature[:, None]

    if top_k is not None and top_k < logits.shape[-1]:
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    if noise is None:
        noise = gumbel_noise(logits.shape, logits.device, generator)

    return (logits + noise).argmax(dim=-1)


# ── Key/value cache ──
#
# SecundaTransformer re-runs the whole prefix at every call. During
# generation the prefix only grows by one position per step, and with a
# causal mask the hidden states of the old positions never change, so the
# keys and values of every layer can be stored and reused: each step then
# processes a single new position. _forward_with_cache recomputes exactly
# the operations of SecundaTransformer.forward (pre-norm encoder layers,
# FiLM output heads) for the new positions only; tests/test_generation.py
# checks that its logits equal those of the full forward pass.

class KeyValueCache:
    """Per-layer keys and values [B, heads, max_sequence_length, head_dim]."""

    def __init__(self, model, batch_size, device, dtype=torch.float32):
        first_attention = model.transformer.layers[0].self_attn
        self.num_heads = first_attention.num_heads
        self.head_dim = model.embedding_dim // self.num_heads
        shape = (batch_size, self.num_heads, model.max_sequence_length, self.head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in model.transformer.layers]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in model.transformer.layers]
        self.length = 0


def _forward_with_cache(model, cache, codes, input_tension, input_combat, target_tension, target_combat):
    """
    Logits [B, Q, n, V] for n new delayed positions (cache.length to
    cache.length + n - 1), given their codes [B, Q, n] and conditions
    (input_* [B, n], target_* [B, Q, n]). Appends their keys and values.
    """
    batch_size, num_codebooks, new_positions = codes.shape
    start = cache.length
    end = start + new_positions
    if end > model.max_sequence_length:
        raise ValueError(f"Position {end} exceeds max_sequence_length={model.max_sequence_length}.")

    positions = torch.arange(start, end, device=codes.device)
    hidden = (
        model.codebook_embedding(codes)
        + model.position_embedding(positions).unsqueeze(0)
        + model.input_condition_mlp(input_tension, input_combat)
    )

    # Causal mask of the new positions over all cached positions.
    if new_positions > 1:
        query_positions = positions.unsqueeze(1)
        key_positions = torch.arange(end, device=codes.device).unsqueeze(0)
        attention_mask = key_positions <= query_positions          # [n, end], True = attend
    else:
        attention_mask = None

    for index, layer in enumerate(model.transformer.layers):
        attention = layer.self_attn

        # Pre-norm self-attention block (norm_first=True).
        normed = layer.norm1(hidden)
        query, key, value = torch.nn.functional.linear(
            normed, attention.in_proj_weight, attention.in_proj_bias
        ).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(batch_size, new_positions, cache.num_heads, cache.head_dim).transpose(1, 2)

        cache.keys[index][:, :, start:end] = split_heads(key)
        cache.values[index][:, :, start:end] = split_heads(value)

        attended = torch.nn.functional.scaled_dot_product_attention(
            split_heads(query),
            cache.keys[index][:, :, :end],
            cache.values[index][:, :, :end],
            attn_mask=attention_mask,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, new_positions, model.embedding_dim)
        hidden = hidden + attention.out_proj(attended)

        # Pre-norm feed-forward block.
        hidden = hidden + layer.linear2(layer.activation(layer.linear1(layer.norm2(hidden))))

    cache.length = end
    hidden = model.final_norm(hidden)

    # Output heads with per-codebook FiLM, as in SecundaTransformer.forward.
    codebook_logits = []
    for codebook_index in range(num_codebooks):
        gamma, beta = model.output_condition_mlp(
            target_tension[:, codebook_index, :], target_combat[:, codebook_index, :]
        ).chunk(2, dim=-1)
        codebook_logits.append(model.output_heads[codebook_index]((1.0 + gamma) * hidden + beta))

    return torch.stack(codebook_logits, dim=1)


# ── Generation loop ──

def _check_inputs(model, prompt_codes, tension, combat_score):
    if prompt_codes.ndim != 3 or prompt_codes.shape[1] != NUM_CODEBOOKS:
        raise ValueError(
            f"prompt_codes must have shape [B, {NUM_CODEBOOKS}, P], "
            f"got {tuple(prompt_codes.shape)}."
        )

    batch_size, _, prompt_frames = prompt_codes.shape

    if prompt_frames < 1:
        raise ValueError(
            "At least one prompt frame is required: delayed position 0 "
            "is never a training target."
        )

    if int(prompt_codes.min()) < 0 or int(prompt_codes.max()) >= CODEBOOK_SIZE:
        raise ValueError("prompt_codes must contain EnCodec IDs in [0, 1023].")

    if tension.shape != combat_score.shape or tension.ndim != 2:
        raise ValueError(
            "tension and combat_score must both have shape [B, T], got "
            f"{tuple(tension.shape)} and {tuple(combat_score.shape)}."
        )

    if tension.shape[0] != batch_size:
        raise ValueError("Controls and prompt_codes have different batch sizes.")

    total_frames = tension.shape[1]

    if total_frames <= prompt_frames:
        raise ValueError(
            f"total_frames={total_frames} must exceed prompt_frames={prompt_frames}."
        )

    longest_input = total_frames + NUM_CODEBOOKS - 2
    if longest_input > model.max_sequence_length:
        raise ValueError(
            f"total_frames={total_frames} needs {longest_input} positions, "
            f"the model has max_sequence_length={model.max_sequence_length} "
            f"(at most {model.max_sequence_length - NUM_CODEBOOKS + 2} frames)."
        )

    check_control_range("tension", tension)
    check_control_range("combat_score", combat_score)

    return batch_size, prompt_frames, total_frames


def _run_delayed_loop(model, prompt_codes, tension, combat_score, choose_tokens, use_cache=True):
    """
    Shared loop of generate_codes and of the tests.

    choose_tokens(step, logits, frame_index, to_sample) returns the IDs to
    write at delayed position `step`:
        logits:      [B, Q, V] predicted for delayed position `step`
        frame_index: [Q] original frame of each codebook at that position
        to_sample:   [Q] bool, True where the cell must be generated
    With use_cache=True each step processes only the newest position
    (KeyValueCache); with use_cache=False the whole prefix is re-run, as
    SecundaTransformer.forward does. Both give the same logits.
    Returns the filled delayed buffer [B, Q, T + Q - 1].
    """
    if model.training:
        raise RuntimeError("Call model.eval() before generating (dropout is on).")

    batch_size, prompt_frames, total_frames = _check_inputs(
        model, prompt_codes, tension, combat_score
    )

    device = prompt_codes.device
    num_codebooks = NUM_CODEBOOKS
    delayed_time = total_frames + num_codebooks - 1

    # Delayed buffer: PAD everywhere, prompt codes at their delayed positions.
    delayed_codes = torch.full(
        (batch_size, num_codebooks, delayed_time),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    for q in range(num_codebooks):
        delayed_codes[:, q, q:q + prompt_frames] = prompt_codes[:, q, :]

    conditions = build_generation_conditions(
        tension.to(device).float(), combat_score.to(device).float(), num_codebooks
    )

    codebook_offsets = torch.arange(num_codebooks, device=device)
    cache = KeyValueCache(model, batch_size, device) if use_cache else None

    # The first cell to generate is frame prompt_frames of codebook 0, at
    # delayed position prompt_frames; the last one is frame T - 1 of
    # codebook Q - 1, at delayed position T + Q - 2.
    for step in range(prompt_frames, delayed_time):
        frame_index = step - codebook_offsets
        to_sample = (frame_index >= prompt_frames) & (frame_index < total_frames)

        if cache is None:
            logits = model(
                input_codes=delayed_codes[:, :, :step],
                input_tension=conditions["input_tension"][:, :step],
                input_combat=conditions["input_combat"][:, :step],
                target_tension=conditions["target_tension"][:, :, :step],
                target_combat=conditions["target_combat"][:, :, :step],
            )[:, :, -1, :]
        else:
            # First step: all prompt positions at once; then one new
            # position (the column written at the previous step) per step.
            new = slice(cache.length, step)
            logits = _forward_with_cache(
                model,
                cache,
                codes=delayed_codes[:, :, new],
                input_tension=conditions["input_tension"][:, new],
                input_combat=conditions["input_combat"][:, new],
                target_tension=conditions["target_tension"][:, :, new],
                target_combat=conditions["target_combat"][:, :, new],
            )[:, :, -1, :]

        chosen = choose_tokens(step, logits, frame_index, to_sample)
        delayed_codes[:, to_sample, step] = chosen[:, to_sample]

    return delayed_codes


@torch.inference_mode()
def generate_codes(
    model,
    prompt_codes,
    tension,
    combat_score,
    temperature=DEFAULT_TEMPERATURE,
    top_k=DEFAULT_TOP_K,
    seed=None,
    shared_noise=False,
    use_cache=True,
):
    """
    Generate codes for a whole segment.

    prompt_codes: [B, Q, P] real EnCodec codes (Q0-Q7) of the first P frames
    tension, combat_score: [B, T] control tracks in [0, 1] for all T frames
        (the first P values should be the prompt's own controls, see
        with_prompt_controls)
    temperature: a number, or Q values (one per codebook, see sample_top_k).
    seed: makes sampling reproducible.
    shared_noise: if True, every batch element uses the same random
        numbers at every step ("common random numbers"). With the same
        prompt, two elements then differ only because their controls
        differ, and an element gives the same result whether it is
        generated alone or in a batch with the same seed. This is the
        paired design of the experiments. If False, the batch elements
        are independent samples.
    use_cache: reuse the keys and values of past positions (KeyValueCache).
        Same logits up to floating-point rounding, much faster. Notebooks
        03 and 05 were run before the cache existed (use_cache=False).

    Returns codes [B, Q, T]; the first P frames equal the prompt.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=prompt_codes.device)
        generator.manual_seed(int(seed))

    def choose_tokens(step, logits, frame_index, to_sample):
        batch_size, num_codebooks, vocab_size = logits.shape
        noise_batch = 1 if shared_noise else batch_size
        noise = gumbel_noise(
            (noise_batch, num_codebooks, vocab_size), logits.device, generator
        )
        return sample_top_k(
            logits,
            temperature=temperature,
            top_k=top_k,
            noise=noise,
        )

    delayed_codes = _run_delayed_loop(
        model, prompt_codes, tension, combat_score, choose_tokens, use_cache=use_cache
    )

    return CodebookSerializer().undelay(delayed_codes)


@torch.inference_mode()
def generate_long_codes(
    model,
    prompt_codes,
    tension,
    combat_score,
    context_frames=375,
    seed=None,
    **sampling,
):
    """
    Generate more frames than one window (750 frames = 10 s) by sliding.

    The first window starts from the prompt; every following window starts
    from the last `context_frames` frames already generated (5 s by
    default) and adds window - context_frames new frames. Each window is
    an ordinary generate_codes call, so the conditions are built exactly
    as in training; window k uses seed + k.

    tension, combat_score: [B, T] for the whole output (any T).
    Returns codes [B, Q, T]; the first P frames equal the prompt. With
    T <= 750 the result is the same as generate_codes.
    """
    window = model.max_sequence_length - NUM_CODEBOOKS + 2
    total_frames = tension.shape[1]

    if not 0 < context_frames < window:
        raise ValueError(f"context_frames must lie in (0, {window}).")

    codes = prompt_codes
    window_index = 0

    while codes.shape[-1] < total_frames:
        available = codes.shape[-1]
        context = available if window_index == 0 else min(context_frames, available)
        start = available - context
        end = min(start + window, total_frames)

        window_codes = generate_codes(
            model,
            codes[:, :, start:],
            tension[:, start:end],
            combat_score[:, start:end],
            seed=None if seed is None else seed + window_index,
            **sampling,
        )
        codes = torch.cat([codes, window_codes[:, :, context:]], dim=-1)
        window_index += 1

    return codes


# ── Prompt and decoding helpers ──

def load_prompt(tensor_path, start_frame, prompt_frames, num_codebooks=NUM_CODEBOOKS):
    """
    Read a prompt from a tokenised track (.pt file produced by the
    tokenizer). Returns (codes [1, Q, P], tension [P], combat_score [P]).
    """
    data = torch.load(tensor_path, map_location="cpu", weights_only=False)
    end_frame = start_frame + prompt_frames

    if end_frame > data["tokens"].shape[-1]:
        raise ValueError(
            f"{tensor_path} has {data['tokens'].shape[-1]} frames, "
            f"requested frames {start_frame}-{end_frame}."
        )

    codes = data["tokens"][:num_codebooks, start_frame:end_frame].long().unsqueeze(0)
    tension = data["tension"][start_frame:end_frame].float()
    combat_score = data["combat_score"][start_frame:end_frame].float()

    return codes, tension, combat_score


def load_codec(device):
    from encodec import EncodecModel

    codec = EncodecModel.encodec_model_24khz().to(device)
    # 6 kbps = 8 codebooks. Decoding uses as many codebooks as provided,
    # this only documents the operating point.
    codec.set_target_bandwidth(6.0)
    codec.eval()
    return codec


@torch.inference_mode()
def decode_codes(codec, codes):
    """
    codes [B, Q, T] -> waveform [B, N] at 24 kHz.

    No peak normalisation: tension is defined on the signal level, so the
    level produced by the model must be kept for the measurements.
    """
    codes = codes.to(next(codec.parameters()).device).long()
    waveform = codec.decode([(codes, None)])
    return waveform[:, 0, :].float().cpu()


def save_wav(path, waveform):
    """Write a mono waveform as 16-bit WAV. Returns the fraction of clipped samples."""
    import soundfile

    audio = np.asarray(torch.as_tensor(waveform).cpu(), dtype=np.float32)
    clipped_fraction = float(np.mean(np.abs(audio) > 1.0))
    soundfile.write(str(path), np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    return clipped_fraction


def real_time_factor(generation_seconds, generated_frames, batch_size=1):
    """Generation time divided by the duration of the generated audio (<1 = faster than real time)."""
    audio_seconds = generated_frames / FRAME_RATE * batch_size
    return generation_seconds / audio_seconds


class Timer:
    """with Timer() as timer: ...; timer.seconds (synchronises CUDA)."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        return False
