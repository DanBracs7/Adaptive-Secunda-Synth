from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_dataset import SecundaAudioDataset
from frame_codec import (
    CODEBOOK_SIZE,
    MODEL_VOCAB_SIZE,
    NUM_CODEBOOKS,
    PAD_TOKEN_ID,
    CodebookSerializer,
    FrameCodebookEmbedding,
    multi_codebook_cross_entropy,
    prepare_next_frame_batch,
)


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
    "skyrim": (
        PROJECT_ROOT /
        "datasets" /
        "skyrim"
    ),
    "witcher3": (
        PROJECT_ROOT /
        "datasets" /
        "witcher3"
    ),
}

BATCH_SIZE = 2
SEGMENT_SECONDS = 5.0
FRAME_RATE = 75
EMBEDDING_DIM = 128


def get_loader_batch():
    dataset = SecundaAudioDataset(
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

    return next(iter(loader))


def test_delay_round_trip(original_codes, serializer):
    delayed_codes = serializer.delay(original_codes)
    recovered_codes = serializer.undelay(delayed_codes)

    original_time = original_codes.shape[-1]

    expected_delayed_time = (
        original_time +
        NUM_CODEBOOKS -
        1
    )

    assert delayed_codes.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        expected_delayed_time,
    )

    assert torch.equal(
        recovered_codes,
        original_codes,
    )

    assert torch.equal(
        delayed_codes[:, 0, :original_time],
        original_codes[:, 0, :],
    )

    assert torch.equal(
        delayed_codes[:, 1, 1:1 + original_time],
        original_codes[:, 1, :],
    )

    assert torch.equal(
        delayed_codes[:, 31, 31:31 + original_time],
        original_codes[:, 31, :],
    )

    assert torch.all(
        delayed_codes[:, 1, 0] == PAD_TOKEN_ID
    )

    assert torch.all(
        delayed_codes[:, 31, :31] == PAD_TOKEN_ID
    )

    print("Delay-pattern round-trip test passed.")
    print("original codes shape:", tuple(original_codes.shape))
    print("delayed codes shape:", tuple(delayed_codes.shape))


def test_prepared_batch(batch, serializer):
    prepared = prepare_next_frame_batch(
        batch=batch,
        serializer=serializer,
    )

    original_time = int(
        SEGMENT_SECONDS * FRAME_RATE
    )

    delayed_time = (
        original_time +
        NUM_CODEBOOKS -
        1
    )

    model_time = delayed_time - 1

    input_codes = prepared["input_codes"]
    target_codes = prepared["target_codes"]

    target_tension = prepared["target_tension"]
    target_combat = prepared["target_combat"]
    target_mask = prepared["target_mask"]

    assert input_codes.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        model_time,
    )

    assert target_codes.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        model_time,
    )

    assert target_tension.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        model_time,
    )

    assert target_combat.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        model_time,
    )

    assert target_mask.shape == (
        BATCH_SIZE,
        NUM_CODEBOOKS,
        model_time,
    )

    assert torch.equal(
    input_codes,
    serializer.delay(batch["tokens"])[:, :, :-1],
)

    assert torch.equal(
        target_codes,
        serializer.delay(batch["tokens"])[:, :, 1:],
    )

    assert torch.equal(
        target_mask,
        target_codes != PAD_TOKEN_ID,
    )

    assert int(input_codes.min()) >= 0
    assert int(input_codes.max()) <= PAD_TOKEN_ID

    assert int(target_codes.min()) >= 0
    assert int(target_codes.max()) <= PAD_TOKEN_ID

    print("\nPrepared delayed-batch test passed.")
    print("input codes shape:", tuple(input_codes.shape))
    print("target codes shape:", tuple(target_codes.shape))
    print(
        "target tension shape:",
        tuple(target_tension.shape),
    )
    print(
        "target combat shape:",
        tuple(target_combat.shape),
    )
    print(
        "valid target positions:",
        int(target_mask.sum()),
    )

    return prepared


def test_frame_embeddings(prepared):
    embedding_layer = FrameCodebookEmbedding(
        embedding_dim=EMBEDDING_DIM,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    )

    frame_embeddings = embedding_layer(
        prepared["input_codes"]
    )

    expected_shape = (
        BATCH_SIZE,
        prepared["input_codes"].shape[-1],
        EMBEDDING_DIM,
    )

    assert tuple(frame_embeddings.shape) == expected_shape
    assert torch.isfinite(frame_embeddings).all()

    print("\nFrame embedding test passed.")
    print(
        "frame embeddings shape:",
        tuple(frame_embeddings.shape),
    )


def test_multi_codebook_loss(prepared):
    test_time_steps = 8

    target_codes = prepared["target_codes"][
        :,
        :,
        :test_time_steps,
    ]

    dummy_logits = torch.randn(
        BATCH_SIZE,
        NUM_CODEBOOKS,
        test_time_steps,
        MODEL_VOCAB_SIZE,
    )

    loss = multi_codebook_cross_entropy(
        logits=dummy_logits,
        target_codes=target_codes,
        pad_token_id=PAD_TOKEN_ID,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("\nMulti-codebook loss test passed.")
    print(f"Dummy cross-entropy loss: {float(loss):.4f}")


def test_codebook_aligned_conditions(
    batch,
    prepared,
):
    original_tension = batch["tension"]
    original_combat = batch["combat_score"]

    target_tension = prepared["target_tension"]
    target_combat = prepared["target_combat"]
    target_mask = prepared["target_mask"]

    original_time = original_tension.shape[-1]
    model_time = target_tension.shape[-1]

    for codebook_index in range(NUM_CODEBOOKS):
        for model_step in range(model_time):
            original_frame = (
                model_step +
                1 -
                codebook_index
            )

            is_valid = (
                0 <= original_frame < original_time
            )

            assert bool(
                target_mask[
                    0,
                    codebook_index,
                    model_step,
                ]
            ) == is_valid

            if is_valid:
                assert torch.equal(
                    target_tension[
                        0,
                        codebook_index,
                        model_step,
                    ],
                    original_tension[
                        0,
                        original_frame,
                    ],
                )

                assert torch.equal(
                    target_combat[
                        0,
                        codebook_index,
                        model_step,
                    ],
                    original_combat[
                        0,
                        original_frame,
                    ],
                )

    print("\nCodebook-aligned conditioning test passed.")
    print(
        "Each real delayed target code has tension/combat "
        "from its matching original audio frame."
    )


def main():
    torch.manual_seed(42)

    batch = get_loader_batch()
    serializer = CodebookSerializer()

    print("Original loader batch:")
    print("tokens shape:", tuple(batch["tokens"].shape))
    print("tension shape:", tuple(batch["tension"].shape))
    print(
        "combat score shape:",
        tuple(batch["combat_score"].shape),
    )

    test_delay_round_trip(
        original_codes=batch["tokens"].long(),
        serializer=serializer,
    )

    prepared = test_prepared_batch(
        batch=batch,
        serializer=serializer,
    )

    test_frame_embeddings(prepared)
    test_multi_codebook_loss(prepared)

    test_codebook_aligned_conditions(
        batch=batch,
        prepared=prepared,
    )

    print("\nAll delayed frame-codec tests passed.")


if __name__ == "__main__":
    main()