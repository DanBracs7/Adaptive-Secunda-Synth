import math

import torch
import torch.nn as nn

from frame_codec import (
    CODEBOOK_SIZE,
    MODEL_VOCAB_SIZE,
    NUM_CODEBOOKS,
    PAD_TOKEN_ID,
    FrameCodebookEmbedding,
)


class ConditionMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tension, combat_score):
        if tension.shape != combat_score.shape:
            raise ValueError(
                "tension and combat_score must have identical shapes."
            )

        conditions = torch.stack(
            [tension, combat_score],
            dim=-1,
        )

        return self.network(conditions)


class DelayedCodebookTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=512,
        dropout=0.10,
        max_sequence_length=800,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.pad_token_id = pad_token_id
        self.model_vocab_size = MODEL_VOCAB_SIZE
        self.max_sequence_length = max_sequence_length

        self.codebook_embedding = FrameCodebookEmbedding(
            embedding_dim=embedding_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            pad_token_id=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

        self.input_condition_mlp = ConditionMLP(
            input_dim=2,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        # Produces gamma and beta for each codebook-specific target condition.
        self.output_condition_mlp = ConditionMLP(
            input_dim=2,
            hidden_dim=embedding_dim,
            output_dim=2 * embedding_dim,
        )

        self.output_heads = nn.ModuleList(
            [
                nn.Linear(
                    embedding_dim,
                    self.model_vocab_size,
                )
                for _ in range(num_codebooks)
            ]
        )

    def _causal_mask(self, sequence_length, device):
        mask = torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
        )

        return torch.triu(
            mask,
            diagonal=1,
        )

    def forward(
        self,
        input_codes,
        input_tension,
        input_combat,
        target_tension,
        target_combat,
    ):
        """
        input_codes:
            [B, Q, S]

        input_tension, input_combat:
            [B, S]

        target_tension, target_combat:
            [B, Q, S]

        returns logits:
            [B, Q, S, 1025]
        """
        if input_codes.ndim != 3:
            raise ValueError(
                "input_codes must have shape [B, Q, S]."
            )

        batch_size, found_codebooks, sequence_length = input_codes.shape

        if found_codebooks != self.num_codebooks:
            raise ValueError(
                f"Expected {self.num_codebooks} codebooks, "
                f"got {found_codebooks}."
            )

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_sequence_length={self.max_sequence_length}."
            )

        expected_shared_condition_shape = (
            batch_size,
            sequence_length,
        )

        if input_tension.shape != expected_shared_condition_shape:
            raise ValueError(
                "input_tension must have shape "
                f"{expected_shared_condition_shape}, "
                f"got {tuple(input_tension.shape)}."
            )

        if input_combat.shape != expected_shared_condition_shape:
            raise ValueError(
                "input_combat must have shape "
                f"{expected_shared_condition_shape}, "
                f"got {tuple(input_combat.shape)}."
            )

        expected_target_condition_shape = (
            batch_size,
            self.num_codebooks,
            sequence_length,
        )

        if target_tension.shape != expected_target_condition_shape:
            raise ValueError(
                "target_tension must have shape "
                f"{expected_target_condition_shape}, "
                f"got {tuple(target_tension.shape)}."
            )

        if target_combat.shape != expected_target_condition_shape:
            raise ValueError(
                "target_combat must have shape "
                f"{expected_target_condition_shape}, "
                f"got {tuple(target_combat.shape)}."
            )

        frame_embeddings = self.codebook_embedding(
            input_codes
        )

        positions = torch.arange(
            sequence_length,
            device=input_codes.device,
        )

        positional_embeddings = self.position_embedding(
            positions
        ).unsqueeze(0)

        input_condition_embeddings = self.input_condition_mlp(
            input_tension,
            input_combat,
        )

        hidden = (
            frame_embeddings
            + positional_embeddings
            + input_condition_embeddings
        )

        causal_mask = self._causal_mask(
            sequence_length,
            input_codes.device,
        )

        hidden = self.transformer(
            hidden,
            mask=causal_mask,
        )

        hidden = self.final_norm(hidden)

        codebook_logits = []

        for codebook_index in range(self.num_codebooks):
            codebook_tension = target_tension[
                :,
                codebook_index,
                :,
            ]

            codebook_combat = target_combat[
                :,
                codebook_index,
                :,
            ]

            film_parameters = self.output_condition_mlp(
                codebook_tension,
                codebook_combat,
            )

            gamma, beta = film_parameters.chunk(
                2,
                dim=-1,
            )

            conditioned_hidden = (
                (1.0 + gamma) * hidden
                + beta
            )

            logits = self.output_heads[codebook_index](
                conditioned_hidden
            )

            codebook_logits.append(logits)

        return torch.stack(
            codebook_logits,
            dim=1,
        )