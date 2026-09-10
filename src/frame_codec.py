import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1024
PAD_TOKEN_ID = CODEBOOK_SIZE
MODEL_VOCAB_SIZE = CODEBOOK_SIZE + 1



def validate_code_tensor(
    codes,
    num_codebooks=NUM_CODEBOOKS,
    codebook_size=CODEBOOK_SIZE,
):
    if codes.ndim != 3:
        raise ValueError(
            "Expected codes shaped [batch, num_codebooks, time], "
            f"got {tuple(codes.shape)}"
        )

    if codes.shape[1] != num_codebooks:
        raise ValueError(
            f"Expected {num_codebooks} codebooks, "
            f"got {codes.shape[1]}"
        )

    if codes.numel() == 0:
        raise ValueError("Code tensor is empty.")

    minimum = int(codes.min())
    maximum = int(codes.max())

    if minimum < 0 or maximum >= codebook_size:
        raise ValueError(
            f"Code IDs must lie in [0, {codebook_size - 1}], "
            f"but found min={minimum}, max={maximum}."
        )


def validate_condition_tensor(condition, name, batch_size, time_steps):
    if condition.ndim != 2:
        raise ValueError(
            f"{name} must be shaped [batch, time], "
            f"got {tuple(condition.shape)}"
        )

    expected_shape = (batch_size, time_steps)

    if tuple(condition.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"got {tuple(condition.shape)}"
        )


class CodebookSerializer:
    """
    Implements the diagonal delayed-codebook pattern.

    Input:
        codes: [B, Q, T]

    Output:
        delayed_codes: [B, Q, T + Q - 1]

    Codebook q is delayed by q positions:
        delayed_codes[:, q, q:q+T] = codes[:, q, :]

    Empty positions receive PAD_TOKEN_ID = 1024.
    """

    def __init__(
        self,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.pad_token_id = pad_token_id

        if pad_token_id < codebook_size:
            raise ValueError(
                "PAD token must not overlap valid EnCodec IDs."
            )

    @property
    def model_vocab_size(self):
        return self.pad_token_id + 1

    def delay(self, codes):
        validate_code_tensor(
            codes,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
        )

        batch_size, _, time_steps = codes.shape
        delayed_time = time_steps + self.num_codebooks - 1

        delayed_codes = torch.full(
            (
                batch_size,
                self.num_codebooks,
                delayed_time,
            ),
            fill_value=self.pad_token_id,
            dtype=codes.dtype,
            device=codes.device,
        )

        for codebook_index in range(self.num_codebooks):
            start = codebook_index
            end = start + time_steps

            delayed_codes[
                :,
                codebook_index,
                start:end,
            ] = codes[:, codebook_index, :]

        return delayed_codes

    def undelay(self, delayed_codes):
        if delayed_codes.ndim != 3:
            raise ValueError(
                "Expected delayed codes shaped [B, Q, S], "
                f"got {tuple(delayed_codes.shape)}"
            )

        batch_size, found_codebooks, delayed_time = delayed_codes.shape

        if found_codebooks != self.num_codebooks:
            raise ValueError(
                f"Expected {self.num_codebooks} codebooks, "
                f"got {found_codebooks}."
            )

        original_time = delayed_time - self.num_codebooks + 1

        if original_time <= 0:
            raise ValueError(
                "Delayed sequence is too short to recover codes."
            )

        recovered_codes = torch.empty(
            (
                batch_size,
                self.num_codebooks,
                original_time,
            ),
            dtype=delayed_codes.dtype,
            device=delayed_codes.device,
        )

        for codebook_index in range(self.num_codebooks):
            start = codebook_index
            end = start + original_time

            recovered_codes[
                :,
                codebook_index,
                :,
            ] = delayed_codes[
                :,
                codebook_index,
                start:end,
            ]

        validate_code_tensor(
            recovered_codes,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
        )

        return recovered_codes


def delay_conditions_by_codebook(tension, combat_score, num_codebooks=NUM_CODEBOOKS, pad_value=0.0):
    """
    Aligns time-varying controls to the staggered codebook targets.
    
    Inputs:
        tension, combat_score: [B, T]
        
    Outputs:
        delayed_tension: [B, Q, T + Q - 1]
        delayed_combat:  [B, Q, T + Q - 1]
        valid_mask:      [B, Q, T + Q - 1] (True where real audio exists)
    """
    batch_size, original_time = tension.shape
    delayed_time = original_time + num_codebooks - 1

    # Initialize with a pad value (e.g., 0.0). These positions will be ignored in the loss.
    delayed_tension = torch.full(
        (batch_size, num_codebooks, delayed_time), 
        pad_value, dtype=tension.dtype, device=tension.device
    )
    delayed_combat = torch.full(
        (batch_size, num_codebooks, delayed_time), 
        pad_value, dtype=combat_score.dtype, device=combat_score.device
    )
    valid_mask = torch.zeros(
        (batch_size, num_codebooks, delayed_time), 
        dtype=torch.bool, device=tension.device
    )

    for q in range(num_codebooks):
        start = q
        end = q + original_time
        delayed_tension[:, q, start:end] = tension
        delayed_combat[:, q, start:end] = combat_score
        valid_mask[:, q, start:end] = True

    return delayed_tension, delayed_combat, valid_mask


def prepare_next_frame_batch(batch, serializer=None):
    """
    Updated to return codebook-aligned conditions.
    """
    if serializer is None:
        serializer = CodebookSerializer()

    tokens = batch["tokens"].long()
    tension = batch["tension"].float()
    combat_score = batch["combat_score"].float()

    delayed_codes = serializer.delay(tokens)
    delayed_tension, delayed_combat, valid_mask = delay_conditions_by_codebook(
        tension, combat_score, serializer.num_codebooks
    )

    # Autoregressive shift: inputs get 0 to end-1, targets get 1 to end
    input_codes = delayed_codes[:, :, :-1]
    target_codes = delayed_codes[:, :, 1:]
    
    # Conditions align with the TARGET frame, so we slice them 1 to end as well
    target_tension = delayed_tension[:, :, 1:]
    target_combat = delayed_combat[:, :, 1:]
    target_mask = valid_mask[:, :, 1:]

    return {
        "input_codes": input_codes,
        "target_codes": target_codes,
        "target_tension": target_tension,
        "target_combat": target_combat,
        "target_mask": target_mask
    }

class FrameCodebookEmbedding(nn.Module):
    """
    Converts delayed [B, Q, S] IDs into [B, S, D] frame representations.

    PAD_TOKEN_ID has a zero embedding and does not contribute to the
    summed frame representation.
    """

    def __init__(
        self,
        embedding_dim,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        pad_token_id=PAD_TOKEN_ID,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.pad_token_id = pad_token_id
        self.model_vocab_size = pad_token_id + 1

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=self.model_vocab_size,
                    embedding_dim=embedding_dim,
                    padding_idx=pad_token_id,
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, delayed_codes):
        if delayed_codes.ndim != 3:
            raise ValueError(
                "Expected delayed codes shaped [B, Q, S], "
                f"got {tuple(delayed_codes.shape)}"
            )

        if delayed_codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"Expected {self.num_codebooks} codebooks, "
                f"got {delayed_codes.shape[1]}."
            )

        minimum = int(delayed_codes.min())
        maximum = int(delayed_codes.max())

        if minimum < 0 or maximum > self.pad_token_id:
            raise ValueError(
                f"Delayed IDs must lie in [0, {self.pad_token_id}], "
                f"but found min={minimum}, max={maximum}."
            )

        code_embeddings = []

        for codebook_index, embedding in enumerate(self.embeddings):
            code_ids = delayed_codes[:, codebook_index, :]
            code_embeddings.append(embedding(code_ids))

        stacked = torch.stack(
            code_embeddings,
            dim=2,
        )

        frame_embeddings = stacked.sum(dim=2)
        frame_embeddings = frame_embeddings / (
            self.num_codebooks ** 0.5
        )

        return frame_embeddings


def multicodebook_cross_entropy_weighted(
    logits: torch.Tensor,
    target_codes: torch.Tensor,
    pad_token_id: int,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted multi-codebook cross-entropy.

    logits: B, Q, S, V (V = CODEBOOK_SIZE+1)
    target_codes: B, Q, S, with EnCodec IDs [0..CODEBOOK_SIZE-1] or PAD_TOKEN_ID.
    pad_token_id: ID to ignore in loss.
    loss_weights: Q, weights per codebook.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shaped (B,Q,S,V), got {logits.shape}")
    if target_codes.ndim != 3:
        raise ValueError(f"Expected targets shaped (B,Q,S), got {target_codes.shape}")
    if logits.shape[0] != target_codes.shape[0] or logits.shape[1] != target_codes.shape[1] or logits.shape[2] != target_codes.shape[2]:
        raise ValueError(f"logits and targets must agree on B,Q,S: {logits.shape} vs {target_codes.shape}")
    if logits.shape[1] != loss_weights.shape[0]:
        raise ValueError(f"loss_weights must have length Q={logits.shape[1]}, got {loss_weights.shape[0]}")

    B, Q, S, V = logits.shape

    # Flatten over time per codebook: treat each (q, t) as a token position.
    logits_flat = logits.reshape(B * Q * S, V)
    targets_flat = target_codes.reshape(B * Q * S)

    # Mask out PAD positions
    non_pad_mask = (targets_flat != pad_token_id)
    if non_pad_mask.sum() == 0:
        # No valid positions anywhere -> zero loss
        return logits.new_zeros(())

    logits_valid = logits_flat[non_pad_mask]
    targets_valid = targets_flat[non_pad_mask]

    # Cross-entropy over valid positions
    ce_per_pos = F.cross_entropy(
        logits_valid,
        targets_valid,
        reduction="none",
    )

    # Map positions back to (q, t) to apply per-codebook weights
    pos_idx = torch.nonzero(non_pad_mask, as_tuple=False).squeeze(1)
    q_idx = (pos_idx // S) % Q  # since we flattened as B,Q,S

    # Apply weights: each position gets the weight of its codebook q
    weights_per_pos = loss_weights[q_idx]

    weighted_ce = ce_per_pos * weights_per_pos
    # Normalize by sum of weights to keep the scale reasonable
    loss = weighted_ce.sum() / weights_per_pos.sum()
    return loss