

!pip install -q encodec

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from encodec import EncodecModel
from scipy.io.wavfile import write as wav_write

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change only this checkpoint dataset path if your Kaggle slug differs.
MANIFEST_PATH = Path(
    "/kaggle/input/datasets/danielebracoloni/"
    "slakh-and-custom-csvs/all_tracks.csv"
)
SLAKH_ROOT = Path(
    "/kaggle/input/datasets/danielebracoloni/"
    "tesors-slakh-375-checkpoint/tensors_final/train"
)
CHECKPOINT_PATH = Path(
    "/kaggle/input/datasets/danielebracoloni/secunda-model-v2/secunda_slakh_8cb_448d_14l/secunda_slakh8cb_best_v2.pt"
)

NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1024
PAD_TOKEN_ID = CODEBOOK_SIZE
MODEL_VOCAB_SIZE = CODEBOOK_SIZE + 1
FRAME_RATE = 75
SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * FRAME_RATE)
MAX_SEQUENCE_LENGTH = 1024
SAMPLE_RATE = 24_000

assert MANIFEST_PATH.exists(), MANIFEST_PATH
assert SLAKH_ROOT.exists(), SLAKH_ROOT
assert CHECKPOINT_PATH.exists(), CHECKPOINT_PATH

class SecundaAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        source_roots,
        split="train",
        stage=None,
        segment_seconds=5.0,
        frame_rate=75,
        fixed_start=False,
        seed=42,
        samples_per_track=1,
    ):
        self.manifest_path = Path(manifest_path)

        self.source_roots = {
            key: Path(value)
            for key, value in source_roots.items()
        }

        self.segment_frames = int(
            round(segment_seconds * frame_rate)
        )

        if self.segment_frames < 2:
            raise ValueError(
                "segment_seconds must produce at least 2 frames."
            )

        if samples_per_track < 1:
            raise ValueError(
                "samples_per_track must be at least 1."
            )

        self.fixed_start = fixed_start
        self.seed = seed
        self.epoch = 0
        self.samples_per_track = samples_per_track

        with open(
            self.manifest_path,
            newline="",
            encoding="utf-8",
        ) as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row
            for row in all_rows
            if row["split"] == split
            and row["split"] != "excluded"
            and (
                stage is None
                or row["stage"] == stage
            )
        ]

        if not self.rows:
            raise ValueError(
                f"No rows found for split={split}, stage={stage}"
            )

        self._validate_rows()

    def _validate_rows(self):
        for row in self.rows:
            source = row["source"]

            if source not in self.source_roots:
                raise KeyError(
                    f"No root configured for source={source}"
                )

            filename = Path(row["path"]).name
            tensor_path = self.source_roots[source] / filename

            if not tensor_path.exists():
                raise FileNotFoundError(tensor_path)

            if int(row["num_codebooks"]) != 32:
                raise ValueError(
                    f"{tensor_path}: expected 32 codebooks"
                )

            if int(row["frame_rate"]) != 75:
                raise ValueError(
                    f"{tensor_path}: expected 75 fps"
                )

            if float(row["bandwidth_kbps"]) != 24.0:
                raise ValueError(
                    f"{tensor_path}: expected 24 kbps"
                )

            if int(row["num_frames"]) < self.segment_frames:
                raise ValueError(
                    f"{tensor_path}: has only "
                    f"{row['num_frames']} frames, but needs "
                    f"{self.segment_frames}"
                )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows) * self.samples_per_track

    def _path_for_row(self, row):
        source = row["source"]
        filename = Path(row["path"]).name
        return self.source_roots[source] / filename

    def _choose_start(self,row_index,crop_index,total_frames,):
        max_start = total_frames - self.segment_frames
    
        if max_start <= 0:
            return 0
    
        if self.fixed_start:
            # Validation:
            # same interior crop for this validation track every epoch.
            # It is deterministic, but not forced to begin at frame 0.
            generator = random.Random(
                self.seed
                + 99_999_937
                + row_index * 10_000
                + crop_index
            )
        else:
            # Training:
            # deterministic but changes with epoch.
            generator = random.Random(
                self.seed
                + self.epoch * 1_000_000
                + row_index * 10_000
                + crop_index
            )
    
        return generator.randint(0, max_start)



    def __getitem__(self, index):
        row_index = index // self.samples_per_track
        crop_index = index % self.samples_per_track

        row = self.rows[row_index]
        tensor_path = self._path_for_row(row)

        data = torch.load(
            tensor_path,
            map_location="cpu",
        )

        tokens = data["tokens"].long()
        tension = data["tension"].float()
        combat_score = data["combat_score"].float()

        if tokens.ndim != 2:
            raise ValueError(
                f"{tensor_path}: tokens must be [n_q, T]"
            )

        total_frames = tokens.shape[-1]

        if total_frames != len(tension):
            raise ValueError(
                f"{tensor_path}: tokens/tension mismatch"
            )

        if total_frames != len(combat_score):
            raise ValueError(
                f"{tensor_path}: tokens/combat mismatch"
            )

        start = self._choose_start(
            row_index=row_index,
            crop_index=crop_index,
            total_frames=total_frames,
        )

        end = start + self.segment_frames

        return {
            "tokens": tokens[:NUM_CODEBOOKS, start:end],
            "tension": tension[start:end],
            "combat_score": combat_score[start:end],
            "source": row["source"],
            "track_id": row["track_id"],
            "path": row["path"],
            "start_frame": start,
            "crop_index": crop_index,
        }
            

class CodebookSerializer:
    def __init__(self, numcodebooks=8, codebooksize=1024, padtokenid=1024):
        self.numcodebooks = numcodebooks
        self.codebooksize = codebooksize
        self.padtokenid = padtokenid

    def delay(self, codes):
        b, q, t = codes.shape
        if (codes < 0).any() or (codes >= self.codebooksize).any():
            raise ValueError("Invalid source EnCodec IDs")
        out = torch.full(
            (b, q, t + q - 1), self.padtokenid,
            dtype=codes.dtype, device=codes.device
        )
        for codebook in range(q):
            out[:, codebook, codebook:codebook + t] = codes[:, codebook]
        return out

    def undelay(self, delayed):
        b, q, delayed_t = delayed.shape
        t = delayed_t - q + 1
        if t <= 0:
            raise ValueError("Delayed sequence too short")
        out = torch.empty((b, q, t), dtype=delayed.dtype, device=delayed.device)
        for codebook in range(q):
            out[:, codebook] = delayed[:, codebook, codebook:codebook + t]
        if (out < 0).any() or (out >= self.codebooksize).any():
            raise ValueError("Undelayed codes contain PAD/invalid IDs")
        return out

def delay_conditions(tension, combat, num_codebooks=8):
    b, t = tension.shape
    dt = t + num_codebooks - 1
    tension_d = torch.zeros((b, num_codebooks, dt), dtype=tension.dtype)
    combat_d = torch.zeros((b, num_codebooks, dt), dtype=combat.dtype)
    valid = torch.zeros((b, num_codebooks, dt), dtype=torch.bool)
    for q in range(num_codebooks):
        tension_d[:, q, q:q+t] = tension
        combat_d[:, q, q:q+t] = combat
        valid[:, q, q:q+t] = True
    return tension_d, combat_d, valid

def prepare_batch(batch, serializer):
    tokens = batch["tokens"].long()
    tension = batch["tension"].float()
    combat = batch["combat_score"].float()
    delayed = serializer.delay(tokens)
    td, cd, valid = delay_conditions(tension,combat,NUM_CODEBOOKS,)
    return {
        "inputcodes": delayed[:, :, :-1],
        "targetcodes": delayed[:, :, 1:],
        "inputtension": td[:, 0, :-1],
        "inputcombat": cd[:, 0, :-1],
        "targettension": td[:, :, 1:],
        "targetcombat": cd[:, :, 1:],
        "targetmask": valid[:, :, 1:],
    }

class ConditionMLP(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputdim, hiddendim), nn.GELU(), nn.Linear(hiddendim, outputdim)
        )
    def forward(self, tension, combat):
        return self.network(torch.stack([tension, combat], dim=-1))

class FrameCodebookEmbedding(nn.Module):
    def __init__(self, embeddingdim, numcodebooks, codebooksize, padtokenid):
        super().__init__()
        self.numcodebooks = numcodebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebooksize + 1, embeddingdim, padding_idx=padtokenid)
            for _ in range(numcodebooks)
        ])
    def forward(self, codes):
        values = [emb(codes[:, q]) for q, emb in enumerate(self.embeddings)]
        return torch.stack(values, dim=2).sum(dim=2) / math.sqrt(self.numcodebooks)

class SecundaTransformer(nn.Module):
    def __init__(self, embeddingdim, numlayers, numheads, feedforwarddim,
                 dropout, maxsequencelength, numcodebooks, codebooksize, padtokenid):
        super().__init__()
        self.numcodebooks = numcodebooks
        self.codebookembedding = FrameCodebookEmbedding(
            embeddingdim, numcodebooks, codebooksize, padtokenid
        )
        self.positionembedding = nn.Embedding(maxsequencelength, embeddingdim)
        self.inputconditionmlp = ConditionMLP(2, embeddingdim, embeddingdim)
        layer = nn.TransformerEncoderLayer(
            d_model=embeddingdim, nhead=numheads, dim_feedforward=feedforwarddim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=numlayers)
        self.finalnorm = nn.LayerNorm(embeddingdim)
        self.outputconditionmlp = ConditionMLP(2, embeddingdim, 2 * embeddingdim)
        self.outputheads = nn.ModuleList([
            nn.Linear(embeddingdim, codebooksize + 1) for _ in range(numcodebooks)
        ])

    def forward(self, inputcodes, inputtension, inputcombat, targettension, targetcombat):
        b, q, s = inputcodes.shape
        hidden = self.codebookembedding(inputcodes)
        hidden = hidden + self.positionembedding(torch.arange(s, device=inputcodes.device))[None]
        hidden = hidden + self.inputconditionmlp(inputtension, inputcombat)
        mask = torch.triu(torch.full((s, s), float("-inf"), device=inputcodes.device), diagonal=1)
        hidden = self.transformer(hidden, mask=mask)
        hidden = self.finalnorm(hidden)
        outputs = []
        for codebook in range(self.numcodebooks):
            film = self.outputconditionmlp(
                targettension[:, codebook], targetcombat[:, codebook]
            )
            gamma, beta = film.chunk(2, dim=-1)
            outputs.append(self.outputheads[codebook](hidden * (1 + gamma) + beta))
        return torch.stack(outputs, dim=1)

def decode_codes(codes, path):
    """
    codes: [B, Q, T] integer EnCodec IDs, with values 0..1023.
    Decodes only the first item in the batch.
    """
    codes = codes[:1].long().to(device)  # [1, 8, T]

    assert codes.ndim == 3, (
        f"Expected [B, Q, T], got {tuple(codes.shape)}"
    )
    assert codes.shape[1] == NUM_CODEBOOKS, (
        f"Expected {NUM_CODEBOOKS} codebooks, got {codes.shape[1]}"
    )
    assert int(codes.min()) >= 0
    assert int(codes.max()) < CODEBOOK_SIZE

    # EnCodec expects a list of `(codes, scale)` frames.
    # `scale=None` is correct for non-segmented / normal EnCodec decoding.
    encoded_frames = [
        (codes, None),
    ]

    with torch.inference_mode():
        decoded = codec.decode(encoded_frames)

    # Decoded shape is normally [B, channels, samples].
    audio = decoded[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.98

    wav_write(path, SAMPLE_RATE, audio)
    return audio

# Load checkpoint and instantiate from its saved configuration.
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
cfg = checkpoint.get("config", {})
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Best validation loss:", checkpoint.get("best_val_loss"))
print("Checkpoint config:", cfg)

required = {
    "num_codebooks": 8,
    "embedding_dim": 448,
    "num_layers": 14,
    "num_heads": 8,
    "feedforward_dim": 2304,
    "segment_frames": 750,
}
for key, expected in required.items():
    actual = cfg.get(key)
    if actual != expected:
        raise ValueError(f"Config mismatch for {key}: expected {expected}, got {actual}")

model = SecundaTransformer(
    embeddingdim=cfg["embedding_dim"],
    numlayers=cfg["num_layers"],
    numheads=cfg["num_heads"],
    feedforwarddim=cfg["feedforward_dim"],
    dropout=cfg["dropout"],
    maxsequencelength=cfg["max_sequence_length"],
    numcodebooks=8,
    codebooksize=1024,
    padtokenid=1024,
).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
state = checkpoint["model_state_dict"]

# Map old names (with underscores) to your current class names (without underscores).
renamed_state = {}
for k, v in state.items():
    new_k = (
        k.replace("codebook_embedding", "codebookembedding")
         .replace("position_embedding", "positionembedding")
         .replace("input_condition_mlp", "inputconditionmlp")
         .replace("final_norm", "finalnorm")
         .replace("output_condition_mlp", "outputconditionmlp")
         .replace("output_heads", "outputheads")
    )
    renamed_state[new_k] = v

model.load_state_dict(renamed_state, strict=True)
model.eval()

print("Loaded parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
# Held-out Slakh validation crop.
dataset = SecundaAudioDataset(
    MANIFEST_PATH, {"slakh": SLAKH_ROOT}, split="val", stage="pretrain",
    segment_seconds=10.0, frame_rate=75, fixed_start=True, samples_per_track=1
)
NUM_TEST_TRACKS = 3

random.seed(SEED)
sample_indices = random.sample(range(len(dataset)), k=NUM_TEST_TRACKS)

serializer = CodebookSerializer()

from IPython.display import Audio, display

for rank, sample_index in enumerate(sample_indices, start=1):
    item = dataset[sample_index]

    batch = {
        "tokens": item["tokens"].unsqueeze(0),
        "tension": item["tension"].unsqueeze(0),
        "combat_score": item["combat_score"].unsqueeze(0),
    }

    prepared = prepare_batch(batch, serializer)
    prepared = {
        k: v.to(device)
        for k, v in prepared.items()
        if torch.is_tensor(v)
    }

    with torch.inference_mode():
        logits = model(
            prepared["inputcodes"],
            prepared["inputtension"],
            prepared["inputcombat"],
            prepared["targettension"],
            prepared["targetcombat"],
        )

    predicted_delayed = logits.argmax(dim=-1)

    known_final = prepared["targetcodes"][:, :, -1:].clone()
    full_predicted_delayed = torch.cat(
        [predicted_delayed, known_final],
        dim=-1,
    )

    predicted_codes = serializer.undelay(full_predicted_delayed)
    target_codes = batch["tokens"].long().to(device)

    assert predicted_codes.shape == target_codes.shape, (
        f"Shape mismatch: predicted={tuple(predicted_codes.shape)}, "
        f"target={tuple(target_codes.shape)}"
    )

    pred_path = f"/kaggle/working/slakh_predicted_track{rank}.wav"
    true_path = f"/kaggle/working/slakh_ground_truth_track{rank}.wav"

    pred_audio = decode_codes(predicted_codes, pred_path)
    true_audio = decode_codes(target_codes, true_path)

    print(
        f"\nTrack {rank} | "
        f"track_id={item['track_id']} | "
        f"start_frame={item['start_frame']}"
    )
    print("Wrote WAV files:", pred_audio.shape, true_audio.shape)

    print("Ground truth:")
    display(Audio(true_path))

    print("Predicted (teacher-forced):")
    display(Audio(pred_path))


