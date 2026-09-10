#!pip install -q encodec
#NOTE this script is designed to run in a Kaggle notebook environment. It generates audio from scratch using a fine-tuned Secunda model on the Slakh dataset, with various conditioning profiles for tension and combat. The generated audio is saved as WAV files and displayed for playback.
#It is not ready to run in a local environment without modifications, as it relies on specific paths and dependencies available in the Kaggle environment.
#It is also for notebook execution, not for direct script execution. It is intended to be run in a Kaggle notebook cell.



import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io.wavfile import write as wav_write
from encodec import EncodecModel

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ──
CHECKPOINT_PATH = Path(
    "/kaggle/input/models/danielebracoloni/secunda-finetuned-on-ost/pytorch/default/1/secunda_slakh8cb_finetune_best.pt"
)
PROMPT_TENSOR_PATH = Path(
    "/kaggle/input/datasets/danielebracoloni/slakh-tensors-final/train/train_367.pt"
)

# ── Model constants (must match training) ──
NUM_CODEBOOKS_MODEL = 8
NUM_CODEBOOKS_CODEC = 32
CODEBOOK_SIZE = 1024
PAD_TOKEN_ID = CODEBOOK_SIZE
MODEL_VOCAB_SIZE = CODEBOOK_SIZE + 1
FRAME_RATE = 75
SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * FRAME_RATE)  # 750
MAX_SEQUENCE_LENGTH = 1024
SAMPLE_RATE = 24_000


# ── Conditioning profiles (duration = SEGMENT_SECONDS) ──

def apply_jitter(base_val, t, noise_std=0.5):
    # Creates a tensor of base_val and adds frame-by-frame Gaussian noise
    base_tensor = torch.full_like(t, base_val)
    noise = torch.randn_like(t) * noise_std
    # Clamp to ensure scores don't drop below 0
    return torch.clamp(base_tensor + noise, min=0.0)

def profile_low(t):
    tension_base = random.uniform(0.0, 4.0)
    combat_base = random.uniform(0.0, 4.0)
    return apply_jitter(tension_base, t), apply_jitter(combat_base, t)

def profile_high(t):
    tension_base = random.uniform(6.0, 9.0)
    combat_base = random.uniform(6.0, 9.0)
    return apply_jitter(tension_base, t), apply_jitter(combat_base, t)

def profile_ramp(t):
    T = t.shape[0]
    alpha = torch.linspace(0.0, 1.0, T, device=t.device)
    
    # Ramp from a random low base to a random high base
    tension_low = random.uniform(0.0, 3.0)
    tension_high = random.uniform(7.0, 9.0)
    combat_low = random.uniform(0.0, 3.0)
    combat_high = random.uniform(7.0, 9.0)
    
    tension = tension_low + (tension_high - tension_low) * alpha
    combat  = combat_low  + (combat_high  - combat_low)  * alpha
    
    # Add frame-by-frame noise on top of the ramp
    noise_t = torch.randn_like(t) * 0.5
    noise_c = torch.randn_like(t) * 0.5
    
    return torch.clamp(tension + noise_t, min=0.0), torch.clamp(combat + noise_c, min=0.0)

def profile_lowT_highC(t):
    tension_base = random.uniform(0.0, 4.0)
    combat_base = random.uniform(6.0, 9.0)
    return apply_jitter(tension_base, t), apply_jitter(combat_base, t)

def profile_highT_lowC(t):
    tension_base = random.uniform(6.0, 9.0)
    combat_base = random.uniform(0.0, 4.0)
    return apply_jitter(tension_base, t), apply_jitter(combat_base, t)

def profile_middle(t):
    tension_base = random.uniform(4.0, 6.0)
    combat_base = random.uniform(4.0, 6.0)
    return apply_jitter(tension_base, t), apply_jitter(combat_base, t)

def profile_low_tension_no_combat(t):
    tension_base = random.uniform(0.0, 4.0)
    combat = torch.zeros_like(t)
    return apply_jitter(tension_base, t), combat

def profile_high_tension_no_combat(t):
    tension_base = random.uniform(6.0, 9.0)
    combat = torch.zeros_like(t)
    return apply_jitter(tension_base, t), combat

def profile_no_tension_low_combat(t):
    tension = torch.zeros_like(t)
    combat_base = random.uniform(0.0, 4.0)
    return tension, apply_jitter(combat_base, t)

def profile_no_tension_high_combat(t):
    tension = torch.zeros_like(t)
    combat_base = random.uniform(6.0, 9.0)
    return tension, apply_jitter(combat_base, t)

CONDITION_PROFILES = {
    "low": profile_low,
    "high": profile_high,
    "ramp": profile_ramp,
    "lowT_highC": profile_lowT_highC,
    "highT_lowC": profile_highT_lowC,
    "middle" : profile_middle,
    "lowT_noC": profile_low_tension_no_combat,
    "highT_noC": profile_high_tension_no_combat,
    "noT_lowC": profile_no_tension_low_combat,
    "noT_highC": profile_no_tension_high_combat,
}


# ── Model definition (same as training, renamed layers) ──
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

# ── Load checkpoint ──
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
cfg = checkpoint.get("config", {})
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Best validation loss:", checkpoint.get("best_val_loss"))

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
    numcodebooks=NUM_CODEBOOKS_MODEL,
    codebooksize=CODEBOOK_SIZE,
    padtokenid=PAD_TOKEN_ID,
).to(device)

state = checkpoint["model_state_dict"]
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

# ── Codec ──
codec = EncodecModel.encodec_model_24khz().to(device)
# Setting bandwidth to 6.0 forces Encodec to expect exactly 8 codebooks,
# allowing pure from-scratch decoding without the 32cb residual trick.
codec.set_target_bandwidth(6.0) 
codec.eval()

# ── Helper: delay pattern (same as training) ──
def delay_codes(codes):
    B, Q, T = codes.shape
    out = torch.full(
        (B, Q, T + Q - 1), PAD_TOKEN_ID,
        dtype=codes.dtype, device=codes.device
    )
    for q in range(Q):
        out[:, q, q:q+T] = codes[:, q, :]
    return out

# ── Empty variables for pure from-scratch generation ──
PROMPT_FRAMES = 225 
prompt_tokens_8cb = torch.empty((NUM_CODEBOOKS_MODEL, 0), dtype=torch.long)
prompt_tension    = torch.empty((0,), dtype=torch.float32)
prompt_combat     = torch.empty((0,), dtype=torch.float32)

# ── Autoregressive from-scratch generation ──
def generate_from_scratch(
    profile_name,
    total_frames=SEGMENT_FRAMES,
    prompt_frames=prompt_tokens_8cb,
    prompt_tens=prompt_tension,
    prompt_comb=prompt_combat,
    temperature=1.0,
    top_k=250,
):
    profile_fn = CONDITION_PROFILES[profile_name]

    t = torch.arange(total_frames + 1, device=device).float()
    tension_full, combat_full = profile_fn(t) 

    P = prompt_frames.shape[-1]
    assert P <= total_frames

    tokens_8cb = prompt_frames.clone().to(device)

    for cur_len in range(P, total_frames):
        L = cur_len
        codes_sofar = tokens_8cb[:, :L].unsqueeze(0) 
        delayed = delay_codes(codes_sofar)           
        input_codes = delayed[:, :, :-1]             
        S = input_codes.shape[-1]
    
        if S > total_frames:
            input_codes = input_codes[:, :, :total_frames]
            S = total_frames
        
        input_tension = tension_full[:S].unsqueeze(0)        
        input_combat  = combat_full[:S].unsqueeze(0)         
        
        target_tension = (
            tension_full[1:S+1]
            .unsqueeze(0)            
            .unsqueeze(1)            
            .expand(-1, NUM_CODEBOOKS_MODEL, -1)  
        )
        target_combat = (
            combat_full[1:S+1]
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(-1, NUM_CODEBOOKS_MODEL, -1)
        )
        
        with torch.inference_mode():
            logits = model(
                input_codes,
                input_tension,
                input_combat,
                target_tension,
                target_combat,
            ) 

        next_logits = logits[:, :, -1, :].squeeze(0).clone() 

        if temperature != 1.0:
            next_logits = next_logits / temperature

        # CRITICAL FIX: Forbid the generation of the PAD token before sampling
        next_logits[:, PAD_TOKEN_ID] = float('-inf')

        # Top-K filtering to eliminate garbage tokens while maintaining diversity
        top_v, _ = torch.topk(next_logits, top_k, dim=-1)
        min_top_v = top_v[:, -1].unsqueeze(-1)
        
        next_logits = torch.where(
            next_logits < min_top_v, 
            torch.full_like(next_logits, float('-inf')), 
            next_logits
        )

        probs = F.softmax(next_logits, dim=-1) 

        # Sample stochastically
        next_tokens = torch.multinomial(probs, 1).squeeze(-1) 

        tokens_8cb = torch.cat(
            [tokens_8cb, next_tokens.unsqueeze(-1)], dim=-1
        ) 

    return tokens_8cb


# ── Decode directly with exactly 8 codebooks ──
def decode_8cb(tokens_8cb, path):
    tokens_8cb = tokens_8cb.unsqueeze(0).to(device)

    with torch.inference_mode():
        decoded = codec.decode([(tokens_8cb, None)])

    audio = decoded[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.98

    wav_write(path, SAMPLE_RATE, audio)
    return audio

# ── Generate and save for each profile ──
from IPython.display import Audio, display

for profile_name in CONDITION_PROFILES.keys():
    print(f"\nGenerating from scratch with profile: {profile_name}")

    tokens_gen = generate_from_scratch(
        profile_name,
        total_frames=SEGMENT_FRAMES,
        prompt_frames=prompt_tokens_8cb,
        prompt_tens=prompt_tension,
        prompt_comb=prompt_combat,
        temperature=1.0,
        top_k=50, # You can adjust this between 50 and 250 to test the variance
    )

    out_path = f"/kaggle/working/gen_scratch_{profile_name}.wav"
    audio = decode_8cb(tokens_gen, out_path)

    print(f"Saved: {out_path}, audio shape: {audio.shape}")
    display(Audio(out_path))