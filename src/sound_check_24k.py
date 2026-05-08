import torch
from pathlib import Path
from encodec import EncodecModel
from scipy.io import wavfile


def load_codec(device: str = "cpu"):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # same as Kaggle
    model.to(device)
    model.eval()
    return model


def decode_tokens_to_wav(tokens: torch.Tensor, model: EncodecModel, device: str = "cpu"):
    """
    tokens: [n_q, T] int64 tensor on CPU
    returns: audio [samples] float32 on CPU
    """
    # EnCodec (original library) expects a list of (codes, scales)
    # codes: [batch, n_q, frames]
    codes = tokens.unsqueeze(0).to(device)  # [1, n_q, T]
    # We didn't save scales, so approximate with ones
    scales = torch.ones(1, device=device)

    with torch.no_grad():
        # model.decode takes a list of (codes, scales) for each chunk
        dec = model.decode([(codes, scales)])  # [batch=1, channels, samples]

    # Convert to mono [samples]
    audio = dec[0].mean(dim=0).cpu().numpy().astype("float32")
    return audio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sample_numb = "24"
    data_dir = Path(f"slakh2100-encodec{sample_numb}k-tension-pt") / "train"
    train = "train_230"  # change this to pick another file
    pt_path = data_dir / f"{train}.pt"
    print(f"Loading {pt_path}")

    output_dir = Path(f"output/sound_check{sample_numb}k")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt_path, map_location="cpu")
    track_id = data.get("track_id", "UNKNOWN")
    tokens = data["tokens"]
    tension = data["tension"]

    print("track_id:", track_id)
    print("tokens shape:", tokens.shape)
    print("tension shape:", tension.shape)
    print("tension min/max:", float(tension.min()), float(tension.max()))

    model = load_codec(device=device)
    sr = model.sample_rate  # 24000
    print(f"Model sample rate: {sr}")

    audio_np = decode_tokens_to_wav(tokens, model, device=device)
    print("decoded audio shape:", audio_np.shape, "dtype:", audio_np.dtype)

    out_path = output_dir / f"sound_check_{train}.wav"
    wavfile.write(str(out_path), sr, audio_np)
    print(f"Saved decoded audio to {out_path}")


if __name__ == "__main__":
    main()