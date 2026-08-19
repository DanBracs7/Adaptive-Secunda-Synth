import torch
from pathlib import Path

from encodec import EncodecModel
from scipy.io import wavfile


def load_codec(device: str = "cpu"):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(24.0)
    model.to(device)
    model.eval()
    return model


def decode_tokens_to_wav(
    tokens: torch.Tensor,
    model: EncodecModel,
    device: str = "cpu",
):
    if tokens.ndim != 2:
        raise ValueError(
            f"Expected tokens shaped [n_q, T], "
            f"but received {tuple(tokens.shape)}"
        )

    codes = tokens.unsqueeze(0).long().to(device)  # [1, n_q, T]

    # The saved tensors contain codes only. For this 24 kHz model,
    # no artificial scale should be introduced at decoding time.
    scales = None

    with torch.no_grad():
        decoded = model.decode([(codes, scales)])

    # 24 kHz EnCodec is mono. This keeps the script robust if a
    # future codec returns multiple channels.
    audio = decoded[0].mean(dim=0).detach().cpu().numpy().astype("float32")

    return audio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Dataset configuration ──
    sample_numb = "24"

    # Current Slakh checkpoint folder:
    folder_name = "datasets/skyrim"

    # Future examples:
    # folder_name = "ost_tensors"
    # folder_name = "skyrim_tensors"
    # folder_name = "witcher_tensors"

    split = ""             #train test or ""
    data_dir = Path(folder_name) / split

    # Enter a file BASE NAME without .pt.
    # Slakh example: "train_025"
    # Future OST example: "skyrim_004" or "witcher_012"
    tensor_name = "skyrim_006"

    pt_path = data_dir / f"{tensor_name}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Tensor file not found: {pt_path.resolve()}"
        )

    print(f"Loading {pt_path}")

    output_dir = Path("output") / f"sound_check_{folder_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt_path, map_location="cpu")

    required_keys = {"tokens", "tension", "combat_score"}
    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise KeyError(
            f"{pt_path.name} is missing: {sorted(missing_keys)}"
        )

    track_id = data.get("track_id", "UNKNOWN")
    tokens = data["tokens"].long()
    tension = data["tension"].float()
    combat_score = data["combat_score"].float()

    if tokens.shape[-1] != len(tension):
        raise ValueError(
            f"Token/tension mismatch: "
            f"tokens={tokens.shape[-1]}, tension={len(tension)}"
        )

    if tokens.shape[-1] != len(combat_score):
        raise ValueError(
            f"Token/combat mismatch: "
            f"tokens={tokens.shape[-1]}, "
            f"combat_score={len(combat_score)}"
        )

    print("track_id:", track_id)
    print("tokens shape:", tuple(tokens.shape))
    print("tension shape:", tuple(tension.shape))
    print("combat score shape:", tuple(combat_score.shape))

    print(
        "tension min / mean / max:",
        f"{float(tension.min()):.3f} / "
        f"{float(tension.mean()):.3f} / "
        f"{float(tension.max()):.3f}",
    )

    print(
        "combat min / mean / max:",
        f"{float(combat_score.min()):.3f} / "
        f"{float(combat_score.mean()):.3f} / "
        f"{float(combat_score.max()):.3f}",
    )

    model = load_codec(device=device)
    sr = model.sample_rate
    print(f"Model sample rate: {sr}")

    audio_np = decode_tokens_to_wav(
        tokens=tokens,
        model=model,
        device=device,
    )

    peak = float(abs(audio_np).max())

    if peak > 1.0:
        audio_np = audio_np / peak * 0.99

    print(
        "decoded audio shape:",
        audio_np.shape,
        "dtype:",
        audio_np.dtype,
    )

    out_path = output_dir / f"sound_check_{tensor_name}.wav"

    wavfile.write(
        str(out_path),
        sr,
        audio_np,
    )

    print(f"Saved decoded audio → {out_path}")


if __name__ == "__main__":
    main()