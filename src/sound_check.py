import torch
from pathlib import Path
from transformers import EncodecModel, AutoProcessor
import torchaudio


def load_codec(device: str = "cpu"):
    codec_id = "facebook/encodec_32khz"
    model = EncodecModel.from_pretrained(codec_id).to(device)
    processor = AutoProcessor.from_pretrained(codec_id)
    model.eval()
    return model, processor


def decode_tokens_to_wav(tokens: torch.Tensor, model: EncodecModel, sample_rate: int = 32000, device: str = "cpu"):
    """
    tokens: [n_q, T] int64 tensor on CPU
    returns: audio [1, samples] float32 on CPU
    """
    tokens = tokens.to(device)
    # EnCodec expects [nb_frames, batch, n_q, frames]; we fake a single frame & batch
    codes = tokens.unsqueeze(0).unsqueeze(0)  # [1, 1, n_q, T]
    # We don't have per-chunk scales saved, so use 1.0 as a reasonable default
    scales = torch.ones(1, 1, device=device)

    with torch.no_grad():
        audio = model.decode(codes, scales, padding_mask=None)[0]  # [batch=1, samples]

    audio = audio.cpu()
    return audio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Path to your unzipped .pt files
    data_dir = Path("slakh2100-encodec32k-tension-pt") / "train"

    # Pick one example
    pt_path = data_dir / "train_000.pt"
    print(f"Loading {pt_path}")
    output_dir = Path("output/sound_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(pt_path, map_location="cpu")
    tokens = data["tokens"]      # [n_q, T]
    tension = data["tension"]    # [T]
    print("tokens shape:", tokens.shape)
    print("tension shape:", tension.shape)
    print("tension min/max:", float(tension.min()), float(tension.max()))

    model, processor = load_codec(device=device)
    sr = processor.sampling_rate  # should be 32000

    audio = decode_tokens_to_wav(tokens, model, sample_rate=sr, device=device)

    out_path = output_dir / "sound_check_train_000.wav"
    torchaudio.save(str(out_path), audio.unsqueeze(0), sample_rate=sr)
    print(f"Saved decoded audio to {out_path}")


if __name__ == "__main__":
    main()