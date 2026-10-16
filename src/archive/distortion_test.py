import torch
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly
from scipy.io import wavfile
from transformers import AutoProcessor, EncodecModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Point to the ORIGINAL Slakh mix.flac you downloaded
    track_path = Path("track173(id_398)mix.flac")  # <-- adjust filename/path
    assert track_path.exists(), f"File not found: {track_path}"

    # 2) Load with soundfile (no torchaudio / torchcodec)
    audio, sr = sf.read(str(track_path))  # shape: [samples] or [samples, channels]
    print(f"Loaded {track_path}")
    print(f"Original audio shape: {audio.shape}, sample rate: {sr}")

    # 3) Convert to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    print("After mono:", audio.shape)

    # 4) Resample to 32 kHz
    target_sr = 32000
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr
    print("After resample:", audio.shape, "sr:", sr)

    # 4.5) Normalize peak to at most 0.95 to avoid clipping EnCodec
    peak = max(abs(audio.min()), abs(audio.max()))
    print("Peak before normalize:", peak)
    if peak > 0:
        audio = audio / peak * 0.95
    print("Peak after normalize:", max(abs(audio.min()), abs(audio.max())))
    # 5) Load EnCodec 32 kHz
    codec_id = "facebook/encodec_32khz"
    processor = AutoProcessor.from_pretrained(codec_id)
    model = EncodecModel.from_pretrained(codec_id).to(device)
    model.eval()

    # 6) Encode
    inputs = processor(
        raw_audio=audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)
    padding_mask = inputs.get("padding_mask", None)
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    with torch.no_grad():
        enc = model.encode(input_values, padding_mask)
        # 7) Decode using BOTH audio_codes and audio_scales (correct round-trip)
        dec = model.decode(enc.audio_codes, enc.audio_scales, padding_mask)[0]  # [1, samples]

    dec = dec.cpu()
    print("Decoded audio shape:", dec.shape)

    # Make it 1D [samples]
    dec_np = dec.squeeze().numpy().astype("float32")
    print("Decoded audio after squeeze:", dec_np.shape, dec_np.dtype)

    # 8) Save round-tripped audio
    out_dir = Path("output/roundtrip_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roundtrip_track173.wav"

    wavfile.write(str(out_path), sr, dec_np)
    print(f"Saved round-trip audio to {out_path}")

if __name__ == "__main__":
    main()