import torch
from pathlib import Path

#You can do the opposite to find very “chill” tracks: sort by mean_T ascending.

def main():
    #dataset = "slakh2100-encodec24k-tension-pt"
    dataset = "custom_tracks"
    data_dir = Path(f"{dataset}") / "train"
    stats = []

    for pt_path in sorted(data_dir.glob("train_*.pt")):
        data = torch.load(pt_path, map_location="cpu")
        tension = data["tension"]
        mean_T = float(tension.mean())
        frac_high = float((tension > 0.8).float().mean())
        track_id = data["track_id"]
        stats.append((pt_path.name, mean_T, frac_high, track_id))

    # Sort by fraction of high-tension frames, then mean
    stats.sort(key=lambda x: (x[2], x[1]), reverse=True)

    for name, mean_T, frac_high, track_id in stats[:50]:
        print(f"{name}: mean_T={mean_T:.3f}, frac_T>0.8={frac_high:.3f}, track_id={track_id}")

if __name__ == "__main__":
    main()