import torch
from pathlib import Path

#You can do the opposite to find very “chill” tracks: sort by mean_T ascending.

def main():
    data_dir = Path("slakh2100-encodec32k-tension-pt") / "train"
    stats = []

    for pt_path in sorted(data_dir.glob("train_*.pt")):
        data = torch.load(pt_path, map_location="cpu")
        tension = data["tension"]
        mean_T = float(tension.mean())
        frac_high = float((tension > 0.8).float().mean())
        stats.append((pt_path.name, mean_T, frac_high))

    # Sort by fraction of high-tension frames, then mean
    stats.sort(key=lambda x: (x[2], x[1]), reverse=True)

    for name, mean_T, frac_high in stats[:10]:
        print(f"{name}: mean_T={mean_T:.3f}, frac_T>0.8={frac_high:.3f}")

if __name__ == "__main__":
    main()