import torch
from pathlib import Path


# ── Configuration ──
dataset = "datasets/skyrim  "
split = ""    #select train, test or ""

# Works for train_000.pt, skyrim_000.pt, witcher_000.pt, etc.
file_pattern = "*.pt"

# Available modes:
# "high_tension"
# "low_tension"
# "high_combat"
# "low_combat"
# "high_tension_low_combat"
# "low_tension_high_combat"
sort_mode = "low_tension_high_combat"

top_k = 40
high_threshold = 0.8


def sort_key(row, mode):
    _, mean_tension, mean_combat, frac_t_high, frac_c_high, _ = row

    if mode == "high_tension":
        return (frac_t_high, mean_tension)

    if mode == "low_tension":
        return (-mean_tension, -frac_t_high)

    if mode == "high_combat":
        return (frac_c_high, mean_combat)

    if mode == "low_combat":
        return (-mean_combat, -frac_c_high)

    if mode == "high_tension_low_combat":
        return (mean_tension - mean_combat, mean_tension)

    if mode == "low_tension_high_combat":
        return (mean_combat - mean_tension, mean_combat)

    raise ValueError(
        f"Unknown sort_mode: {mode}. "
        "Choose high_tension, low_tension, high_combat, "
        "low_combat, high_tension_low_combat, or "
        "low_tension_high_combat."
    )


def main():
    data_dir = Path(dataset) / split

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {data_dir.resolve()}"
        )

    stats = []

    for pt_path in sorted(data_dir.glob(file_pattern)):
        data = torch.load(pt_path, map_location="cpu")

        required_keys = {"track_id", "tension", "combat_score"}
        missing_keys = required_keys - set(data.keys())

        if missing_keys:
            print(
                f"Skipping {pt_path.name}: "
                f"missing {sorted(missing_keys)}"
            )
            continue

        tension = data["tension"].float()
        combat_score = data["combat_score"].float()

        if len(tension) != len(combat_score):
            print(
                f"Skipping {pt_path.name}: "
                f"tension length={len(tension)}, "
                f"combat length={len(combat_score)}"
            )
            continue

        mean_tension = float(tension.mean())
        mean_combat = float(combat_score.mean())

        frac_t_high = float(
            (tension > high_threshold).float().mean()
        )

        frac_c_high = float(
            (combat_score > high_threshold).float().mean()
        )

        track_id = str(data["track_id"])

        stats.append(
            (
                pt_path.name,
                mean_tension,
                mean_combat,
                frac_t_high,
                frac_c_high,
                track_id,
            )
        )

    if not stats:
        raise RuntimeError(
            f"No valid .pt tensor files found in {data_dir.resolve()}"
        )

    stats.sort(
        key=lambda row: sort_key(row, sort_mode),
        reverse=True,
    )

    print(f"\nDataset folder: {data_dir}")
    print(f"Valid tracks: {len(stats)}")
    print(f"Sort mode: {sort_mode}")
    print("-" * 115)

    for (
        file_name,
        mean_tension,
        mean_combat,
        frac_t_high,
        frac_c_high,
        track_id,
    ) in stats[:top_k]:

        print(
            f"{file_name:<18} | "
            f"mean_T={mean_tension:.3f} | "
            f"mean_C={mean_combat:.3f} | "
            f"T>{high_threshold:.1f}={frac_t_high:.3f} | "
            f"C>{high_threshold:.1f}={frac_c_high:.3f} | "
            f"track_id={track_id}"
        )

    dataset_mean_tension = sum(row[1] for row in stats) / len(stats)
    dataset_mean_combat = sum(row[2] for row in stats) / len(stats)

    print("-" * 115)
    print(f"Dataset mean tension: {dataset_mean_tension:.3f}")
    print(f"Dataset mean combat : {dataset_mean_combat:.3f}")


if __name__ == "__main__":
    main()