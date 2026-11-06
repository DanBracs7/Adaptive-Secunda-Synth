"""
Figures of the report, drawn from the saved results (no GPU, no checkpoints).

  figure1  Controls measured on generated audio (experiment E2, notebook 05):
           (a) tension requested and measured around a step at 5 s,
           (b) fraction of each requested change that is realised.
           Source: results/control_adherence/{tension_curves.npz, clips.csv}.
  figure_a_e4
           Decomposition of the Baseline -> Slakh-FT improvement per codebook
           (experiment E4, notebook 07) and validation curves of R1 and R2.
           Source: results/e4/decomposition/decomposition.csv,
           results/e4/R{1,2}/history.json.

Every figure is written as PDF (vector, for LaTeX) and PNG (preview) to
figures/. Widths follow the two-column template: 3.25 in for one column.

Run from the repository root:
    python src/report_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import control_adherence as ca  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"

COLUMN_WIDTH = 3.25   # inches
CHECKPOINT_COLORS = {"Slakh-FT": "#2a78d6", "OST-FT": "#eb6834"}
STEP_COLORS = {"extra training": "#2a78d6", "new data": "#eb6834", "new loss weights": "#1baf7a"}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
PROMPT_SHADE = "#efeeea"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,          # editable text in the PDF
    "savefig.dpi": 300,
})


def save(figure, name):
    FIGURES.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(FIGURES / f"{name}{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("written", FIGURES / f"{name}.pdf")


def mean_and_ci(curves):
    """Mean over units and a normal-approximation 95% CI of the mean, per frame."""
    curves = np.asarray(curves, dtype=np.float64)
    mean = curves.mean(axis=0)
    half_width = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    return mean, mean - half_width, mean + half_width


def realised_fractions(clips):
    """
    One row per (effect, checkpoint): mean measured change divided by the
    requested change, with its bootstrap 95% CI (control_adherence.paired_effect).
    """
    tension_change = ca.LEVELS["tension"]["high"] - ca.LEVELS["tension"]["low"]
    combat_change = ca.LEVELS["combat_score"]["high"] - ca.LEVELS["combat_score"]["low"]
    rows = []
    for checkpoint, group in clips.groupby("checkpoint", sort=False):
        factorial = ca.factorial_effects(group).set_index(["requested", "measured"])
        by_tension = ca.combat_effect_by_tension(group).set_index("comparison")
        effects = [
            ("tension → tension", factorial.loc[("tension", "tension")], tension_change),
            ("combat → combat", factorial.loc[("combat_score", "combat_score")], combat_change),
            ("combat \u2192 combat, high tension", by_tension.loc["combat effect at high tension"], combat_change),
            ("combat \u2192 combat, low tension", by_tension.loc["combat effect at low tension"], combat_change),
            ("tension → combat (cross-talk)", factorial.loc[("tension", "combat_score")], tension_change),
            ("combat → tension", factorial.loc[("combat_score", "tension")], combat_change),
        ]
        for label, row, change in effects:
            rows.append({"effect": label, "checkpoint": checkpoint, "fraction": row["effect"] / change,
                         "low": row["ci_low"] / change, "high": row["ci_high"] / change})
    return pd.DataFrame(rows)


def figure1():
    curves = np.load(RESULTS / "control_adherence" / "tension_curves.npz")
    clips = pd.read_csv(RESULTS / "control_adherence" / "clips.csv")

    figure, (step_axis, effect_axis) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH, 3.55), gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.55})

    # (a) Tension step: requested vs measured, mean over the 24 prompt x seed units.
    frames = np.arange(ca.TOTAL_FRAMES)
    seconds = frames / ca.FRAME_RATE
    step_axis.axvspan(0, ca.PROMPT_FRAMES / ca.FRAME_RATE, color=PROMPT_SHADE, lw=0)
    step_axis.text(0.5, 0.93, "prompt", ha="center", va="top", color=TEXT_SECONDARY, fontsize=6)
    requested = np.where(frames < ca.STEP_FRAME, ca.LEVELS["tension"]["low"], ca.LEVELS["tension"]["high"])
    after_prompt = frames >= ca.PROMPT_FRAMES
    step_axis.plot(seconds[after_prompt], requested[after_prompt], ls=(0, (3, 2)), lw=1.0,
                   color=TEXT_SECONDARY, label="requested")
    for checkpoint, color in CHECKPOINT_COLORS.items():
        keys = [key for key in curves.files if key.startswith(f"{checkpoint}|") and key.endswith("|tension_step")]
        mean, low, high = mean_and_ci([curves[key] for key in keys])
        step_axis.fill_between(seconds, low, high, color=color, alpha=0.18, lw=0)
        step_axis.plot(seconds, mean, color=color, lw=1.1, label=f"{checkpoint} (measured)")
    step_axis.set_xlim(0, 10)
    step_axis.set_ylim(0, 1)
    step_axis.set_xticks([0, 1, 2, 4, 5, 6, 8, 10])
    step_axis.set_yticks([0, 0.14, 0.5, 0.77, 1])
    step_axis.set_yticklabels(["0", "0.14", "0.5", "0.77", "1"])
    step_axis.set_xlabel("time (s)")
    step_axis.set_ylabel("tension")
    step_axis.grid(axis="y", color=GRID, lw=0.5)
    step_axis.set_axisbelow(True)
    step_axis.legend(loc="lower left", bbox_to_anchor=(0.11, 0.2), frameon=False, handlelength=1.8)
    step_axis.set_title("(a) Tension step at 5 s", loc="left", color=TEXT)

    # (b) Fraction of each requested change that is realised.
    table = realised_fractions(clips)
    labels = list(dict.fromkeys(table["effect"]))
    positions = np.arange(len(labels))[::-1].astype(float)
    effect_axis.axvline(0, color=TEXT_SECONDARY, lw=0.6)
    effect_axis.axvline(1, color=TEXT_SECONDARY, lw=0.8, ls=(0, (3, 2)))
    effect_axis.text(1.0, len(labels) - 0.35, "requested", ha="center", va="bottom", color=TEXT_SECONDARY,
                     fontsize=6)
    for offset, (checkpoint, color) in zip((0.14, -0.14), CHECKPOINT_COLORS.items()):
        rows = table[table["checkpoint"] == checkpoint].set_index("effect").loc[labels]
        y = positions + offset
        effect_axis.errorbar(rows["fraction"], y,
                             xerr=[rows["fraction"] - rows["low"], rows["high"] - rows["fraction"]],
                             fmt="o", ms=3.2, color=color, ecolor=color, elinewidth=0.9, capsize=0,
                             label=checkpoint)
    effect_axis.set_yticks(positions)
    effect_axis.set_yticklabels(labels)
    effect_axis.tick_params(axis="y", length=0)
    effect_axis.set_ylim(-0.6, len(labels) - 0.1)
    effect_axis.set_xlim(-0.15, 1.12)
    effect_axis.set_xlabel("fraction of the requested change realised (95% CI)")
    effect_axis.grid(axis="x", color=GRID, lw=0.5)
    effect_axis.set_axisbelow(True)
    effect_axis.spines["left"].set_visible(False)
    effect_axis.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.1)
    effect_axis.set_title("(b) Effect of each control", loc="left", color=TEXT)

    save(figure, "fig1_control_adherence")
    return table


def figure_a_e4():
    decomposition = pd.read_csv(RESULTS / "e4" / "decomposition" / "decomposition.csv")
    steps = decomposition[decomposition["step"] != "total"]

    figure, (codebook_axis, curve_axis) = plt.subplots(
        1, 2, figsize=(2 * COLUMN_WIDTH + 0.25, 2.1), gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.32})

    # (a) Change per codebook for each step, with 95% CIs.
    codebooks = np.arange(8)
    width = 0.26
    for index, (_, row) in enumerate(steps.iterrows()):
        values = np.array([row[f"delta_q{q}"] for q in codebooks])
        low = np.array([row[f"q{q}_low"] for q in codebooks])
        high = np.array([row[f"q{q}_high"] for q in codebooks])
        x = codebooks + (index - 1) * (width + 0.02)
        codebook_axis.bar(x, values, width, color=STEP_COLORS[row["step"]], lw=0,
                          label=row["step"])
        codebook_axis.errorbar(x, values, yerr=[values - low, high - values], fmt="none",
                               ecolor=TEXT, elinewidth=0.6, capsize=0)
    codebook_axis.axhline(0, color=TEXT_SECONDARY, lw=0.6)
    codebook_axis.set_xticks(codebooks)
    codebook_axis.set_xticklabels([f"Q{q}" for q in codebooks])
    codebook_axis.tick_params(axis="x", length=0)
    codebook_axis.set_ylabel("Δ test cross-entropy")
    codebook_axis.grid(axis="y", color=GRID, lw=0.5)
    codebook_axis.set_axisbelow(True)
    codebook_axis.legend(loc="upper right", frameon=False)
    codebook_axis.set_title("(a) Change per codebook (negative = better)", loc="left", color=TEXT)

    # (b) Validation loss of the two control runs (same weights and validation set).
    for run, color in (("R1", STEP_COLORS["extra training"]), ("R2", STEP_COLORS["new data"])):
        history = json.loads((RESULTS / "e4" / run / "history.json").read_text(encoding="utf-8"))
        curve_axis.plot(np.asarray(history["global_step"]) / 1000, history["val_loss"], color=color, lw=1.1,
                        marker="o", ms=2.2, label=run)
        curve_axis.annotate(f"{history['val_loss'][-1]:.3f}", (history["global_step"][-1] / 1000,
                            history["val_loss"][-1]), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=6, color=TEXT_SECONDARY)
    curve_axis.set_xlabel("training step (thousands)")
    curve_axis.set_ylabel("validation loss")
    curve_axis.grid(axis="y", color=GRID, lw=0.5)
    curve_axis.set_axisbelow(True)
    curve_axis.legend(loc="upper right", frameon=False)
    curve_axis.set_title("(b) Validation loss of R1 and R2", loc="left", color=TEXT)

    save(figure, "figA_e4_decomposition")


def main():
    table = figure1()
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    figure_a_e4()


if __name__ == "__main__":
    sys.exit(main())
