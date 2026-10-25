"""
Central path configuration for notebooks and scripts.

Before the revision every notebook hardcoded its own Kaggle paths. All
paths are now resolved here, in this order:

  1. environment variable SECUNDA_<NAME> (e.g. SECUNDA_CKPT_SLAKH_FT),
     useful to point a single entry somewhere else;
  2. files that live inside the repository (manifest, normalisation
     statistics) are the same in every environment;
  3. Kaggle defaults when /kaggle/input exists;
  4. local defaults, relative to the repository root.

Local layout expected by the defaults (same folders used by
scripts/build_manifest.py):

  datasets/slakh_final_1395_tensors/{train,test}/*.pt
  datasets/skyrim/*.pt
  datasets/witcher3/*.pt
  checkpoints/<original checkpoint file names>
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_DATASETS = KAGGLE_INPUT / "datasets" / "danielebracoloni"
KAGGLE_MODELS = KAGGLE_INPUT / "models" / "danielebracoloni"

# Files versioned in the repository: identical on Kaggle and locally.
REPO_PATHS = {
    "manifest": REPO_ROOT / "manifests" / "all_tracks.csv",
    "norm_stats": REPO_ROOT / "configs" / "norm_stats.json",
}

KAGGLE_PATHS = {
    "slakh_train": KAGGLE_DATASETS / "slakh-tensors-final" / "train",
    "slakh_test": KAGGLE_DATASETS / "slakh-tensors-final" / "test",
    "skyrim": (
        KAGGLE_DATASETS / "ost-tensors-sky-w3" / "ost_tensors" / "skyrim"
    ),
    "witcher3": (
        KAGGLE_DATASETS / "ost-tensors-sky-w3" / "ost_tensors" / "witcher3"
    ),
    # Baseline: initial Slakh training (60 epochs).
    "ckpt_baseline": (
        KAGGLE_DATASETS
        / "secunda-slakh-60-epochs"
        / "secunda_slakh8cb_best_final.pt"
    ),
    # Slakh-FT: Baseline fine-tuned on the enlarged Slakh pool.
    "ckpt_slakh_ft": (
        KAGGLE_MODELS
        / "secunda-finetuned-on-slakh" / "pytorch" / "default" / "1"
        / "secunda_slakh8cb_finetuneslakh__best.pt"
    ),
    # OST-FT: Slakh-FT fine-tuned on Skyrim + Witcher 3.
    "ckpt_ost_ft": (
        KAGGLE_MODELS
        / "secunda-finetuned-on-ost-v2" / "pytorch" / "default" / "1"
        / "secunda_slakh8cb_finetune_best_ost.pt"
    ),
    # E4 control runs R1 and R2 (notebook 06): model-only copies of final.pt.
    "ckpt_e4_r1": (
        KAGGLE_DATASETS / "secunda-e4-control-runs" / "secunda_e4_R1_final_model.pt"
    ),
    "ckpt_e4_r2": (
        KAGGLE_DATASETS / "secunda-e4-control-runs" / "secunda_e4_R2_final_model.pt"
    ),
    "output_dir": Path("/kaggle/working"),
}

LOCAL_PATHS = {
    "slakh_train": (
        REPO_ROOT / "datasets" / "slakh_final_1395_tensors" / "train"
    ),
    "slakh_test": (
        REPO_ROOT / "datasets" / "slakh_final_1395_tensors" / "test"
    ),
    "skyrim": REPO_ROOT / "datasets" / "skyrim",
    "witcher3": REPO_ROOT / "datasets" / "witcher3",
    "ckpt_baseline": (
        REPO_ROOT / "checkpoints" / "secunda_slakh8cb_best_final.pt"
    ),
    "ckpt_slakh_ft": (
        REPO_ROOT / "checkpoints" / "secunda_slakh8cb_finetuneslakh__best.pt"
    ),
    "ckpt_ost_ft": (
        REPO_ROOT / "checkpoints" / "secunda_slakh8cb_finetune_best_ost.pt"
    ),
    "ckpt_e4_r1": REPO_ROOT / "checkpoints" / "secunda_e4_R1_final_model.pt",
    "ckpt_e4_r2": REPO_ROOT / "checkpoints" / "secunda_e4_R2_final_model.pt",
    "output_dir": REPO_ROOT / "outputs",
}


def running_on_kaggle():
    return KAGGLE_INPUT.exists()


def get_path(name):
    """Return the path registered under `name` (see module docstring)."""
    environment_variable = "SECUNDA_" + name.upper()

    if environment_variable in os.environ:
        return Path(os.environ[environment_variable])

    if name in REPO_PATHS:
        return REPO_PATHS[name]

    table = KAGGLE_PATHS if running_on_kaggle() else LOCAL_PATHS

    if name not in table:
        raise KeyError(
            f"Unknown path name '{name}'. "
            f"Known names: {sorted(set(REPO_PATHS) | set(table))}"
        )

    return table[name]


def source_roots_for_split(split):
    """
    Mapping source -> tensor folder, as expected by SecundaAudioDataset.

    Slakh test tensors keep source="slakh" in the manifest but live in a
    different folder from the Slakh train/val tensors
    (see scripts/build_manifest.py), so the root depends on the split.
    """
    slakh_key = "slakh_test" if split == "test" else "slakh_train"

    return {
        "slakh": get_path(slakh_key),
        "skyrim": get_path("skyrim"),
        "witcher3": get_path("witcher3"),
    }
