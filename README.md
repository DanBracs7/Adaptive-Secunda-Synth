# Secunda: Controllable Autoregressive Generation for Adaptive Videogame Music

Secunda is a 48.5M-parameter causal transformer that generates EnCodec
tokens of 24 kHz music. It is conditioned frame by frame, through FiLM,
on two acoustic controls:

- **Tension**: frame RMS energy, normalised to [0, 1].
- **Combat score**: a weighted sum of five features of percussive,
  aggressive activity (percussive-energy ratio, percussive loudness,
  spectral centroid, dissonance proxy, note shortness) over a causal
  5-second window.

The model predicts the first eight EnCodec codebooks (Q0-Q7) with the
MusicGen delay pattern. It was trained on Slakh2100 and then adapted to
Skyrim and Witcher 3 soundtrack excerpts.

- Report: `DLAIReport2026.pdf` - to be updated
- Demo: *link  to add after publication*

## Results at a glance

All numbers come from the executed notebooks listed below.

- **The model uses the controls (E1).** Replacing the real controls with
  those of another track raises the test cross-entropy by 0.87; almost
  all of the effect comes from tension.
- **The generated audio follows tension (E2).** On 288 generated clips,
  tension realises 95% (Slakh-FT) and 88% (OST-FT) of the requested
  change and follows a step within one frame. The combat score has a
  small effect, only at high tension.
- **Listening check (E3).** Blind A/B pairs drawn at random from the E2
  clips, with loudness matching.
- **What drives the fine-tuning gain (E4).** Two control runs show that
  91% of the Baseline -> Slakh-FT improvement comes from the extra
  training, 21% from the larger dataset, and that the new loss weights
  cost 12%.

| Checkpoint | Training | Test cross-entropy (unweighted, 151 Slakh tracks) |
|---|---|---:|
| Baseline | 35,220 steps on 704 Slakh tracks | 5.26 |
| R1 (control run) | Baseline + 25,625 steps, same 704 tracks, Baseline loss weights | 4.99 |
| R2 (control run) | Baseline + 25,625 steps, 1,229 tracks, Baseline loss weights | 4.93 |
| Slakh-FT | Baseline + 25,625 steps, 1,229 tracks, new loss weights | 4.97 |
| OST-FT | Slakh-FT + 1,428 steps on the soundtrack excerpts | 5.05 |

The first submission called Baseline, Slakh-FT and OST-FT "CP3", "CP1"
and "CP2".

## Checkpoints and data

| What | Kaggle | License |
|---|---|---|
| Baseline | [secunda-slakh-60-epochs](https://www.kaggle.com/datasets/danielebracoloni/secunda-slakh-60-epochs) | CC BY 4.0 |
| Slakh-FT | [secunda-finetuned-on-slakh](https://www.kaggle.com/models/danielebracoloni/secunda-finetuned-on-slakh) | CC BY 4.0 |
| OST-FT | [secunda-finetuned-on-ost-v2](https://www.kaggle.com/models/danielebracoloni/secunda-finetuned-on-ost-v2) | research and educational use |
| R1, R2 | [secunda-e4-control-runs](https://www.kaggle.com/datasets/danielebracoloni/secunda-e4-control-runs) | CC BY 4.0 |
| Slakh tensors | [slakh-tensors-final](https://www.kaggle.com/datasets/danielebracoloni/slakh-tensors-final) | CC BY 4.0 |

The Slakh tensors are derived from
[Slakh2100](http://www.slakh.com/) (Manilow et al., 2019, CC BY 4.0):
EnCodec tokens (32 codebooks, 75 frames/s) of the mixtures, with the
frame-aligned tension and combat-score curves.

The soundtrack tensors are encoded game music under copyright and are
**not** distributed. The parts of notebooks 02-05 that read them (OST
statistics, OST validation, OST-FT prompts) cannot be re-run without
them; their outputs are kept in the executed notebooks.

## Notebooks

Every notebook is executed, keeps its outputs, and lists its Kaggle
inputs in its first cell. On Kaggle it clones this repository and imports
the code from `src/`; all paths are resolved by `src/paths.py`.

| Notebook | Content | Report |
|---|---|---|
| `00_checkpoint_provenance` | Identity and training configuration of every checkpoint | Method |
| `01_eval_teacher_forced` | Per-codebook test cross-entropy of Baseline, Slakh-FT, OST-FT | Table 1 |
| `02_control_statistics_audit` | Distributions of the controls, normalisation checks | Method, appendix |
| `03_generation_smoke_test` | Corrected generation: logits equal to teacher forcing, speed | Method |
| `04_condition_ablation` | E1: zeroed, constant, shuffled and swapped controls | Section 4 |
| `05_control_adherence` | E2: controls measured on 288 generated clips | Section 4, Figure 1 |
| `06_e4_control_runs` (`_R1`, `_R2`) | E4: training and test evaluation of the control runs | Section 5 |
| `07_e4_decomposition` | E4: decomposition with bootstrap confidence intervals | Section 5, Table 1 |
| `archive/` | Original notebooks of the first submission (tokenizer, training, evaluation) | - |

Scripts outside the notebooks:

- `src/listening_test.py`: builds the blind listening page (E3) and
  analyses the answers in `results/listening_test/responses/`.
- `src/demo_clips.py`: generates the demo clips from the passages in
  `configs/demo_prompts.csv`.
- `src/prompt_candidates.py`: stored controls of the demo passages and
  Slakh prompt candidates.
- `src/benchmark_generation.py`: generation speed with and without the
  key/value cache.

## Repository structure

```text
src/          model, data, controls, generation, training loop, experiments
tests/        tests for every module (no GPU or checkpoint needed)
notebooks/    executed notebooks 00-07, archive/ with the first submission
configs/      normalisation statistics, preprocessing configuration, demo passages
manifests/    track list with train / validation / test split
results/      tables and figures written by the notebooks and scripts
```

## Running locally

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the tests from the repository root (Git Bash or any POSIX shell):

```bash
for test in tests/test_*.py; do python "$test"; done
```

Local data and checkpoints go where `src/paths.py` expects them
(`datasets/slakh_final_1395_tensors/{train,test}/`, `checkpoints/`); any
single path can be overridden with an environment variable
`SECUNDA_<NAME>`, for example `SECUNDA_CKPT_OST_FT`.

## Limitations

The generated audio still contains artefacts and longer continuations
tend to repeat themselves; at 6 kbps the codec itself limits fidelity.
Tension acts mostly as a loudness control, while the combat score has a
small effect, only at high tension. Generation runs slower than real
time on a CPU, so the demo plays pre-generated clips.
