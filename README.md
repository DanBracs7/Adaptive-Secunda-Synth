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

- Report: [`DLAIReport2026.pdf`](DLAIReport2026.pdf)
- Demo: <https://danbracs7.github.io/Adaptive-Secunda-Synth/> (see *Demo* below)


## Changes since the first submission

The first submission is tagged
[`submission-v1`](https://github.com/DanBracs7/Adaptive-Secunda-Synth/tree/submission-v1);
the full difference is
[`submission-v1...main`](https://github.com/DanBracs7/Adaptive-Secunda-Synth/compare/submission-v1...main).
Since then, the evaluation adds four experiments on the controls and on
the fine-tuning (E1-E4); the generation code is new (`src/generation.py`,
which undoes the delay pattern and passes the controls on their training
scale; the old script is in `src/archive/`); the report is rewritten; the
notebooks run from this repository and are saved with their outputs; and
the checkpoints and tensors are public on Kaggle. A key/value cache for
faster generation and a playable browser demo are also new.

## Results at a glance

All numbers come from the executed notebooks listed below.

- **The model uses the controls (E1).** Replacing the real controls with
  those of another track raises the test cross-entropy by 0.87; almost
  all of the effect comes from tension.
- **The generated audio follows tension (E2).** On 288 generated clips,
  tension realises 95% (Slakh-FT) and 88% (OST-FT) of the requested
  change and follows a step within one frame. The combat score has a
  small effect, only at high tension.
- **Listening check (E3).** 20 blind A/B pairs drawn at random from the
  E2 clips, prompt removed, judged by three listeners (the author and two
  others). With matched loudness the tension contrast was not heard
  (10/24 answers), while the combat-score contrast at high tension was
  heard in 19/24 answers (7/8, 4/8 and 8/8 per listener; two-sided
  binomial p = 0.007 on the pooled answers, 12/16 and p = 0.08 without
  the author). Counts per block: `results/listening_test/summary.csv`.
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

## Demo

<https://danbracs7.github.io/Adaptive-Secunda-Synth/> is a small browser
scene: walk through a valley with the arrow keys, WASD or the on-screen
pad. Inside the enemy camp the game requests combat music (tension 0.77,
combat score 0.85), outside it exploration music (0.30, 0.21). The page
is `docs/index.html`, one self-contained file served by GitHub Pages.

Secunda does not generate in real time on a CPU (about twice the clip
duration), so the demo plays clips generated in advance by OST-FT and
crossfades between them: 1.5 s when the state changes, 0.5 s between
consecutive clips of the same state.

- Every clip continues 5 s of a Skyrim or Witcher 3 passage
  (`configs/demo_prompts.csv`), with the controls moving from the
  prompt's own level to the target in 1 s. Only the generated part is
  played: the soundtracks are under copyright and are never published.
- Exploration clips are 5 s of generated audio. Some combat clips are
  loops: since the quality of a continuation drops with its distance
  from the prompt, four 3-s continuations of the same prompt (different
  seeds) are joined with 0.25-s equal-power crossfades into one 11.25-s
  loop.
- The clips were chosen by ear (4 exploration and 6 combat clips, out of
  the clips of 36 passages), so they show the better outputs, not the
  average one.
- A second section plays E2 clips of three Slakh2100 test prompts
  (CC BY 4.0, so their 1-s real prompt is kept): the same prompt and
  random numbers with low controls, high controls, and a tension step at
  5 s. These prompts were picked by a rule, not by ear: their measured
  tension change is the closest to the median of the 12 E2 prompts.

To rebuild the page (needs the OST-FT checkpoint and the soundtrack
audio, which is not distributed, and the WAV files of notebook 05):

```bash
python src/demo_clips.py bank --settings base fine_t07       # 5-s clips
python src/demo_clips.py bank --settings base --generated-seconds 3 --takes 4 --loop --labels fight --tag 3s
python src/demo_review.py      # outputs/demo_bank/review.html: choose the clips, export demo_selection.csv
python src/demo_page.py        # docs/index.html
```

## Checkpoints and data

| What | Kaggle | License |
|---|---|---|
| Baseline | [secunda-slakh-60-epochs](https://www.kaggle.com/datasets/danielebracoloni/secunda-slakh-60-epochs) | CC BY 4.0 |
| Slakh-FT | [secunda-finetuned-on-slakh](https://www.kaggle.com/models/danielebracoloni/secunda-finetuned-on-slakh) | CC BY 4.0 |
| OST-FT | [secunda-finetuned-on-ost-v2](https://www.kaggle.com/models/danielebracoloni/secunda-finetuned-on-ost-v2) | CC BY-NC 4.0 |
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
| `01_eval_teacher_forced` | Per-codebook test cross-entropy of Baseline, Slakh-FT, OST-FT | Table 1, appendix |
| `02_control_statistics_audit` | Distributions of the controls, normalisation checks | Method, appendix |
| `03_generation_smoke_test` | Corrected generation: logits equal to teacher forcing, speed | Method |
| `04_condition_ablation` | E1: zeroed, constant, shuffled and swapped controls | Section 4, appendix |
| `05_control_adherence` | E2: controls measured on 288 generated clips | Section 4, Figure 1, appendix |
| `06_e4_control_runs` (`_R1`, `_R2`) | E4: training and test evaluation of the control runs | Section 4, Table 1 |
| `07_e4_decomposition` | E4: decomposition with bootstrap confidence intervals | Section 4, appendix |
| `archive/` | Original notebooks of the first submission (tokenizer, training, evaluation) | - |

Scripts outside the notebooks:

- `src/listening_test.py`: builds the blind listening page (E3) and
  analyses the answers in `results/listening_test/responses/` into
  `results/listening_test/summary.csv`.
- `src/report_figures.py`: the figures of the report (`figures/`), drawn
  from the files in `results/`.
- `src/demo_clips.py`: generates the demo clips from the passages in
  `configs/demo_prompts.csv`, as single takes or as loops of short takes.
- `src/demo_review.py`: local review page to choose the demo clips by ear
  (it plays the game-music prompts, so it is never published).
- `src/demo_page.py`: builds the demo page `docs/index.html`.
- `src/prompt_candidates.py`: stored controls of the demo passages and
  Slakh prompt candidates.
- `src/benchmark_generation.py`: generation speed with and without the
  key/value cache.

## Repository structure

```text
src/          model, data, controls, generation, training loop, experiments, demo
tests/        tests of the code (no GPU or checkpoint needed)
notebooks/    executed notebooks 00-07, archive/ with the first submission
configs/      normalisation statistics, excluded tensors, preprocessing configuration, demo passages
manifests/    track list with train / validation / test split (all_tracks.csv; older versions kept for the earlier splits)
csv/          per-track feature tables, used to check configs/norm_stats.json (notebook 02, src/verify_norm_stats.py)
results/      tables and figures written by the notebooks and scripts
figures/      figures of the report
docs/         demo page (GitHub Pages)
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
small effect, only at high tension. A small listening check (three
listeners, E3) heard that effect, but not the tension contrast once
loudness was matched. Generation runs slower than real time on a CPU,
so the demo plays pre-generated clips.
