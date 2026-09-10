# Adaptive Secunda Synth


Secunda is a controllable autoregressive audio-generation prototype for
short adaptive videogame-music cues. The model predicts discrete EnCodec
audio tokens conditioned on two continuous, frame-aligned controls:

- **Tension**: normalized short-time RMS energy.
- **Combat score**: an acoustic-activity proxy based on percussiveness,
  articulation, brightness, and harmonic roughness.

The project uses a causal Transformer with FiLM conditioning to predict
the first eight EnCodec residual codebooks (Q0--Q7). It was trained on
Slakh instrumental material and subsequently fine-tuned on videogame soundtracks.
([Simple Model Showcase](https://www.kaggle.com/code/danielebracoloni/secunda-model-showcase))

## Main results

Evaluation was conducted on 151 held-out Slakh tracks.

| Checkpoint | Description | Weighted test loss |
|---|---|---:|
| CP3 | Initial Slakh baseline | 5.54 |
| CP1 | Fine-tuned on enlarged Slakh data | 5.25 |
| CP2 | Further fine-tuned on OST material | 5.32 |

The Slakh fine-tune improves every modeled codebook relative to the
baseline. OST adaptation retains most of this improvement while showing
a small loss increase on Slakh, consistent with domain adaptation.



## Core of the project

- [Secunda Model Showcase](https://www.kaggle.com/code/danielebracoloni/secunda-model-showcase)
- `notebooks/evaluate_test_slakh.ipynb` reproduces the
  checkpoint comparison reported in the paper.
- `notebooks/finetune-of-slakh-baseline-training-8kb.ipynb` documents
  the final Slakh fine-tuning configuration.
- `src/model.py` contains the FiLM-conditioned `SecundaTransformer`.
- `src/frame_codec.py` implements delayed codebook serialization and
  the weighted multi-codebook loss.
- `src/audio_dataset.py` loads the tensor manifest and generates
  deterministic training and validation crops.

## Training configuration

The final Transformer configuration was:

```text
Embedding dimension: 448
Transformer layers: 14
Attention heads: 8
Feedforward dimension: 2304
Parameters: approximately 48.5M
Modelled EnCodec codebooks: Q0--Q7
Segment duration: 10 seconds
Frame rate: 75 Hz
Batch size: 24
Samples per track per epoch: 20
```

Audio was encoded with EnCodec at 24 kHz and 24.0 kbps. Source tensors
contain 32 codebooks, while the model predicts the first eight.

## Reproducibility note

Training and evaluation were executed on Kaggle using a Tesla T4 GPU.
The raw audio, preprocessed EnCodec tensors, generated checkpoints, and
Kaggle-specific mounted paths are not included in this repository due to
their size and source-data constraints.

The notebooks retain the original Kaggle paths to document the executed
experiments. To reproduce training, attach equivalent tensor datasets
and update the path configuration cells.

## Limitations

The model predicts coarse codebooks more accurately than later residual
codebooks. As a result, generated audio remains an experimental
prototype and can contain noticeable artifacts. The principal expected
improvements are more training data, longer training, and stronger
modeling of higher residual codebooks.

## Report

The full project report is available in the `report/` directory.