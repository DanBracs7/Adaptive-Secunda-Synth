"""
Experiment E3: small blind A/B listening check on the clips of E2.

Each pair contains two clips generated from the same prompt with the same
random numbers (notebook 05), differing only in the requested controls.
The listener hears A and B and answers one forced-choice question: which
of the two sounds more like combat music. Relative judgements work even
when neither clip would be good enough to use.

Blocks (pairs chosen at random with a fixed seed, never by ear; every
prompt appears in one block only):
  A  tension low vs high (combat score low), levels as generated:
     the "chill vs combat" contrast of the demo, loudness included;
  B  the same contrast with the two clips matched in loudness (RMS):
     does tension change more than the volume?
  C  combat score low vs high at high tension, loudness matched:
     the only condition where E2 measured a combat-score effect.

The first second of every clip (the real prompt) is removed, so listeners
compare generated audio only and no original game music is distributed.

Usage, from the repository root (needs the WAV files written by notebook 05):
    python src/listening_test.py build      # outputs/listening_test/listening_test.html
    python src/listening_test.py analyze    # results/listening_test/responses/*.csv
"""

import argparse
import base64
import csv
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIPS_CSV = PROJECT_ROOT / "results" / "control_adherence" / "clips.csv"
AUDIO_DIR = PROJECT_ROOT / "outputs" / "control_adherence" / "audio"
PAGE_PATH = PROJECT_ROOT / "outputs" / "listening_test" / "listening_test.html"
RESULTS_DIR = PROJECT_ROOT / "results" / "listening_test"

SAMPLE_RATE = 24_000
TRIM_SECONDS = 1.0            # the real prompt
FADE_SECONDS = 0.02           # avoids a click at the cut
TARGET_RMS_DB = -20.0         # loudness-matched blocks
PEAK_LIMIT = 0.99
SELECTION_SEED = 2026

# (block, number of prompts per checkpoint, quieter/less combat condition, louder/more combat condition)
BLOCKS = (
    ("A", 2, "tL_cL", "tH_cL"),
    ("B", 4, "tL_cL", "tH_cL"),
    ("C", 4, "tH_cL", "tH_cH"),
)
LOUDNESS_MATCHED = {"A": False, "B": True, "C": True}
CHECKPOINTS = ("Slakh-FT", "OST-FT")


# ── Pair selection ──

def select_pairs(clips_csv=CLIPS_CSV, seed=SELECTION_SEED):
    """
    Deterministic random choice of prompts and seeds for every block.
    Returns a list of dicts: pair_id, block, checkpoint, prompt, seed,
    low_condition, high_condition ("high" = the clip requested to be more
    combat-like).
    """
    with open(clips_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rng = random.Random(seed)
    pairs = []

    for checkpoint in CHECKPOINTS:
        prompts = sorted({int(r["prompt"]) for r in rows if r["checkpoint"] == checkpoint})
        rng.shuffle(prompts)
        position = 0
        for block, count, low_condition, high_condition in BLOCKS:
            for prompt in prompts[position:position + count]:
                pairs.append({
                    "pair_id": f"{block}-{checkpoint}-p{prompt:02d}",
                    "block": block,
                    "checkpoint": checkpoint,
                    "prompt": prompt,
                    "seed": rng.choice((0, 1)),
                    "low_condition": low_condition,
                    "high_condition": high_condition,
                })
            position += count

    return pairs


def clip_path(audio_dir, checkpoint, prompt, seed, condition):
    return Path(audio_dir) / f"{checkpoint}_{prompt:02d}_s{seed}_{condition}.wav"


# ── Audio preparation ──

def rms_db(audio):
    return 20 * math.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-12)


def trim_and_fade(audio, sample_rate=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float64)[int(TRIM_SECONDS * sample_rate):].copy()
    fade = int(FADE_SECONDS * sample_rate)
    ramp = np.linspace(0.0, 1.0, fade)
    audio[:fade] *= ramp
    audio[-fade:] *= ramp[::-1]
    return audio


def prepare_pair(low_audio, high_audio, loudness_matched):
    """
    Trim the prompt, then either match both clips to TARGET_RMS_DB or keep
    their levels. The same extra attenuation is applied to both clips when
    a peak would exceed PEAK_LIMIT, so their relative level is unchanged.
    """
    low, high = trim_and_fade(low_audio), trim_and_fade(high_audio)

    if loudness_matched:
        low_gain = 10 ** ((TARGET_RMS_DB - rms_db(low)) / 20)
        high_gain = 10 ** ((TARGET_RMS_DB - rms_db(high)) / 20)
    else:
        low_gain = high_gain = 1.0

    peak = max(np.abs(low).max() * low_gain, np.abs(high).max() * high_gain)
    limiter = min(1.0, PEAK_LIMIT / peak) if peak > 0 else 1.0

    return low * low_gain * limiter, high * high_gain * limiter


def encode_mp3(audio, sample_rate=SAMPLE_RATE):
    buffer = io.BytesIO()
    soundfile.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate,
                    format="MP3", subtype="MPEG_LAYER_III")
    return buffer.getvalue()


# ── Page ──

PAGE_TEMPLATE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascolto a coppie</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deded8;--accent:#2f5d8a;--accent-fg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#21211f;--line:#34342f;--accent:#7fb0e0;--accent-fg:#10151a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:640px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.4rem;margin:0 0 8px}p{margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:16px}
.muted{color:var(--muted);font-size:.9rem}.row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:10px 16px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}
.play{flex:1;min-width:120px}.choice{flex:1;min-width:120px}.played{border-color:var(--accent)}
input{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);width:100%}
progress{width:100%;height:8px}textarea{width:100%;height:120px;font:12px ui-monospace,monospace}
</style></head><body><main>
<h1>Ascolto a coppie</h1>
<div id="intro" class="card">
<p>Sentirai __N__ coppie di brevi clip (9 secondi) generati da un modello. Per ogni coppia ascolta <b>A</b> e <b>B</b> e scegli quale dei due suona <b>più da combattimento</b> (più intenso, adatto a uno scontro).</p>
<p class="muted">Nessuno dei due deve essere perfetto: scegli quello che lo è <i>di più</i>, anche se la differenza è piccola. Meglio con le cuffie. Circa 8 minuti.</p>
<label class="muted" for="who">Iniziali o soprannome (facoltativo)</label>
<input id="who" maxlength="20" autocomplete="off">
<div class="row"><button class="primary" id="start">Inizia</button></div>
</div>
<div id="trial" class="card" hidden>
<progress id="bar" max="__N__" value="0"></progress>
<p class="muted" id="count"></p>
<div class="row"><button class="play" id="playA">▶ Ascolta A</button><button class="play" id="playB">▶ Ascolta B</button></div>
<p>Quale suona più da combattimento?</p>
<div class="row"><button class="choice" id="chooseA" disabled>A</button><button class="choice" id="chooseB" disabled>B</button></div>
<p class="muted">Le risposte si attivano dopo aver ascoltato entrambi i clip.</p>
</div>
<div id="done" class="card" hidden>
<p><b>Fatto, grazie!</b> Scarica il file delle risposte e mandalo a chi ti ha inviato il test, oppure copia il testo qui sotto.</p>
<div class="row"><button class="primary" id="download">Scarica risposte (CSV)</button><button id="copy">Copia testo</button></div>
<textarea id="csv" readonly></textarea>
</div>
<audio id="player"></audio>
</main>
<script>
const PAIRS = __PAIRS__;
const AUDIO = __AUDIO__;
const listenerId = Math.random().toString(36).slice(2, 8);
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const order = shuffle(PAIRS.map((_, i) => i));
const answers = [];
let index = 0, current = null, heard = {A: false, B: false};
const $ = id => document.getElementById(id);
const player = $("player");

function showTrial() {
  const pair = PAIRS[order[index]];
  const highFirst = Math.random() < 0.5;
  current = {pair, highPosition: highFirst ? "A" : "B",
             A: highFirst ? pair.high : pair.low, B: highFirst ? pair.low : pair.high};
  heard = {A: false, B: false};
  for (const id of ["playA", "playB"]) $(id).classList.remove("played");
  $("chooseA").disabled = $("chooseB").disabled = true;
  $("bar").value = index;
  $("count").textContent = `Coppia ${index + 1} di ${PAIRS.length}`;
}
function play(side) {
  player.src = "data:audio/mpeg;base64," + AUDIO[current[side]];
  player.play().catch(() => {});   // a new click interrupts the previous clip
  heard[side] = true;
  $("play" + side).classList.add("played");
  if (heard.A && heard.B) $("chooseA").disabled = $("chooseB").disabled = false;
}
function choose(side) {
  player.pause();
  answers.push({pair_id: current.pair.pair_id, block: current.pair.block, checkpoint: current.pair.checkpoint,
                trial: index + 1, high_position: current.highPosition, choice: side,
                chose_high: side === current.highPosition ? 1 : 0});
  index += 1;
  if (index < PAIRS.length) showTrial(); else finish();
}
function finish() {
  const who = ($("who").value || "").replace(/[^\\w-]/g, "") || listenerId;
  const header = "listener,pair_id,block,checkpoint,trial,high_position,choice,chose_high";
  const lines = answers.map(a => [who, a.pair_id, a.block, a.checkpoint, a.trial, a.high_position, a.choice, a.chose_high].join(","));
  const text = [header, ...lines].join("\\n") + "\\n";
  $("csv").value = text;
  $("trial").hidden = true; $("done").hidden = false;
  $("download").onclick = () => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], {type: "text/csv"}));
    link.download = `ascolto_${who}.csv`; link.click();
  };
  $("copy").onclick = () => { $("csv").select(); navigator.clipboard?.writeText(text); };
}
$("start").onclick = () => { $("intro").hidden = true; $("trial").hidden = false; showTrial(); };
$("playA").onclick = () => play("A");
$("playB").onclick = () => play("B");
$("chooseA").onclick = () => choose("A");
$("chooseB").onclick = () => choose("B");
</script></body></html>
"""


def build_page(pairs, audio_dir=AUDIO_DIR, page_path=PAGE_PATH):
    """Write the self-contained listening page (clips embedded as MP3)."""
    audio = {}
    page_pairs = []

    for pair in pairs:
        low_wave, low_rate = soundfile.read(
            clip_path(audio_dir, pair["checkpoint"], pair["prompt"], pair["seed"], pair["low_condition"]))
        high_wave, high_rate = soundfile.read(
            clip_path(audio_dir, pair["checkpoint"], pair["prompt"], pair["seed"], pair["high_condition"]))
        if low_rate != SAMPLE_RATE or high_rate != SAMPLE_RATE:
            raise ValueError(f"{pair['pair_id']}: expected {SAMPLE_RATE} Hz audio.")

        low, high = prepare_pair(low_wave, high_wave, LOUDNESS_MATCHED[pair["block"]])
        low_key, high_key = f"{pair['pair_id']}-low", f"{pair['pair_id']}-high"
        audio[low_key] = base64.b64encode(encode_mp3(low)).decode("ascii")
        audio[high_key] = base64.b64encode(encode_mp3(high)).decode("ascii")
        page_pairs.append({"pair_id": pair["pair_id"], "block": pair["block"],
                           "checkpoint": pair["checkpoint"], "low": low_key, "high": high_key})

    html = (PAGE_TEMPLATE
            .replace("__N__", str(len(page_pairs)))
            .replace("__PAIRS__", json.dumps(page_pairs))
            .replace("__AUDIO__", json.dumps(audio)))
    page_path = Path(page_path)
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(html, encoding="utf-8")
    return page_path


def write_pairs_csv(pairs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)


# ── Analysis ──

def wilson_interval(successes, trials, z=1.959964):
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return centre - half_width, centre + half_width


def summarize(responses):
    """
    responses: list of dicts with block, checkpoint, listener, chose_high.
    One row per block (and per block x checkpoint): how often the clip
    requested to be more combat-like was chosen, Wilson 95% CI and a
    two-sided binomial test against 0.5 (chance).
    """
    from scipy.stats import binomtest

    groups = {}
    for response in responses:
        for key in ((response["block"], "all"), (response["block"], response["checkpoint"])):
            groups.setdefault(key, []).append(int(response["chose_high"]))

    rows = []
    for (block, checkpoint), values in sorted(groups.items()):
        successes, trials = sum(values), len(values)
        low, high = wilson_interval(successes, trials)
        rows.append({
            "block": block, "checkpoint": checkpoint, "answers": trials,
            "chose_requested": successes, "proportion": successes / trials,
            "ci_low": low, "ci_high": high,
            "p_value": binomtest(successes, trials, 0.5).pvalue,
            "listeners": len({r["listener"] for r in responses
                              if r["block"] == block and checkpoint in ("all", r["checkpoint"])}),
        })
    return rows


def analyze(results_dir=RESULTS_DIR):
    responses = []
    for path in sorted((Path(results_dir) / "responses").glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as file:
            responses.extend(csv.DictReader(file))
    if not responses:
        print(f"No response files in {Path(results_dir) / 'responses'}")
        return []

    rows = summarize(responses)
    with open(Path(results_dir) / "summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(responses)} answers from {len({r['listener'] for r in responses})} listeners")
    print(f"{'block':6s}{'checkpoint':11s}{'answers':>8s}{'chose req.':>11s}{'prop.':>7s}{'95% CI':>16s}{'p':>9s}")
    for row in rows:
        print(f"{row['block']:6s}{row['checkpoint']:11s}{row['answers']:8d}{row['chose_requested']:11d}"
              f"{row['proportion']:7.2f}   [{row['ci_low']:.2f}, {row['ci_high']:.2f}]{row['p_value']:9.3f}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "analyze"))
    args = parser.parse_args(argv)

    if args.command == "build":
        pairs = select_pairs()
        write_pairs_csv(pairs, RESULTS_DIR / "pairs.csv")
        path = build_page(pairs)
        print(f"{len(pairs)} pairs -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"Pair list -> {RESULTS_DIR / 'pairs.csv'}")
    else:
        analyze()


if __name__ == "__main__":
    sys.exit(main())
