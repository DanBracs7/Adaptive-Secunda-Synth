"""
Review page for the demo bank: listen to every passage of
configs/demo_prompts.csv under each bank setting (outputs/demo_bank/<setting>/,
written by `python src/demo_clips.py bank`) and choose by ear the setting
and the clips of the demo.

For every passage the page shows, for each setting, the generated clip
alone (what the demo would play) and the same clip after its real prompt
(to judge how the continuation follows it). A folder written with
`demo_clips.py bank --loop` shows its loops (short takes joined, from
loops.csv) instead of the single takes. A checkbox marks a clip for
the demo and a short note records why. The choices are kept in the
browser while you listen and are exported as demo_selection.csv
(setting, clip, label, title, prompt_start, keep, note), which the demo
page script reads.

The page links the WAV files next to it instead of embedding them, so it
works only inside outputs/demo_bank/. It plays real game music (the
prompts), so it stays local and is never published.

Run from the repository root:
    python src/demo_review.py          # outputs/demo_bank/review.html
"""

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = PROJECT_ROOT / "outputs" / "demo_bank"
PAGE_NAME = "review.html"
LABEL_ORDER = ("chill", "fight")


# ── Bank ──

def passage_id(clip):
    """Clip id without the take or loop suffix: the same passage in every folder."""
    return re.sub(r"_(t\d+|loop)$", "", clip)


def read_bank(bank_dir=BANK_DIR):
    """
    Read every folder of outputs/demo_bank/: its loops.csv when present,
    otherwise its clips.csv.
    Returns (settings, passages):
      settings  {folder: temperature string}, in folder-name order;
      passages  list of dicts, one per passage (label, title, prompt_start,
                id, per_setting {folder: [rows]}), chill first, then fight,
                each group by title and prompt start.
    """
    bank_dir = Path(bank_dir)
    settings, passages = {}, {}
    for folder in sorted(path for path in bank_dir.iterdir() if path.is_dir()):
        table = folder / "loops.csv" if (folder / "loops.csv").exists() else folder / "clips.csv"
        if not table.exists():
            continue
        with open(table, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        settings[folder.name] = rows[0]["temperature"]
        for row in rows:
            key = passage_id(row["clip"])
            passage = passages.setdefault(key, {
                "id": key, "label": row["label"], "title": row["title"],
                "prompt_start": row["prompt_start"], "per_setting": {}})
            passage["per_setting"].setdefault(folder.name, []).append(row)
    if not settings:
        raise FileNotFoundError(f"No clips.csv or loops.csv in {bank_dir}: run `python src/demo_clips.py bank` first")

    def sort_key(passage):
        minutes, seconds = passage["prompt_start"].split(":")
        label_rank = LABEL_ORDER.index(passage["label"]) if passage["label"] in LABEL_ORDER else len(LABEL_ORDER)
        return label_rank, passage["title"], int(minutes) * 60 + int(seconds)

    return settings, sorted(passages.values(), key=sort_key)


def clip_files(setting, row):
    """Paths of the generated and the with-prompt WAV of a row, relative to the bank folder."""
    generated = row.get("generated_file") or f"{row['clip']}_generated.wav"
    with_prompt = row.get("with_prompt_file") or f"{row['clip']}_with_prompt.wav"
    return f"{setting}/{generated}", f"{setting}/{with_prompt}"


# ── Page ──

PAGE_TEMPLATE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo bank review</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deded8;--accent:#2f5d8a;--accent-fg:#fff;--keep:#e5f4e4}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#21211f;--line:#34342f;--accent:#7fb0e0;--accent-fg:#10151a;--keep:#1f3320}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 64px}h1{font-size:1.35rem;margin:0 0 6px}h2{font-size:1.1rem;margin:28px 0 8px}
p{margin:0 0 10px}.muted{color:var(--muted);font-size:.9rem}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:10px}
.head{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:8px}
.cell{border:1px solid var(--line);border-radius:8px;padding:8px 10px}.cell.keep{background:var(--keep)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 10px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}button.playing{border-color:var(--accent);font-weight:600}
input[type=text]{font:inherit;width:100%;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
label{cursor:pointer}.count{font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1>Demo bank: scelta dei clip</h1>
<p class="muted">Per ogni passaggio: <b>generato</b> = solo la parte generata (quello che andrebbe nella demo); <b>loop</b> = più take brevi dallo stesso prompt, unite con dissolvenze; <b>con prompt</b> = prompt reale + parte generata, per sentire se la continuazione segue il prompt. Spunta <b>tieni</b> sui clip da usare nella demo. Le scelte restano salvate in questo browser; alla fine premi <b>Esporta CSV</b> e salva il file come <code>outputs/demo_bank/demo_selection.csv</code>.</p>
<p class="muted">Impostazioni: __SETTINGS__</p>
<div class="bar"><button class="primary" id="export">Esporta CSV</button><span class="count" id="counts"></span><button id="stop">&#9632; Stop</button></div>
__BODY__
<audio id="player"></audio>
</main>
<script>
const CLIPS = __CLIPS__;
const STORE = "secunda-demo-review";
const $ = id => document.getElementById(id);
const player = $("player");
let state = {};
try { state = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} };
let playing = null;

function play(button) {
  if (playing === button) { player.pause(); button.classList.remove("playing"); playing = null; return; }
  if (playing) playing.classList.remove("playing");
  player.src = button.dataset.src;
  player.play().catch(() => {});
  button.classList.add("playing"); playing = button;
}
player.onended = () => { if (playing) playing.classList.remove("playing"); playing = null; };
$("stop").onclick = () => { player.pause(); if (playing) playing.classList.remove("playing"); playing = null; };

function updateCounts() {
  const counts = {};
  for (const c of CLIPS) if (state[c.key]?.keep) {
    counts[c.setting] = counts[c.setting] || {chill: 0, fight: 0};
    counts[c.setting][c.label] = (counts[c.setting][c.label] || 0) + 1;
  }
  $("counts").textContent = Object.keys(counts).length
    ? "Tenuti: " + Object.entries(counts).map(([s, n]) => `${s} ${n.chill || 0} chill / ${n.fight || 0} fight`).join(" · ")
    : "Nessun clip tenuto";
}

for (const c of CLIPS) {
  const cell = document.querySelector(`[data-key="${c.key}"]`);
  const box = cell.querySelector("input[type=checkbox]");
  const note = cell.querySelector("input[type=text]");
  const saved = state[c.key] || {};
  box.checked = !!saved.keep; note.value = saved.note || "";
  cell.classList.toggle("keep", box.checked);
  box.onchange = () => { state[c.key] = {...(state[c.key] || {}), keep: box.checked}; cell.classList.toggle("keep", box.checked); save(); updateCounts(); };
  note.oninput = () => { state[c.key] = {...(state[c.key] || {}), note: note.value}; save(); };
  for (const button of cell.querySelectorAll("button[data-src]")) button.onclick = () => play(button);
}
updateCounts();

$("export").onclick = () => {
  const quote = v => /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const lines = ["setting,clip,label,title,prompt_start,keep,note"];
  for (const c of CLIPS) {
    const s = state[c.key] || {};
    lines.push([c.setting, c.clip, c.label, c.title, c.prompt_start, s.keep ? 1 : 0, s.note || ""].map(v => quote(String(v))).join(","));
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"}));
  link.download = "demo_selection.csv"; link.click();
};
</script></body></html>
"""


def build_page(settings, passages):
    """Return the review page as a string (audio linked by relative path)."""
    clips, sections = [], []
    for label in dict.fromkeys(p["label"] for p in passages):
        cards = []
        for passage in (p for p in passages if p["label"] == label):
            cells = []
            for setting in settings:
                setting_rows = passage["per_setting"].get(setting)
                if not setting_rows:
                    cells.append(f'<div class="cell muted">{html.escape(setting)}: clip mancante</div>')
                    continue
                for row in setting_rows:
                    key = f"{setting}|{row['clip']}"
                    generated, with_prompt = clip_files(setting, row)
                    clips.append({"key": key, "setting": setting, "clip": row["clip"], "label": label,
                                  "title": passage["title"], "prompt_start": passage["prompt_start"]})
                    clipped = float(row.get("clipped_fraction") or 0.0)
                    details = f"RMS {row.get('rms_dbfs', '?')} dBFS" + (f", clipping {clipped:.1%}" if clipped > 0 else "")
                    if row.get("takes"):
                        play_label = f"&#9654; loop ({html.escape(row['takes'])} take, {html.escape(row.get('loop_seconds', '?'))} s)"
                    else:
                        play_label = "&#9654; generato"
                    name = html.escape(row["clip"].rsplit("_", 1)[-1])
                    cells.append(
                        f'<div class="cell" data-key="{html.escape(key)}">'
                        f'<b>{html.escape(setting)}</b> <span class="muted">{name} &middot; {html.escape(details)}</span>'
                        f'<div class="row"><button data-src="{html.escape(generated)}">{play_label}</button>'
                        f'<button data-src="{html.escape(with_prompt)}">&#9654; con prompt</button>'
                        f'<label><input type="checkbox"> tieni</label></div>'
                        f'<div class="row"><input type="text" placeholder="nota (facoltativa)"></div></div>')
            cards.append(
                f'<div class="card"><div class="head"><b>{html.escape(passage["title"])}</b>'
                f'<span class="muted">prompt da {html.escape(passage["prompt_start"])}</span></div>'
                f'<div class="grid">{"".join(cells)}</div></div>')
        count = len(cards)
        sections.append(f"<h2>{html.escape(label.capitalize())} ({count})</h2>" + "".join(cards))

    settings_text = "; ".join(f"<b>{html.escape(s)}</b>: temperatura {html.escape(t)}" for s, t in settings.items())
    return (PAGE_TEMPLATE.replace("__SETTINGS__", settings_text)
            .replace("__BODY__", "\n".join(sections))
            .replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    args = parser.parse_args(argv)

    settings, passages = read_bank(args.bank_dir)
    page_path = args.bank_dir / PAGE_NAME
    page_path.write_text(build_page(settings, passages), encoding="utf-8")
    labels = [p["label"] for p in passages]
    print(f"{len(passages)} passages ({labels.count('chill')} chill, {labels.count('fight')} fight), "
          f"settings: {', '.join(settings)}")
    print(f"Review page -> {page_path}")


if __name__ == "__main__":
    sys.exit(main())
