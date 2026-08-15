"""Build a self-contained HTML scorer for the item E blank sheet.

Reads reports/scoring_sheets/item_e_scoring_sheet_BLANK.csv and writes
reports/item_e_scorer.html, which sits alongside reports/strips/ and loads the
images by relative path. Opens offline in any browser.

The page shows one strip at a time, full width, with keyboard shortcuts, and
exports a CSV in exactly the blank sheet's format when finished. Progress is
kept in browser storage so a closed tab does not lose an hour of work.

Nothing here scores anything or interprets a score. It is a data-entry
convenience over the committed blank sheet, and the exported CSV is the
artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path  # noqa: E402

PAGE = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Item E scoring</title>
<style>
 :root { color-scheme: dark; }
 body { background:#141414; color:#e8e8e8; font:15px/1.5 system-ui,sans-serif;
        margin:0; padding:16px 20px; }
 header { display:flex; align-items:baseline; gap:18px; margin-bottom:10px; }
 h1 { font-size:16px; font-weight:600; margin:0; }
 .meta { color:#9a9a9a; font-size:13px; }
 .bar { height:4px; background:#2a2a2a; border-radius:2px; margin:8px 0 14px; }
 .bar div { height:100%; background:#6aa9ff; border-radius:2px; width:0; }
 .strip { background:#0c0c0c; border:1px solid #2a2a2a; border-radius:6px;
          padding:10px; overflow-x:auto; }
 .strip img { display:block; image-rendering:pixelated; max-width:100%; }
 .btns { display:flex; gap:10px; margin:14px 0 8px; flex-wrap:wrap; }
 button { background:#232323; color:#e8e8e8; border:1px solid #3a3a3a;
          border-radius:5px; padding:9px 14px; font-size:14px; cursor:pointer; }
 button:hover { background:#2e2e2e; }
 button.sel { background:#2c4a7c; border-color:#6aa9ff; }
 kbd { background:#333; border-radius:3px; padding:1px 5px; font-size:12px;
       color:#bbb; margin-right:6px; }
 .row { display:flex; gap:12px; align-items:center; margin:8px 0; }
 input[type=text], input[type=number] {
   background:#1c1c1c; color:#e8e8e8; border:1px solid #3a3a3a;
   border-radius:4px; padding:7px 9px; font-size:14px; }
 input[type=text] { flex:1; }
 .nav { display:flex; gap:10px; margin-top:16px; align-items:center; }
 .done { color:#7bd88f; }
 #warn { display:none; background:#5a1f1f; border:1px solid #a33; color:#ffd9d9;
         padding:10px 14px; border-radius:5px; margin-bottom:12px;
         font-weight:600; }
 .hint { color:#8a8a8a; font-size:13px; margin-top:14px; }
 #export { background:#2c4a7c; border-color:#6aa9ff; }
 input:focus { outline:2px solid #d08b3a; }
 #typing { display:none; background:#4a3410; border:1px solid #d08b3a;
           color:#ffe2b8; padding:7px 12px; border-radius:5px;
           margin:8px 0; font-size:13px; }
</style>
<div id="warn"></div>
<header>
  <h1>Item E crop-retention scoring</h1>
  <span class="meta" id="pos"></span>
  <span class="meta" id="left"></span>
</header>
<div class="bar"><div id="prog"></div></div>
<div class="strip"><img id="img" alt=""></div>
<div class="btns">
  <button data-v="usable"><kbd>1</kbd>usable</button>
  <button data-v="degrades"><kbd>2</kbd>degrades</button>
  <button data-v="not_usable"><kbd>3</kbd>not usable</button>
  <button data-v="cannot_tell"><kbd>4</kbd>cannot tell</button>
</div>
<div class="row">
  <label class="meta" id="ufLabel">usable frames</label>
  <input type="number" id="uf" min="0" step="1" style="width:90px">
  <span class="meta" id="ufOf"></span>
  <input type="text" id="note" placeholder="note (optional) — where did it fail?">
</div>
<div id="typing">Typing a note. Keyboard shortcuts are OFF. Press
  <kbd>Esc</kbd> or click the image to turn them back on.</div>
<div class="nav">
  <button id="prev"><kbd>&larr;</kbd>back</button>
  <button id="next">skip<kbd style="margin-left:6px">&rarr;</kbd></button>
  <button id="export">download CSV</button>
  <span class="meta" id="saved"></span>
</div>
<p class="hint">
  usable = identifiable in every frame &middot; degrades = identifiable at the
  centre frame but lost, clipped or unrecognisable in at least one other
  &middot; not usable = not identifiable even at the centre &middot; cannot
  tell = too few pixels or too ambiguous to judge.<br>
  For <b>degrades</b>, type how many frames are usable before pressing 2.
  Blur that still reads as a drone counts as usable. usable and not usable
  fill the count automatically.<br>
  Scoring a strip advances automatically. Progress is kept in this browser;
  download the CSV when finished.
</p>
<script>
const ROWS = __ROWS__;
const KEY = "ch4_item_e_scores_v1";
let state = {}, STORAGE_OK = false;
try {
  localStorage.setItem(KEY + "_probe", "1");
  STORAGE_OK = localStorage.getItem(KEY + "_probe") === "1";
  localStorage.removeItem(KEY + "_probe");
  if (STORAGE_OK) state = JSON.parse(localStorage.getItem(KEY) || "{}");
} catch (e) { STORAGE_OK = false; }
let i = 0;
while (i < ROWS.length && state[ROWS[i].strip_id]) i++;
if (i >= ROWS.length) i = 0;

const $ = id => document.getElementById(id);

let sinceExport = 0;

function save() {
  if (!STORAGE_OK) { $("saved").textContent = ""; return; }
  try { localStorage.setItem(KEY, JSON.stringify(state));
        $("saved").textContent = "progress saved"; }
  catch (e) { STORAGE_OK = false; showWarn(); }
}

function showWarn() {
  const w = $("warn");
  w.style.display = "block";
  w.textContent = "This browser will not save progress for a local file. "
    + "Nothing is being stored. Press \u201cdownload CSV\u201d every 20 "
    + "strips, or open this page in Firefox instead, or you will lose your "
    + "work if the tab closes.";
}

window.onbeforeunload = () => {
  if (Object.keys(state).length && sinceExport > 0) return "Unexported scores.";
};

function blurInputs() {
  if (document.activeElement && document.activeElement.tagName === "INPUT")
    document.activeElement.blur();
  $("typing").style.display = "none";
}

function render() {
  blurInputs();
  const r = ROWS[i], s = state[r.strip_id] || {};
  $("img").src = "strips/" + r.strip_id + ".png";
  $("pos").textContent = "row " + (i + 1) + " of " + ROWS.length;
  const done = Object.keys(state).length;
  $("left").textContent = done + " scored, " + (ROWS.length - done) + " left";
  $("prog").style.width = (100 * done / ROWS.length) + "%";
  $("uf").value = s.usable_frames || "";
  $("ufOf").textContent = r.T ? "of " + r.T : "";
  $("note").value = s.note || "";
  document.querySelectorAll(".btns button").forEach(b =>
    b.classList.toggle("sel", s.score === b.dataset.v));
}

function score(v) {
  const r = ROWS[i];
  let uf = $("uf").value;
  // Fill the obvious cases so only 'degrades' needs typing.
  if (uf === "") {
    if (v === "usable") uf = r.T;
    else if (v === "not_usable") uf = 0;
  }
  state[r.strip_id] = { score: v, usable_frames: uf, note: $("note").value };
  save();
  sinceExport++;
  if (!STORAGE_OK && sinceExport >= 20) {
    $("export").textContent = "download CSV  (" + sinceExport + " unexported)";
  }
  if (i < ROWS.length - 1) i++;
  render();
}

document.querySelectorAll(".btns button").forEach(b =>
  b.onclick = () => score(b.dataset.v));
$("prev").onclick = () => { if (i > 0) i--; render(); };
$("next").onclick = () => { if (i < ROWS.length - 1) i++; render(); };

document.addEventListener("focusin", ev => {
  if (ev.target.tagName === "INPUT") $("typing").style.display = "block";
});
document.addEventListener("focusout", () => {
  $("typing").style.display = "none";
});
$("img").onclick = blurInputs;

document.onkeydown = ev => {
  if (ev.target.tagName === "INPUT") {
    if (ev.key === "Escape" || ev.key === "Enter") { blurInputs(); }
    return;
  }
  const map = {1:"usable", 2:"degrades", 3:"not_usable", 4:"cannot_tell"};
  if (map[ev.key]) { score(map[ev.key]); ev.preventDefault(); }
  else if (ev.key === "ArrowLeft") { if (i > 0) i--; render(); }
  else if (ev.key === "ArrowRight") { if (i < ROWS.length - 1) i++; render(); }
};

$("export").onclick = () => {
  const esc = s => /[",\\n]/.test(s || "") ? '"' + String(s).replace(/"/g, '""') + '"'
                                          : (s || "");
  let out = "sheet_row,strip_id,score,usable_frames,note\\n";
  ROWS.forEach(r => {
    const s = state[r.strip_id] || {};
    out += [r.sheet_row, r.strip_id, esc(s.score),
            esc(s.usable_frames), esc(s.note)].join(",") + "\\n";
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type:"text/csv"}));
  a.download = "item_e_scoring_sheet_COMPLETED.csv";
  a.click();
  sinceExport = 0;
  $("export").textContent = "download CSV";
};

if (!STORAGE_OK) showWarn();
render();
</script>
"""

PAGE_F = PAGE.replace(
  "Item E crop-retention scoring", "Item F motion separability"
).replace("""<div class="btns">
  <button data-v="usable"><kbd>1</kbd>usable</button>
  <button data-v="degrades"><kbd>2</kbd>degrades</button>
  <button data-v="not_usable"><kbd>3</kbd>not usable</button>
  <button data-v="cannot_tell"><kbd>4</kbd>cannot tell</button>
</div>""", """<div class="btns">
  <button data-v="drone"><kbd>1</kbd>drone</button>
  <button data-v="background"><kbd>2</kbd>background</button>
  <button data-v="cannot_tell"><kbd>3</kbd>cannot tell</button>
</div>
<div class="meta" style="margin-top:4px">what decided your call?</div>
<div class="btns" id="cues">
  <button data-c="appearance"><kbd>a</kbd>appearance</button>
  <button data-c="motion"><kbd>m</kbd>motion</button>
  <button data-c="both"><kbd>b</kbd>both</button>
  <button data-c="neither"><kbd>n</kbd>neither</button>
</div>""").replace("""<div class="row">
  <label class="meta" id="ufLabel">usable frames</label>
  <input type="number" id="uf" min="0" step="1" style="width:90px">
  <span class="meta" id="ufOf"></span>
  <input type="text" id="note" placeholder="note (optional) \u2014 where did it fail?">
</div>""", """<div class="row">
  <input type="text" id="note" placeholder="note (optional)">
</div>""").replace("""  $("uf").value = s.usable_frames || "";
  $("ufOf").textContent = r.T ? "of " + r.T : "";""",
"""  document.querySelectorAll("#cues button").forEach(b =>
    b.classList.toggle("sel", s.cue === b.dataset.c));""").replace(
"""  let uf = $("uf").value;
  // Fill the obvious cases so only 'degrades' needs typing.
  if (uf === "") {
    if (v === "usable") uf = r.T;
    else if (v === "not_usable") uf = 0;
  }
  state[r.strip_id] = { score: v, usable_frames: uf, note: $("note").value };""",
"""  const prev = state[r.strip_id] || {};
  state[r.strip_id] = { call: v, cue: prev.cue || "", note: $("note").value };""").replace(
"""  let out = "sheet_row,strip_id,score,usable_frames,note\\n";
  ROWS.forEach(r => {
    const s = state[r.strip_id] || {};
    out += [r.sheet_row, r.strip_id, esc(s.score),
            esc(s.usable_frames), esc(s.note)].join(",") + "\\n";""",
"""  let out = "sheet_row,strip_id,call,cue,note\\n";
  ROWS.forEach(r => {
    const s = state[r.strip_id] || {};
    out += [r.sheet_row, r.strip_id, esc(s.call),
            esc(s.cue), esc(s.note)].join(",") + "\\n";""").replace(
'a.download = "item_e_scoring_sheet_COMPLETED.csv";',
'a.download = "item_f_scoring_sheet_COMPLETED.csv";').replace(
"""  const map = {1:"usable", 2:"degrades", 3:"not_usable", 4:"cannot_tell"};
  if (map[ev.key]) { score(map[ev.key]); ev.preventDefault(); }""",
"""  const cmap = {a:"appearance", m:"motion", b:"both", n:"neither"};
  if (cmap[ev.key]) {
    const r = ROWS[i], st = state[r.strip_id] || {};
    st.cue = cmap[ev.key]; state[r.strip_id] = st; save(); render();
    ev.preventDefault(); return;
  }
  const map = {1:"drone", 2:"background", 3:"cannot_tell"};
  if (map[ev.key]) { score(map[ev.key]); ev.preventDefault(); }""").replace(
"""document.querySelectorAll(".btns button").forEach(b =>
  b.onclick = () => score(b.dataset.v));""",
"""document.querySelectorAll(".btns button[data-v]").forEach(b =>
  b.onclick = () => score(b.dataset.v));
document.querySelectorAll("#cues button").forEach(b =>
  b.onclick = () => { const r = ROWS[i], st = state[r.strip_id] || {};
    st.cue = b.dataset.c; state[r.strip_id] = st; save(); render(); });""").replace(
"""  usable = identifiable in every frame &middot; degrades = identifiable at the
  centre frame but lost, clipped or unrecognisable in at least one other
  &middot; not usable = not identifiable even at the centre &middot; cannot
  tell = too few pixels or too ambiguous to judge.<br>
  For <b>degrades</b>, type how many frames are usable before pressing 2.
  Blur that still reads as a drone counts as usable. usable and not usable
  fill the count automatically.<br>""",
"""  Is this a drone or background clutter? Then: <b>what decided your call?</b>
  <b>appearance</b> if a single frame would have told you, <b>motion</b> if
  only the change across frames did, <b>both</b> if either alone sufficed,
  <b>neither</b> if you could not say. The question applies to background
  calls as much as drone calls.<br>
  Set the cue first (a / m / b / n), then the call (1 / 2 / 3), which
  advances.<br>""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ch4.yaml")
    ap.add_argument("--item", choices=["e", "f", "stride2"],
                    default="e")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out = repo_path(cfg, cfg["reports"]["dir"])
    it = args.item
    sheet = (repo_path(cfg, cfg["reports"]["scoring_sheet_dir"])
             / f"item_{it}_scoring_sheet_BLANK.csv")
    lines = [l for l in sheet.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    # T comes from the manifest so the page can show "of N" beside the count.
    man_item = "e" if it == "stride2" else it
    man = repo_path(cfg, cfg["reports"]["strip_manifest_dir"]) \
        / f"item_{man_item}_sampling_manifest.csv"
    T_by_id = {r["strip_id"]: int(r["T"]) for r in csv.DictReader(
        open(man, encoding="utf-8"))}
    dup = repo_path(cfg, cfg["reports"]["strip_manifest_dir"]) \
        / f"item_{it}_duplicate_map.csv"
    if dup.exists():
        for r in csv.DictReader(open(dup, encoding="utf-8")):
            T_by_id[r["repeat_id"]] = T_by_id.get(r["original_id"], 0)
    rows = [{"sheet_row": r["sheet_row"], "strip_id": r["strip_id"],
             "T": T_by_id.get(r["strip_id"], 0)}
            for r in csv.DictReader(lines)]

    page = (PAGE_F if it == "f" else PAGE).replace("__ROWS__",
                                                    json.dumps(rows))
    if it == "stride2":
        page = page.replace("Item E crop-retention scoring",
                            "Stride-2 pass, T=8 crop retention")
        page = page.replace("item_e_scoring_sheet_COMPLETED.csv",
                            "item_stride2_scoring_sheet_COMPLETED.csv")
        page = page.replace("ch4_item_e_scores_v1", "ch4_stride2_scores_v1")
    target = out / f"item_{it}_scorer.html"
    target.write_text(page, encoding="utf-8")

    print(f"{len(rows)} rows")
    print(f"written {target}")
    print("Open it in a browser. It loads images from reports/strips/, so "
          "keep it where it is.")


if __name__ == "__main__":
    main()
