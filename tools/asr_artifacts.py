r"""Detector of Silero sound artifacts (hum / stretched word / swallowed or looped text) over the ASR cache of finished chapters.

  .venv\Scripts\python.exe tools\asr_artifacts.py sample_book [0 49] [--book pepel-pobedy.fb2] [--long 1.5] [--csv out.csv]

Reads <outdir>\_asr\chN.json (made by tools\asr_cache_cuda.py). Per chapter prints rows "chapter / mm:ss / kind / heard / expected":
  long     a word whose ASR duration exceeds --long seconds (Whisper may stretch a word over a pause: check against the text)
  hum      an interjection (ммм, эээ, аа ...) that the source text does not have at that place
  missing  >= 6 source words absent from the audio (swallowed piece)
  extra    >= 6 ASR words absent from the text (garbage / loop)
  garble   a 3-8 word window where the audio differs from the text by more than half of the characters
ASR does not judge stress; it only finds places worth listening to."""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from chapter_asr_check import norm, lev, source_words  # noqa: E402

HUM_RE = re.compile(r"^(м+|э+м*|хм+|ам+|ы+)$")  # vowel-only a/у/о/е are ASR misreads of conjunctions, not hum


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def chapter_rows(n, words, src, long_s):
    rows = []
    kept = [(norm(x["w"]), x) for x in words if norm(x["w"])]
    asr = [k for k, _ in kept]
    sm = difflib.SequenceMatcher(None, src, asr, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        t = kept[min(j1, len(kept) - 1)][1]["s"] if kept else 0.0
        exp = " ".join(src[max(i1 - 2, 0):i2 + 2])
        heard = " ".join(asr[max(j1 - 2, 0):j2 + 2])
        for k in range(j1, j2):
            if HUM_RE.match(asr[k]) and not (i2 > i1 and any(HUM_RE.match(s) for s in src[i1:i2])):
                rows.append((n, kept[k][1]["s"], "hum", heard, exp))
        if tag in ("delete", "replace") and i2 - i1 >= 6 and j2 - j1 < (i2 - i1) / 2:
            rows.append((n, t, "missing", heard, " ".join(src[i1:i2])[:160]))
        elif tag in ("insert", "replace") and j2 - j1 >= 6 and i2 - i1 < (j2 - j1) / 2:
            rows.append((n, t, "extra", " ".join(asr[j1:j2])[:160], exp))
        elif tag == "replace" and 3 <= i2 - i1 <= 8 and j2 - j1 >= 2:
            a, b = " ".join(src[i1:i2]), " ".join(asr[j1:j2])
            if lev(a, b) / max(len(a), len(b)) > 0.5:
                rows.append((n, t, "garble", b[:100], a[:100]))
    for k, (w, x) in enumerate(kept):
        d = x["e"] - x["s"]
        if d > long_s and len(w) <= 14:
            ctx = " ".join(asr[max(k - 3, 0):k + 4])
            rows.append((n, x["s"], "long", f"{w} ({d:.1f}s)", ctx))
    rows.sort(key=lambda r: r[1])
    return [r for i, r in enumerate(rows) if not (i and r[1:3] == rows[i - 1][1:3])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("chapters", nargs="*", type=int, help="first and last chapter (default: all cached)")
    ap.add_argument("--book", default="pepel-pobedy.fb2")
    ap.add_argument("--long", type=float, default=1.5)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    out = Path(a.outdir)
    out = out if out.is_absolute() else ROOT / out
    asr_dir = out / "_asr"
    avail = sorted(int(p.stem[2:]) for p in asr_dir.glob("ch*.json") if p.stem[2:].isdigit())
    if len(a.chapters) == 2:
        avail = [n for n in avail if a.chapters[0] <= n <= a.chapters[1]]
    elif a.chapters:
        avail = [n for n in avail if n in a.chapters]
    import fb2_reader as f
    _, chapters = f.parse_fb2(ROOT / a.book)
    allrows = []
    for n in avail:
        words = json.loads((asr_dir / f"ch{n}.json").read_text(encoding="utf-8"))["words"]
        rows = chapter_rows(n, words, source_words(*chapters[n]), a.long)
        allrows += rows
        print(f"ch{n}: {len(rows)} flagged")
        for r in rows:
            print(f"   {mmss(r[1]):>6}  {r[2]:<8} heard: {r[3]}   | text: {r[4]}")
    print(f"\nchapters checked: {len(avail)}   flagged places: {len(allrows)}")
    if a.csv:
        import csv
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["chapter", "time", "kind", "heard", "text"])
            for r in allrows:
                w.writerow([r[0], mmss(r[1]), r[2], r[3], r[4]])


if __name__ == "__main__":
    main()
