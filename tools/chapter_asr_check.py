r"""ASR check of FINISHED chapters (faster-whisper, installed by install.bat). Run from the project root:

  .venv\Scripts\python.exe tools\chapter_asr_check.py audiobook_output5 28 45
  .venv\Scripts\python.exe tools\chapter_asr_check.py audiobook_output5 28 45 --watch одержат,ввергла --model large-v3

Chapter number N = "Глава N" of the book (preface = 0); the file is NNN_*.wav with NNN = N + 1 (as the synthesis names it).
Per chapter: transcript with word timestamps (cached next to the wav in <outdir>\_asr\chN.json), coverage / WER / CER against the
source text (numbers -> words, footnote markers stripped, the same as the synthesis input), missing and extra runs >= 6 words,
a loop ratio, and the places of the --watch words with a short clip around each (<outdir>\_asr\clips\) for a quick listen.
ASR does NOT judge stress (it normalises words): it proves the text was spoken whole; stress is confirmed by ear only."""
import argparse
import difflib
import json
import re
import statistics
import sys
import wave
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CER_LIMIT = 0.05
LOOP_LIMIT = 3.0
norm = lambda w: re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compare(src, asr):
    """Word alignment of the source text against the ASR words -> metrics dict."""
    sm = difflib.SequenceMatcher(None, src, asr, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    w_err = c_err = 0
    missing, extra = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        w_err += max(i2 - i1, j2 - j1)
        c_err += lev(" ".join(src[i1:i2]), " ".join(asr[j1:j2]))
        if tag in ("delete", "replace") and i2 - i1 >= 6 and j2 - j1 < (i2 - i1) / 2:
            missing.append((i1, i2))
        if tag in ("insert", "replace") and j2 - j1 >= 6 and i2 - i1 < (j2 - j1) / 2:
            extra.append((j1, j2))
    n_chars = sum(len(w) for w in src) + max(len(src) - 1, 0)
    best = 0.0
    for i in range(0, max(len(asr) - 60, 0) + 1, 30):
        t = " ".join(asr[i:i + 60]).encode("utf-8")
        if len(t) > 100:
            best = max(best, len(t) / len(zlib.compress(t, 9)))
    cer = c_err / max(n_chars, 1)
    return {
        "source_words": len(src), "asr_words": len(asr), "coverage": round(matched / max(len(src), 1), 3),
        "wer": round(w_err / max(len(src), 1), 4), "cer": round(cer, 4), "loop_ratio": round(best, 2),
        "missing_runs": [" ".join(src[i1:i2])[:90] for i1, i2 in missing[:20]],
        "extra_runs": [" ".join(asr[j1:j2])[:90] for j1, j2 in extra[:20]],
        "resynth": bool(cer > CER_LIMIT or missing or best > LOOP_LIMIT),
    }


def source_words(title, text):
    import fb2_reader as f
    body = f.numbers_to_words_ru(f.strip_footnote_markers(text)) if hasattr(f, "strip_footnote_markers") else f.numbers_to_words_ru(text)
    return [w for w in (norm(x) for x in (title + "\n" + body).split()) if w]


def transcribe(wav, model_name):
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        segs, _ = model.transcribe(str(wav), language="ru", word_timestamps=True, temperature=0.0,
                                   condition_on_previous_text=False, vad_filter=False)
        return [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for s in segs for w in (s.words or [])]
    except Exception as e:                              # no CUDA / cuDNN libs: fall back to CPU
        print(f"  GPU unavailable ({type(e).__name__}: {str(e)[:80]}), using CPU", file=sys.stderr)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segs, _ = model.transcribe(str(wav), language="ru", word_timestamps=True, temperature=0.0,
                                   condition_on_previous_text=False, vad_filter=False)
        return [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for s in segs for w in (s.words or [])]


def cut_clip(wav, start, end, dst):
    with wave.open(str(wav)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(max(int(start * sr), 0))
        frames = w.readframes(int((end - start) * sr))
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(sr)
        o.writeframes(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("chapters", nargs="+", type=int)
    ap.add_argument("--book", default="pepel-pobedy.fb2")
    ap.add_argument("--model", default="medium", help="faster-whisper model (small/medium/large-v3), default medium")
    ap.add_argument("--watch", default="", help="comma-separated words: timestamps + short clips around each")
    ap.add_argument("--redo", action="store_true", help="ignore the cached transcript")
    a = ap.parse_args()

    import fb2_reader as f
    out = Path(a.outdir)
    out = out if out.is_absolute() else ROOT / out
    _, chapters = f.parse_fb2(ROOT / a.book)
    asr_dir = out / "_asr"
    (asr_dir / "clips").mkdir(parents=True, exist_ok=True)
    watch = [norm(w) for w in a.watch.split(",") if norm(w)]
    report, cers = {}, []
    for n in a.chapters:
        wavs = sorted(out.glob(f"{n + 1:03d}_*.wav"))
        if not wavs:
            print(f"ch{n}: no {n + 1:03d}_*.wav in {out}")
            continue
        wav = wavs[0]
        cache = asr_dir / f"ch{n}.json"
        if cache.exists() and not a.redo and cache.stat().st_mtime >= wav.stat().st_mtime:
            words = json.loads(cache.read_text(encoding="utf-8"))["words"]
        else:
            print(f"ch{n}: transcribing {wav.name} ({a.model}) ...", flush=True)
            words = transcribe(wav, a.model)
            cache.write_text(json.dumps({"file": str(wav), "model": a.model, "words": words}, ensure_ascii=False), encoding="utf-8")
        kept = [(norm(x["w"]), x) for x in words if norm(x["w"])]
        m = compare(source_words(*chapters[n]), [k for k, _ in kept])
        hits = []
        for w in watch:
            for k, x in kept:
                if k == w:
                    hits.append((w, x["s"]))
        m["watch"] = [{"word": w, "t": t} for w, t in hits]
        for i, (w, t) in enumerate(hits[:12]):
            clip = asr_dir / "clips" / f"ch{n}_{w}_{i + 1}_{t:.0f}s.wav"
            cut_clip(wav, t - 2.5, t + 3.0, clip)
            m.setdefault("clips", []).append(clip.name)
        report[f"ch{n}"] = m
        cers.append(m["cer"])
        print(f"ch{n}: coverage {m['coverage']}  WER {m['wer']}  CER {m['cer']}  loop {m['loop_ratio']}  "
              f"missing {len(m['missing_runs'])}  extra {len(m['extra_runs'])}  {'RESYNTH' if m['resynth'] else 'ok'}")
        for r in m["missing_runs"][:5]:
            print("   missing:", r)
        for r in m["extra_runs"][:5]:
            print("   extra:  ", r)
        if watch:
            print("   watch:", ", ".join(f"{w}@{t:.0f}s" for w, t in hits[:12]) or "not found")
    (asr_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if cers:
        print(f"\nchapters: {len(cers)}  median CER {statistics.median(cers):.4f}  (differences below ~0.01 are ASR noise)")
        print(f"report: {asr_dir / 'report.json'}   clips: {asr_dir / 'clips'}")
        print("ASR does not judge stress - listen to the clips for that.")


if __name__ == "__main__":
    main()
