r"""Automated checks of synthesized chapters (NOT a stress verifier). Usage:
  .venv\Scripts\python.exe audio_verify.py TESTDIR            # TESTDIR/<variant>/NNN_*.wav + TESTDIR/asr/<V>_chN.json + TESTDIR/text/chN_plain.txt
Per file: duration, silence share, longest silence, clipping; ASR-vs-source word alignment (coverage, missing runs >= 6 words,
repeated runs) and timestamps of the words in WATCH. Writes TESTDIR/verify.json."""
import difflib, json, re, statistics, sys, wave, zlib
from pathlib import Path
import numpy as np

T = Path(sys.argv[1])
CH = {2: 1, 3: 2}   # book index -> "Глава N"; ASR files are named by the 'ch' number in the file name
WATCH = ["транспортами", "транспортах", "транспорта", "транспортный", "основное", "основным", "переходной", "плена", "тканей",
         "четырёхсот", "равно", "натаном", "вильям", "рамиресом", "рамиресу", "бенджамина", "душу", "замок"]
MAXCH = int(sys.argv[3]) if len(sys.argv) > 3 else 99
norm = lambda w: re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))


def audio_stats(path):
    w = wave.open(str(path)); n = w.getnframes(); sr = w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    hop = sr // 50; rms = np.sqrt(np.convolve(x * x, np.ones(hop) / hop, "valid")[::hop])
    sil = rms < 0.005
    longest = cur = 0
    for s in sil:
        cur = cur + 1 if s else 0; longest = max(longest, cur)
    return {"seconds": round(n / sr, 1), "silence_share": round(float(sil.mean()), 3), "longest_silence_s": round(longest / 50, 2),
            "clipped_samples": int((np.abs(x) > 0.999).sum()), "peak": round(float(np.abs(x).max()), 3)}


def _lev(a, b):
    """Plain Levenshtein distance (used only on short mismatching word groups)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rates(src_words, asr_words, sm):
    """WER and CER of the ASR transcript against the source text, from the word alignment `sm`
    (CER: exact edit distance inside every mismatching group, whole length for pure inserts/deletes)."""
    w_err = c_err = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        w_err += max(i2 - i1, j2 - j1)
        c_err += _lev(" ".join(src_words[i1:i2]), " ".join(asr_words[j1:j2]))
    n_chars = sum(len(w) for w in src_words) + max(len(src_words) - 1, 0)
    return w_err / max(len(src_words), 1), c_err / max(n_chars, 1)


def loop_ratio(words, win=60, step=30):
    """Highest len/compressed ratio (zlib) over sliding windows of ASR words. Looping ASR output
    (a repeated phrase) compresses far better than speech. Uncalibrated heuristic: compare chapters."""
    best = 0.0
    for i in range(0, max(len(words) - win, 0) + 1, step):
        t = " ".join(words[i:i + win]).encode("utf-8")
        if len(t) > 100:
            best = max(best, len(t) / len(zlib.compress(t, 9)))
    return round(best, 2)


CER_LIMIT = 0.05   # chapters above this (or with missing runs / loops) are listed for re-synthesis
LOOP_LIMIT = 3.0   # natural prose of the book peaks at ~2.4 (measured on pepel-pobedy.fb2); a 4x phrase loop gives ~10


report = {}
VARS = sys.argv[2].split(",") if len(sys.argv) > 2 else None
for var_dir in sorted(p for p in T.iterdir() if p.is_dir() and p.name[:2] in ("A_", "B_", "C_", "D_", "E_", "F_") and (VARS is None or p.name[0] in VARS)):
    for wav in sorted(var_dir.glob("*.wav")):
        book_idx = int(wav.name[:3]); n = book_idx if var_dir.name[0] in "EF" else CH[book_idx]
        if book_idx > MAXCH: continue
        asr_path = T / "asr" / f"{var_dir.name[0]}_ch{n}.json"
        entry = {"file": str(wav), "bytes": wav.stat().st_size, **audio_stats(wav)}
        src_words = [norm(w) for w in (T / "text" / f"ch{book_idx}_plain.txt").read_text(encoding="utf-8").split()]
        src_words = [w for w in src_words if w][2:]          # the chapter title is repeated twice at the top of the text
        if asr_path.exists():
            asr = json.loads(asr_path.read_text(encoding="utf-8"))["words"]
            a = [norm(x["w"]) for x in asr]; keep = [i for i, w in enumerate(a) if w]; a = [a[i] for i in keep]; asr = [asr[i] for i in keep]
            sm = difflib.SequenceMatcher(None, src_words, a, autojunk=False)
            matched = sum(b.size for b in sm.get_matching_blocks())
            missing = [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("delete", "replace") and i2 - i1 >= 6 and j2 - j1 < (i2 - i1) / 2]
            extra = [(j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("insert", "replace") and j2 - j1 >= 6 and i2 - i1 < (j2 - j1) / 2]
            rep = sum(1 for i in range(len(a) - 8) if a[i:i + 4] == a[i + 4:i + 8])
            wer, cer = error_rates(src_words, a, sm)
            lr = loop_ratio(a)
            entry["asr"] = {"source_words": len(src_words), "asr_words": len(a), "coverage": round(matched / len(src_words), 3),
                            "missing_runs>=6": [{"src_words": f"{i1}-{i2}", "text": " ".join(src_words[i1:i2])[:90]} for i1, i2 in missing[:20]],
                            "extra_runs>=6": [{"asr_words": f"{j1}-{j2}", "t": asr[j1]["s"], "text": " ".join(a[j1:j2])[:90]} for j1, j2 in extra[:20]],
                            "immediate_4gram_repeats": rep,
                            "wer": round(wer, 4), "cer": round(cer, 4), "loop_ratio": lr,
                            "resynth": bool(cer > CER_LIMIT or missing or lr > LOOP_LIMIT)}
            stamps = {}
            for w in WATCH:
                stamps[w] = [x["s"] for x, nw in zip(asr, a) if nw == norm(w)][:6]
            entry["watch_timestamps_s"] = {k: v for k, v in stamps.items() if v}
        report[f"{var_dir.name}/{wav.stem}"] = entry
(T / ("verify_" + (sys.argv[2] if len(sys.argv) > 2 else "all") + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in report.items():
    a = v.get("asr", {})
    print(f"{k:40s} {v['seconds']/60:6.2f} min  silence {v['silence_share']:.1%} (max {v['longest_silence_s']}s)  clip {v['clipped_samples']}"
          f"  cer {a.get('cer', '-')}  wer {a.get('wer', '-')}  loop {a.get('loop_ratio', '-')}  {'RESYNTH' if a.get('resynth') else ''}"
          f"  asr coverage {a.get('coverage', '-')}  missing runs {len(a.get('missing_runs>=6', []))}  extra runs {len(a.get('extra_runs>=6', []))}  repeats {a.get('immediate_4gram_repeats', '-')}")
cers = [v["asr"]["cer"] for v in report.values() if "asr" in v]
if cers:
    bad = [k for k, v in report.items() if v.get("asr", {}).get("resynth")]
    print(f"\nchapters with ASR: {len(cers)}  median CER {statistics.median(cers):.4f}  max {max(cers):.4f}  "
          f"(differences below ~0.01 are noise: the ASR itself errs)")
    print("re-synthesize:", ", ".join(bad) if bad else "none")
