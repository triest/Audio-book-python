r"""Acoustic estimate of the stressed syllable of a spoken word (energy + duration + f0 per syllable nucleus).
EXPERIMENTAL: only usable after validate() shows it recovers the stress of speech where it is KNOWN.
  .venv_cosyvoice3\Scripts\python.exe acoustic_stress.py validate CLIPDIR OUT.json   # clips from acoustic_stress_synth.py
Library: syllable_scores(audio16k, t0, t1, n_syll) -> (pred_index, per-syllable dict, confident)"""
import difflib, json, re, sys
import numpy as np
SR = 16000
VOW = "аеёиоуыэюя"


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-6 else np.zeros_like(v)


def syllable_scores(audio, t0, t1, n_syll, pad=0.03):
    import librosa
    from scipy.signal import find_peaks
    a0, a1 = max(0, int((t0 - pad) * SR)), min(len(audio), int((t1 + pad) * SR))
    x = audio[a0:a1]
    if len(x) < SR * 0.12 or n_syll < 1:
        return None, None, False
    hop = 160
    rms = librosa.feature.rms(y=x, frame_length=400, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-5)
    k = np.ones(3) / 3
    dbs = np.convolve(db, k, "same")
    f0, voiced, _ = librosa.pyin(x, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)
    n = len(dbs)
    if n_syll == 1:
        return 0, {}, True
    peaks, props = find_peaks(dbs, prominence=2.0, distance=max(3, n // (n_syll * 3)))
    confident = len(peaks) >= n_syll
    if len(peaks) > n_syll:
        keep = np.argsort(props["prominences"])[::-1][:n_syll]
        peaks = np.sort(peaks[keep])
    if len(peaks) < n_syll:                                    # fall back to even split of the voiced core
        lo, hi = int(n * 0.05), int(n * 0.95)
        peaks = np.array([int(lo + (i + 0.5) * (hi - lo) / n_syll) for i in range(n_syll)])
    bounds = [0] + [int(peaks[i] + np.argmin(dbs[peaks[i]:peaks[i + 1] + 1])) for i in range(n_syll - 1)] + [n]
    en, du, fm = [], [], []
    for i in range(n_syll):
        s, e = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        en.append(float(dbs[peaks[i]]))
        du.append((e - s) * hop / SR)
        seg = f0[s:e]
        fm.append(12 * np.log2(np.nanmax(seg) / 100.0) if np.isfinite(seg).any() else np.nan)
    fm = np.array(fm)
    if np.isnan(fm).all():
        fm = np.zeros(n_syll)
    fm = np.where(np.isnan(fm), np.nanmin(fm), fm)
    score = _z(en) + _z(du) + _z(fm)
    return int(np.argmax(score)), {"energy_db": en, "dur_s": du, "f0_semitones": fm.tolist(), "score": score.tolist()}, bool(confident)


def locate_word(asr_words, target):
    """Index of the ASR word best matching `target` (by letters)."""
    norm = lambda w: re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))
    best, bi = 0, None
    for i, w in enumerate(asr_words):
        r = difflib.SequenceMatcher(None, norm(w["w"]), norm(target)).ratio()
        if r > best:
            best, bi = r, i
    return (bi, best) if best >= 0.75 else (None, best)


def validate(clipdir, out):
    import librosa, whisper
    from pathlib import Path
    meta = json.load(open(Path(clipdir) / "meta.json", encoding="utf-8"))
    model = whisper.load_model("small", device="cuda")
    rows = []
    for m in meta:
        audio, _ = librosa.load(str(Path(clipdir) / m["file"]), sr=SR, mono=True)
        r = model.transcribe(audio, language="ru", word_timestamps=True, temperature=0.0, condition_on_previous_text=False, fp16=True)
        words = [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]} for sg in r["segments"] for w in sg.get("words", [])]
        i, ratio = locate_word(words, m["word"])
        if i is None:
            rows.append({**m, "pred": None, "why": "word not located", "asr": " ".join(w["w"] for w in words)}); continue
        pred, info, conf = syllable_scores(audio, words[i]["s"], words[i]["e"], m["n_syll"])
        rows.append({**m, "pred": pred, "confident": conf, "ratio": round(ratio, 2)})
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in rows if r["pred"] is not None]
    multi = [r for r in ok if r["n_syll"] > 1]
    hit = sum(r["pred"] == r["true_syllable"] for r in multi)
    chance = sum(1 / r["n_syll"] for r in multi)
    print(f"located {len(ok)}/{len(rows)}; multi-syllable accuracy {hit}/{len(multi)} = {hit/len(multi):.1%} (chance {chance/len(multi):.1%})")
    conf = [r for r in multi if r.get("confident")]
    if conf:
        h2 = sum(r["pred"] == r["true_syllable"] for r in conf)
        print(f"  among 'confident' (>= n peaks found): {h2}/{len(conf)} = {h2/len(conf):.1%}")
    for n in sorted({r['n_syll'] for r in multi}):
        sub = [r for r in multi if r['n_syll'] == n]
        print(f"  {n} syllables: {sum(r['pred']==r['true_syllable'] for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2], sys.argv[3])
