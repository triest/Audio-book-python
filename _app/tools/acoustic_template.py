r"""Template-matching stress estimate: compare the prosody contour (energy + f0, DTW) of a spoken word with Silero renderings of
the same word stressed on every syllable. EXPERIMENTAL - validate() first.
  .venv_cosyvoice3\Scripts\python.exe acoustic_template.py validate TEST_CLIPDIR TEMPLATE_CLIPDIR OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
SR = 16000


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-6 else np.zeros_like(v)


def contour(a):
    import librosa
    hop = 160
    rms = librosa.feature.rms(y=a, frame_length=400, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-5)
    keep = np.where(db > db.max() - 28)[0]
    if len(keep) < 8:
        return None
    lo, hi = keep[0], keep[-1] + 1
    f0, voiced, _ = librosa.pyin(a, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)[:len(db)]
    f0 = f0[lo:hi]; db = db[lo:hi]
    if np.isfinite(f0).sum() < 4:
        st = np.zeros(len(db))
    else:
        idx = np.arange(len(f0)); ok = np.isfinite(f0)
        f0 = np.interp(idx, idx[ok], f0[ok])
        st = 12 * np.log2(f0 / np.median(f0))
    return np.vstack([_z(np.convolve(db, np.ones(3) / 3, "same")), _z(np.convolve(st, np.ones(5) / 5, "same"))])


def dtw_cost(X, Y):
    import librosa
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def best_template(seg_contour, templates):
    """templates: list of contours, index = stressed syllable. Returns (argmin, costs)."""
    costs = [dtw_cost(seg_contour, t) if t is not None else 1e9 for t in templates]
    return int(np.argmin(costs)), costs


def validate(testdir, tmpldir, out):
    import librosa, whisper
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from acoustic_stress import locate_word
    meta = json.load(open(Path(testdir) / "meta.json", encoding="utf-8"))
    tmeta = json.load(open(Path(tmpldir) / "meta.json", encoding="utf-8"))
    templ = {}
    for m in tmeta:
        a, _ = librosa.load(str(Path(tmpldir) / m["file"]), sr=SR, mono=True)
        templ.setdefault(m["word"], {})[m["true_syllable"]] = contour(a)
    model = whisper.load_model("small", device="cuda")
    rows = []
    for m in meta:
        a, _ = librosa.load(str(Path(testdir) / m["file"]), sr=SR, mono=True)
        r = model.transcribe(a, language="ru", word_timestamps=True, temperature=0.0, condition_on_previous_text=False, fp16=True)
        words = [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]} for sg in r["segments"] for w in sg.get("words", [])]
        i, _ = locate_word(words, m["word"])
        if i is None:
            rows.append({**m, "pred": None}); continue
        seg = a[max(0, int((words[i]["s"] - 0.05) * SR)): int((words[i]["e"] + 0.05) * SR)]
        c = contour(seg)
        if c is None:
            rows.append({**m, "pred": None}); continue
        tl = [templ[m["word"]].get(k) for k in range(m["n_syll"])]
        pred, costs = best_template(c, tl)
        rows.append({**m, "pred": pred, "costs": [round(x, 3) for x in costs]})
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in rows if r["pred"] is not None and r["n_syll"] > 1]
    hit = sum(r["pred"] == r["true_syllable"] for r in ok)
    chance = sum(1 / r["n_syll"] for r in ok) / len(ok)
    print(f"template DTW: {hit}/{len(ok)} = {hit/len(ok):.1%} (chance {chance:.1%})")
    for n in sorted({r['n_syll'] for r in ok}):
        sub = [r for r in ok if r['n_syll'] == n]
        print(f"  {n} syllables: {sum(r['pred']==r['true_syllable'] for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2], sys.argv[3], sys.argv[4])
