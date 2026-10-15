r"""Votes for the stressed syllable of names in narrator audio (uses narrator_asr/*.json timestamps + acoustic_stress.py).
  .venv_cosyvoice3\Scripts\python.exe narrator_name_stress.py ASR_DIR OUT.json
Only 2-syllable names are scored (validated 18/21 on known synthetic speech; 3+ syllables are near chance)."""
import glob, json, re, sys, collections
import numpy as np, soundfile as sf, librosa
sys.path.insert(0, ".")
from acoustic_stress import syllable_scores
TARGETS = {  # normalized ASR spelling -> (display, n syllables, our chain's stressed syllable index or None)
    "хонор": ("Хонор", 2, 0), "нимиц": ("Нимиц", 2, 0), "грейсон": ("Грейсон", 2, 0), "трумэн": ("Трумэн", 2, 1), "мэйхью": ("Мэйхью", 2, 0),
    "тревор": ("Тревор", 2, 0), "эстер": ("Эстер", 2, 1), "макквин": ("МакКвин", 2, 1), "оскар": ("Оскар", 2, 1), "томас": ("Томас", 2, 0),
    "девон": ("Девон", 2, 0), "мюллер": ("Мюллер", 2, 0), "хенке": ("Хенке", 2, 0), "кортес": ("Кортес", 2, 1), "мэтьюс": ("Мэтьюс", 2, None),
}
norm = lambda w: re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))
votes = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f, encoding="utf-8")); src = d["src"]
    if not src.lower().endswith(".mp3"): continue
    info = sf.info(src)
    for w in d["words"]:
        n = norm(w["w"])
        if n not in TARGETS: continue
        s0 = max(0.0, w["s"] - 0.2); e0 = w["e"] + 0.2
        x, sr = sf.read(src, start=int(s0 * info.samplerate), frames=int((e0 - s0) * info.samplerate), dtype="float32", always_2d=True)
        a = librosa.resample(x.mean(axis=1), orig_sr=sr, target_sr=16000).astype(np.float32)
        pred, _, conf = syllable_scores(a, w["s"] - s0, w["e"] - s0, TARGETS[n][1])
        if pred is not None: votes[n].append({"pred": pred, "t": round(w["s"], 1), "file": src.split("/")[-1].split("\\")[-1], "conf": conf})
out = {}
for n, v in sorted(votes.items(), key=lambda kv: -len(kv[1])):
    c = collections.Counter(x["pred"] for x in v)
    disp, nsy, ours = TARGETS[n]
    out[disp] = {"n": len(v), "votes_syllable": dict(c), "our_chain_syllable": ours, "samples": v[:5]}
    print(f"{disp:9s} n={len(v):3d} votes by syllable {dict(c)}  (our chain: syllable {ours})")
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
