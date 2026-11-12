r"""Makes validation clips for acoustic_stress.py: each test word is synthesized with EVERY possible stressed syllable in a fixed
carrier sentence (Silero v5_5_ru, speaker xenia). The true stressed syllable is known from the '+' mark.
  .venv\Scripts\python.exe acoustic_stress_synth.py OUTDIR"""
import json, re, sys, wave
from pathlib import Path
import numpy as np, torch
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SPEAKER = sys.argv[2] if len(sys.argv) > 2 else "xenia"
ISOLATED = len(sys.argv) > 3 and sys.argv[3] == "isolated"
WORDS = ["замок", "мука", "хонор", "харрингтон", "нимиц", "грейсон", "мантикора", "тремэйн", "мэйхью", "турвиль", "алисон", "бенджамин",
         "капитан", "корабль", "здание", "начальник", "договор", "красивая", "девушка", "дорога", "правитель", "станция", "офицер", "командор",
         "адмирал", "эскадра", "трумэн", "кромарти", "лафолле", "василиск", "республика", "протектор", "миледи", "милорд", "леонард", "елизавета",
         "александер", "мюллер", "оскар", "парнелл"]
VOW = "аеёиоуыэюя"
imp = torch.package.PackageImporter(str((Path(__file__).resolve().parent.parent / "silero_model_v5_5_ru.pt")))
m = imp.load_pickle("tts_models", "model"); m.to("cpu")
meta = []
for w in WORDS:
    vpos = [i for i, ch in enumerate(w) if ch in VOW]
    for k, p in enumerate(vpos):
        marked = w[:p] + "+" + w[p:]
        text = marked + "." if ISOLATED else f"Он медленно сказал слово, {marked}, и замолчал."
        a = m.apply_tts(text=text, speaker=SPEAKER, sample_rate=48000, put_accent=False, put_yo=False).numpy()
        fn = f"{w}_{k}.wav"
        with wave.open(str(out / fn), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000); f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
        meta.append({"file": fn, "word": w, "true_syllable": k, "n_syll": len(vpos)})
json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(meta), "clips")
