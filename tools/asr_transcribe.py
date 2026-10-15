r"""Whisper (small) transcript with word timestamps for a synthesized WAV -> JSON. Run with the venv that has
openai-whisper:  .venv_cosyvoice\Scripts\python.exe asr_transcribe.py in.wav out.json
Evidence use only: completeness / repeats / timestamps. ASR cannot judge stress."""
import json
import sys

import librosa
import whisper

src, dst = sys.argv[1], sys.argv[2]
audio, _ = librosa.load(src, sr=16000, mono=True)
model = whisper.load_model("small", device="cuda")
res = model.transcribe(audio, language="ru", word_timestamps=True, condition_on_previous_text=False, fp16=True,
                       temperature=0.0)
words = [{"w": w["word"].strip(), "s": round(w["start"], 2), "e": round(w["end"], 2)}
         for seg in res["segments"] for w in seg.get("words", [])]
json.dump({"file": src, "duration": round(len(audio) / 16000, 1), "words": words}, open(dst, "w", encoding="utf-8"),
          ensure_ascii=False)
print(src, len(words), "words")
