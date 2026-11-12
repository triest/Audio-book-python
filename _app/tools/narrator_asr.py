r"""Transcribe a slice of an existing audiobook with word timestamps (Whisper small) for name/pronunciation research.
  .venv_cosyvoice3\Scripts\python.exe narrator_asr.py SRC START_S DUR_S OUT.json FFMPEG
Decodes through ffmpeg to 16 kHz mono; the original file is only read."""
import json, subprocess, sys, tempfile, os
import numpy as np, whisper
src, start, dur, out, ff = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
if src.lower().endswith(".mp3"):
    import soundfile as sf, librosa
    sr0 = sf.info(src).samplerate
    d, sr0 = sf.read(src, start=int(float(start) * sr0), frames=int(float(dur) * sr0), dtype="float32", always_2d=True)
    audio = librosa.resample(d.mean(axis=1), orig_sr=sr0, target_sr=16000).astype(np.float32)
else:
    raw = subprocess.run([ff, "-v", "error", "-ss", start, "-t", dur, "-i", src, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
if len(audio) < 16000:
    sys.exit("no audio decoded: " + src)
m = whisper.load_model("small", device="cuda")
r = m.transcribe(audio, language="ru", word_timestamps=True, condition_on_previous_text=False, temperature=0.0, fp16=True)
words = [{"w": w["word"].strip(), "s": round(w["start"] + float(start), 2), "e": round(w["end"] + float(start), 2)} for sg in r["segments"] for w in sg.get("words", [])]
json.dump({"src": src, "start": float(start), "dur": float(dur), "words": words}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(out, len(words), "words")
