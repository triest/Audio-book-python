r"""ASR of finished chapters with openai-whisper on CUDA (works in .venv_cosyvoice3, where faster-whisper in .venv has no cuBLAS / hits
mkl_malloc). Writes the same cache as chapter_asr_check.py (<outdir>\_asr\chN.json), so afterwards the normal check runs from the cache:

  .venv_cosyvoice3\Scripts\python.exe tools\asr_cache_cuda.py audiobook_output5 28 29 [--model small]
  .venv\Scripts\python.exe tools\chapter_asr_check.py audiobook_output5 28 29 --watch слово1,слово2

Chapter number N = "Глава N" (preface = 0); the file is NNN_*.wav with NNN = N + 1. Run it when the Silero service is NOT synthesizing."""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np


def load_16k(wav):
    with wave.open(str(wav)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128.0) / 128.0 if sw == 1 else a / float(2 ** (8 * sw - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr % 16000 == 0:
        k = sr // 16000
        a = a[: len(a) // k * k].reshape(-1, k).mean(axis=1)
    elif sr != 16000:
        a = np.interp(np.arange(0, len(a) / sr, 1 / 16000), np.arange(len(a)) / sr, a)
    return a.astype(np.float32)


def parse_range(tokens):
    nums = [int(t) for t in tokens]
    return list(range(nums[0], nums[-1] + 1)) if len(nums) == 2 else nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("chapters", nargs="+", help="N or 'FROM TO'")
    ap.add_argument("--model", default="small")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    import torch
    import whisper
    out = Path(a.outdir)
    asr = out / "_asr"
    asr.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(a.model, device=dev)
    for n in parse_range(a.chapters):
        wavs = sorted(out.glob(f"{n + 1:03d}_*.wav"))
        if not wavs:
            print(f"ch{n}: no {n + 1:03d}_*.wav", file=sys.stderr)
            continue
        wav, cache = wavs[0], asr / f"ch{n}.json"
        if cache.exists() and not a.redo and cache.stat().st_mtime >= wav.stat().st_mtime:
            print(f"ch{n}: cache is fresh, skip", flush=True)
            continue
        print(f"ch{n}: transcribing {wav.name} ({a.model}, {dev}) ...", flush=True)
        r = model.transcribe(load_16k(wav), language="ru", word_timestamps=True, temperature=0.0,
                             condition_on_previous_text=False, fp16=(dev == "cuda"), verbose=False)
        words = [{"w": w["word"].strip(), "s": round(w["start"], 2), "e": round(w["end"], 2)}
                 for sg in r["segments"] for w in sg.get("words", [])]
        cache.write_text(json.dumps({"file": str(wav), "model": a.model, "words": words}, ensure_ascii=False), encoding="utf-8")
        print(f"ch{n}: {len(words)} words -> {cache}", flush=True)


if __name__ == "__main__":
    main()
