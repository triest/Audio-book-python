"""Optional local VoxCPM2 REST service.

Runs in .venv_voxcpm and keeps VoxCPM2 isolated from Silero, F5-TTS and ESpeech.
The model is loaded lazily on the first synthesis request.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

SERVICE_VERSION = "2026-11-10.2-gui"
MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
PORT = int(os.environ.get("VOXCPM_PORT", "5014"))

app = FastAPI(title="VoxCPM2 REST", version=SERVICE_VERSION)
_model = None
_lock = threading.Lock()


class SynthesisRequest(BaseModel):
    text: str
    reference_audio: str = ""
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    sample_rate: int = 48000


def _resample(wav: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or wav.size < 2:
        return wav
    n = max(1, round(wav.size * target_rate / source_rate))
    return np.interp(np.linspace(0, wav.size - 1, n), np.arange(wav.size), wav).astype(np.float32)


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from voxcpm import VoxCPM
                _model = VoxCPM.from_pretrained(MODEL_ID, device="auto", load_denoiser=False)
    return _model


@app.get("/health")
def health():
    import torch
    return {"ok": True, "model_loaded": _model is not None, "model": MODEL_ID,
            "lazy_load": True, "engine": "voxcpm2",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "service_version": SERVICE_VERSION}


@app.post("/synthesize")
def synthesize(req: SynthesisRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    try:
        model = _get_model()
        kwargs = dict(text=req.text, cfg_value=req.cfg_value,
                      inference_timesteps=req.inference_timesteps)
        if req.reference_audio:
            ref = Path(req.reference_audio).resolve()
            if not ref.exists():
                raise HTTPException(400, f"reference audio not found: {ref}")
            kwargs["prompt_wav_path"] = str(ref)
        wav = np.asarray(model.generate(**kwargs), dtype=np.float32)
        source_rate = int(model.tts_model.sample_rate)
        wav = _resample(wav, source_rate, int(req.sample_rate))
        buf = io.BytesIO()
        sf.write(buf, wav, int(req.sample_rate), format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/getwav")
def getwav(text_to_speech: str, sample_rate: int = 48000, voice: str = "default"):
    """Compatibility endpoint used by the audiobook orchestrator."""
    return synthesize(SynthesisRequest(text=text_to_speech, sample_rate=sample_rate))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
