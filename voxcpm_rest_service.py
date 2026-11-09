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

SERVICE_VERSION = "2026-11-09.1"
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
    return {"ok": True, "model_loaded": _model is not None, "model": MODEL_ID,
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
        buf = io.BytesIO()
        sf.write(buf, wav, int(model.tts_model.sample_rate), format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
