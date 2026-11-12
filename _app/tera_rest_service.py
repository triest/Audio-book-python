"""REST wrapper for TeraTTSv2 Russian ONNX TTS."""
import io
import os
import threading
from typing import Optional
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response

SERVICE_VERSION = "2026-11-10.1-gui"
PORT = int(os.environ.get("TERA_PORT", "5016"))
MODEL_ID = os.environ.get("TERA_MODEL_ID", "TeraSpace/TeraTTSv2")
VOICES = ["ru_f1", "ru_m5", "ru_f2", "ru_m1"]
app = FastAPI(title="TeraTTSv2 REST")
_model = None
_provider = None
_load_error: Optional[str] = None
_lock = threading.Lock()

def _load_model():
    global _model, _provider, _load_error
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            import onnxruntime as ort
            from transformers import AutoModel
            providers = ort.get_available_providers()
            _provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
            if _provider != "CUDAExecutionProvider" and os.environ.get("TERA_REQUIRE_GPU", "1") == "1":
                raise RuntimeError(f"CUDAExecutionProvider is unavailable; providers={providers}")
            _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, provider=_provider,
                                               threads=int(os.environ.get("TERA_THREADS", "6")))
            _load_error = None
            return _model
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            raise

def _ru_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text_to_speech is empty")
    return text if "<ru>" in text else f"<ru>{text}</ru>"

def _wav_bytes(audio, sample_rate: int) -> bytes:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != 44100 and len(data):
        new_len = max(1, round(len(data) * sample_rate / 44100))
        data = np.interp(np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data).astype(np.float32)
    out = io.BytesIO()
    sf.write(out, np.clip(data, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()

@app.get("/health")
def health():
    return {"status": "ok", "service": "terattsv2", "engine": "terattsv2", "service_version": SERVICE_VERSION,
            "model_loaded": _model is not None, "lazy_load": True, "provider": _provider,
            "voices": VOICES, "load_error": _load_error}

@app.get("/voices")
def voices():
    return {"voices": VOICES}

@app.get("/getwav")
def getwav(text_to_speech: str = Query(...), voice: str = "ru_f1", sample_rate: int = 44100):
    if voice not in VOICES:
        raise HTTPException(status_code=400, detail=f"unknown voice: {voice}")
    try:
        audio = _load_model().generate_speech(_ru_text(text_to_speech), voice=voice, duration_scale=1)
        return Response(content=_wav_bytes(audio, sample_rate), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
