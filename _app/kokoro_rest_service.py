"""Local Kokoro-RU REST service.

The model and the RUAccent frontend are downloaded lazily from Hugging Face.
The service intentionally exposes the same /getwav contract as the existing
CosyVoice client, so chapter splitting, pauses and fingerprints are reused.
"""
from __future__ import annotations

import importlib.util
import io
import os
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

SERVICE_VERSION = "2026-11-10.11-gui"
MODEL_REPO = os.environ.get("KOKORO_MODEL_REPO", "zaakirio/kokoro-ru")
PORT = int(os.environ.get("KOKORO_PORT", "5015"))
VOICES = {"sveta": "female", "masha": "female", "dima": "male"}

app = FastAPI(title="Kokoro-RU REST", version=SERVICE_VERSION)
_models: dict = {}
_g2p = None
_model_snapshot: Path | None = None
_lock = threading.Lock()


def _cached_snapshot(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo, allow_patterns=patterns))


def _prepare_ruaccent_assets() -> None:
    """Materialize RUAccent's Hugging Face snapshot beside its Python package."""
    import ruaccent

    package_dir = Path(ruaccent.__file__).resolve().parent
    marker = package_dir / "dictionary" / "accents_nn.json.gz"
    if marker.exists() and (package_dir / "koziev").exists():
        return
    snapshot = _cached_snapshot("ruaccent/accentuator", ["*"])
    shutil.copytree(snapshot, package_dir, dirs_exist_ok=True)


def _load_g2p():
    global _g2p, _model_snapshot
    if _g2p is not None:
        return _g2p
    snapshot = _cached_snapshot(MODEL_REPO, [
        "ru_g2p.py", "kokoro-config.json", "espeak-data/*",
        "kokoro-ru-v2-base.pth", "kokoro-ru-v2-dima.pth",
    ])
    _model_snapshot = snapshot
    path = snapshot / "ru_g2p.py"
    spec = importlib.util.spec_from_file_location("kokoro_ru_g2p", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    _prepare_ruaccent_assets()
    source = path.read_text(encoding="utf-8")
    source = source.replace("read_text())", "read_text(encoding='utf-8'))")
    exec(compile(source, str(path), "exec"), module.__dict__)
    _g2p = module.RuG2P(
        espeak_data=snapshot / "espeak-data",
        vocab_path=snapshot / "kokoro-config.json",
    )
    return _g2p


def _load_voice(voice: str):
    voice = voice if voice in VOICES else "sveta"
    if voice in _models:
        return _models[voice]
    with _lock:
        if voice in _models:
            return _models[voice]
        import torch
        from kokoro import KModel
        checkpoint = "kokoro-ru-v2-dima.pth" if voice == "dima" else "kokoro-ru-v2-base.pth"
        if _model_snapshot is None:
            _load_g2p()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = KModel(repo_id=MODEL_REPO,
                       config=str(_model_snapshot / "kokoro-config.json"),
                       model=str(_model_snapshot / checkpoint)).to(device).eval()
        voice_path = _model_snapshot / "voices" / f"{voice}.pt"
        if not voice_path.exists():
            voice_path = _model_snapshot / "voices" / f"{voice}.bin"
        pack = torch.load(voice_path,
                          map_location=device, weights_only=False)
        _models[voice] = (model, pack)
        return _models[voice]


@app.get("/health")
def health():
    import torch
    return {"ok": True, "model_loaded": bool(_models), "lazy_load": True,
            "engine": "kokoro-ru", "voices": list(VOICES),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "service_version": SERVICE_VERSION}


def _resample(wav: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or wav.size < 2:
        return wav
    n = max(1, round(wav.size * target_rate / source_rate))
    return np.interp(np.linspace(0, wav.size - 1, n), np.arange(wav.size), wav).astype(np.float32)


def _synthesize(text: str, voice: str, sample_rate: int) -> bytes:
    if not text.strip():
        raise HTTPException(400, "text is empty")
    try:
        import torch
        g2p = _load_g2p()
        model, pack = _load_voice(voice)
        ipa, oov = g2p(text)
        if not ipa:
            raise ValueError("Kokoro-RU produced no phonemes")
        ref_s = pack[len(ipa) - 1]
        if hasattr(ref_s, "to"):
            ref_s = ref_s.to(model.device)
        with torch.no_grad():
            audio = model(ipa, ref_s, 1.0, return_output=True).audio
        wav = _resample(audio.detach().cpu().numpy().astype(np.float32), 24000, sample_rate)
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/getwav")
def getwav(text_to_speech: str, sample_rate: int = 24000, voice: str = "sveta"):
    return Response(_synthesize(text_to_speech, voice, int(sample_rate)), media_type="audio/wav")


@app.post("/synthesize")
def synthesize(payload: dict):
    return Response(_synthesize(payload.get("text", ""), payload.get("voice", "sveta"),
                                 int(payload.get("sample_rate", 24000))), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
