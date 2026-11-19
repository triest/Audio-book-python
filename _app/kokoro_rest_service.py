"""Local Kokoro-RU REST service.

The model and the RUAccent frontend are downloaded lazily from Hugging Face.
The service intentionally exposes the same /getwav contract as the existing
CosyVoice client, so chapter splitting, pauses and fingerprints are reused.
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import sys
import threading
from pathlib import Path

SERVICE_VERSION = "2026-11-18.7-kokoro-init-progress"
MODEL_REPO = os.environ.get("KOKORO_MODEL_REPO", "zaakirio/kokoro-ru")
PORT = int(os.environ.get("KOKORO_PORT", "5015"))
VOICES = {"sveta": "female", "masha": "female", "dima": "male"}

# Prefer the Kokoro environment's ONNX Runtime, which can include CUDA. The
# CosyVoice3 runtime is only a fallback for a partially installed environment.
_shared_site = Path(__file__).resolve().parents[1] / ".venv_cosyvoice3" / "Lib" / "site-packages"
try:
    import numpy as np
    import onnxruntime as ort
except Exception as _kokoro_runtime_error:
    if not _shared_site.is_dir():
        raise
    _shared_site_text = str(_shared_site)
    sys.path.insert(0, _shared_site_text)
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        raise _kokoro_runtime_error
    finally:
        try:
            sys.path.remove(_shared_site_text)
        except ValueError:
            pass

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Kokoro-RU REST", version=SERVICE_VERSION)
_models: dict = {}
_g2p = None
_model_snapshot: Path | None = None
_runtime_device = "auto"
_lock = threading.Lock()


def _ruaccent_device() -> str:
    """Use CUDA for RUAccent's ONNX models when its CUDA provider is installed."""
    try:
        import torch
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            # ONNX Runtime reuses CUDA/cuDNN DLLs shipped with the installed PyTorch.
            preload = getattr(ort, "preload_dlls", None)
            if preload:
                preload()
            return "CUDA"
    except Exception as exc:
        print(f"RUAccent CUDA unavailable; using CPU: {exc}", flush=True)
    return "CPU"


def _cached_snapshot(repo: str, patterns: list[str]) -> Path:
    cache_base = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    )
    snapshot_root = cache_base / f"models--{repo.replace('/', '--')}" / "snapshots"
    if snapshot_root.is_dir():
        for snapshot in sorted(snapshot_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not snapshot.is_dir():
                continue
            complete = True
            for pattern in patterns:
                candidate = snapshot / pattern
                if pattern.endswith("/*"):
                    complete = complete and candidate.parent.is_dir() and any(candidate.parent.iterdir())
                elif pattern == "*":
                    complete = complete and (snapshot / "dictionary" / "accents_nn.json.gz").is_file()
                else:
                    complete = complete and candidate.exists()
            if complete:
                return snapshot
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo, allow_patterns=patterns))


def _prepare_ruaccent_assets() -> None:
    """Materialize RUAccent assets in site-packages.

    Hugging Face snapshots contain relative symlinks to their blob store. A
    directory junction from this venv to the snapshot changes the base path
    used to resolve those relative symlinks on Windows, so the files appear
    missing. Preserve complete local copies when present and dereference the
    snapshot links while copying otherwise.
    """
    import ruaccent

    package_dir = Path(ruaccent.__file__).resolve().parent
    marker = package_dir / "dictionary" / "accents_nn.json.gz"
    asset_names = ("dictionary", "koziev", "nn")
    if marker.is_file() and all((package_dir / name).is_dir() for name in asset_names):
        return

    snapshot = _cached_snapshot("ruaccent/accentuator", ["*"])
    for name in asset_names:
        source = snapshot / name
        target = package_dir / name
        if not source.is_dir():
            raise FileNotFoundError(f"RUAccent asset directory is missing: {source}")
        try:
            attributes = getattr(target.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            attributes = 0
        is_reparse_point = bool(attributes & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if is_reparse_point:
            if target.is_dir():
                target.rmdir()
            else:
                target.unlink()
        elif target.exists() and not target.is_dir():
            target.unlink()
        # copytree follows Hugging Face's blob symlinks by default, creating
        # ordinary files that remain readable from the venv on another drive.
        shutil.copytree(source, target, dirs_exist_ok=True)
    if not marker.is_file():
        raise FileNotFoundError(f"RUAccent accent dictionary is missing: {marker}")


def _load_g2p():
    global _g2p, _model_snapshot
    if _g2p is not None:
        return _g2p
    print("Kokoro-RU: locating model and RUAccent assets...", flush=True)
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
    accent_device = _ruaccent_device()
    print(f"Kokoro-RU: initializing RUAccent on {accent_device}...", flush=True)
    source = path.read_text(encoding="utf-8")
    source = source.replace("read_text())", "read_text(encoding='utf-8'))")
    source = source.replace(
        "use_dictionary=True, tiny_mode=False)",
        f"use_dictionary=True, tiny_mode=False, device={accent_device!r})",
    )
    exec(compile(source, str(path), "exec"), module.__dict__)
    _g2p = module.RuG2P(
        espeak_data=snapshot / "espeak-data",
        vocab_path=snapshot / "kokoro-config.json",
    )
    # Kokoro's RUAccent normally strips '+', the stress marker supplied by
    # silero-stress, before it can skip already-accented words. Preserve it so
    # RUAccent can still accent the remaining words while respecting overrides.
    accent = getattr(_g2p, "_accent", None)
    normalizer = getattr(accent, "normalize", None)
    pattern = getattr(normalizer, "pattern", None)
    if isinstance(pattern, str) and "+" not in pattern:
        patched_pattern = pattern.replace("ёЁ—", "ёЁ+—", 1)
        if patched_pattern != pattern:
            accent.normalize = re.compile(patched_pattern, normalizer.flags)
    print("Kokoro-RU: RUAccent and phonemizer ready.", flush=True)
    return _g2p


def _load_voice(voice: str):
    global _runtime_device
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
        _runtime_device = str(device)
        print(f"Kokoro-RU: loading {voice} voice on {_runtime_device}...", flush=True)
        model = KModel(repo_id=MODEL_REPO,
                       config=str(_model_snapshot / "kokoro-config.json"),
                       model=str(_model_snapshot / checkpoint)).to(device).eval()
        voice_path = _model_snapshot / "voices" / f"{voice}.pt"
        if not voice_path.exists():
            voice_path = _model_snapshot / "voices" / f"{voice}.bin"
        pack = torch.load(voice_path,
                          map_location=device, weights_only=False)
        _models[voice] = (model, pack)
        print(f"Kokoro-RU: voice {voice} ready.", flush=True)
        return _models[voice]


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": bool(_models), "lazy_load": True,
            "engine": "kokoro-ru", "voices": list(VOICES),
            "device": _runtime_device,
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
        print("Kokoro-RU: initializing PyTorch runtime...", flush=True)
        import torch
        print(f"Kokoro-RU: PyTorch ready; CUDA available={torch.cuda.is_available()}.", flush=True)
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
