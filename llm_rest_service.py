# -*- coding: utf-8 -*-
"""Встроенная локальная LLM проекта: маленький HTTP-сервис на llama-cpp-python (GGUF), запускается КАК ОБЫЧНЫЙ ПОДПРОЦЕСС
Windows в своём окружении `.venv_llm` (install.bat ставит всё и качает веса) — без Docker, без LM Studio/Ollama, без облака.

Для чего: подсказки по ударению («Спросить LLM» на вкладке «Проверка ударений», модуль `stress_llm.py`). Конвейер озвучки
эту LLM НЕ использует (режим «100% локально/бесплатно» сохраняется).

Эндпоинты: GET /health, POST /v1/chat/completions (минимальный OpenAI-совместимый ответ), POST /unload.
Модель грузится при первом запросе и ВЫГРУЖАЕТСЯ после простоя (по умолчанию 120 с), чтобы освободить видеопамять для TTS.

Параметры (переменные окружения): LLM_MODEL_PATH, LLM_GPU_LAYERS (-1 = все на GPU, 0 = CPU), LLM_CTX, LLM_IDLE_SECONDS.
`python llm_rest_service.py --download` скачивает веса (huggingface_hub) в llm_models\\.
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

SERVICE_VERSION = "2026-10-20.1-initial"
DEFAULT_PORT = 5013
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("LLM_MODEL_PATH") or ROOT / "llm_models" / MODEL_FILE)
GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", "-1"))
N_CTX = int(os.environ.get("LLM_CTX", "6144"))
IDLE_SECONDS = float(os.environ.get("LLM_IDLE_SECONDS", "120"))

_llm = None
_last_used = time.time()
_lock = threading.Lock()


def download() -> int:
    from huggingface_hub import hf_hub_download
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Модель уже есть: {MODEL_PATH}")
        return 0
    print(f"Скачиваю {MODEL_REPO}/{MODEL_FILE} (~4.7 ГБ)...")
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODEL_PATH.parent))
    print(f"Готово: {path}")
    return 0


def _add_cuda_dll_dirs():
    """CUDA-колесо llama-cpp-python не несёт cudart/cublas: берём их из pip-пакетов nvidia-* этого venv (install.bat ставит)."""
    if sys.platform != "win32":
        return
    for base in (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",):
        for d in base.glob("*/bin") if base.exists() else []:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def _load():
    global _llm
    if _llm is None:
        _add_cuda_dll_dirs()
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Нет файла модели {MODEL_PATH}. Запустите install.bat (или llm_rest_service.py --download).")
        from llama_cpp import Llama
        print(f"Загружаю модель {MODEL_PATH.name} (gpu_layers={GPU_LAYERS}, ctx={N_CTX})...", flush=True)
        _llm = Llama(model_path=str(MODEL_PATH), n_gpu_layers=GPU_LAYERS, n_ctx=N_CTX, verbose=False)
    return _llm


def _unload():
    global _llm
    if _llm is not None:
        _llm = None
        gc.collect()
        print("Модель выгружена (простой) — видеопамять свободна.", flush=True)


def _idle_watcher():
    while True:
        time.sleep(10)
        if _llm is not None and time.time() - _last_used > IDLE_SECONDS and _lock.acquire(blocking=False):
            try:
                _unload()
            finally:
                _lock.release()


def build_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI()

    class Msg(BaseModel):
        role: str
        content: str

    class ChatReq(BaseModel):
        model: str = ""
        messages: list[Msg]
        max_tokens: int = 300
        temperature: float = 0.0

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "llm", "service_version": SERVICE_VERSION, "model": MODEL_PATH.name,
                "model_present": MODEL_PATH.exists(), "loaded": _llm is not None}

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        global _last_used
        with _lock:
            try:
                llm = _load()
                out = llm.create_chat_completion(messages=[m.model_dump() for m in req.messages],
                                                 max_tokens=req.max_tokens, temperature=req.temperature)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
            finally:
                _last_used = time.time()
        return out

    @app.post("/unload")
    def unload():
        with _lock:
            _unload()
        return {"status": "ok"}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download:
        sys.exit(download())
    import uvicorn
    threading.Thread(target=_idle_watcher, daemon=True).start()
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
