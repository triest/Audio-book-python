# -*- coding: utf-8 -*-
"""HTTP-сервис клонирования голоса на движке CosyVoice 3 (FunAudioLLM/CosyVoice,
Fun-CosyVoice3-0.5B) — В ОТЛИЧИЕ от cosyvoice_rest_service.py (который
запускается прямо на Windows в .venv_cosyvoice), этот сервис запускается
ВНУТРИ Docker-контейнера (см. docker/cosyvoice3/).

Почему отдельный контейнер, а не ещё один движок в существующем сервисе:
у CosyVoice3 жёстко закреплены версии зависимостей (torch==2.3.1 и т.д.),
несовместимые с тем, что уже стоит в .venv_cosyvoice для F5-TTS/XTTS/ESpeech
(там версии новее) - установка в общее окружение сломала бы остальные
движки. Плюс часть зависимостей (onnxruntime-gpu, deepspeed) на Windows
недоступна/работает иначе, чем на Linux - внутри Linux-контейнера этой
проблемы нет вовсе.

Реализует ТОТ ЖЕ набор HTTP-эндпоинтов, что и cosyvoice_rest_service.py
(/health, /voices, /getwav, /add_voice, /transcribe, DELETE /voices/{name}) -
поэтому существующая вкладка «Голоса CosyVoice» в fb2_reader_gui.py и вся
логика управления голосами работают без изменений, просто указывают на
другой порт (см. COSYVOICE3_DEFAULT_REST_URL в fb2_reader_gui.py).

Клонирование голоса - zero-shot, по образцу 3-10 сек (см. MAX_PROMPT_SECONDS
ниже - короче, чем у F5-TTS/ESpeech, так рекомендует сама модель).
Текст образца (транскрипт) распознаётся тем же Whisper, что и в
cosyvoice_rest_service.py, если не указан явно.
"""
import io
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, File
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool
import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn

app = FastAPI()

# Меняется при каждом значимом изменении этого файла - клиент
# (fb2_reader_gui.py) сверяет эту версию через /health, как и для основного
# сервиса CosyVoice (см. COSYVOICE3_EXPECTED_SERVICE_VERSION там).
SERVICE_VERSION = "2026-08-23.1"

BASE_DIR = Path(__file__).resolve().parent
# Внутри контейнера сюда смонтирован volume (см. docker-compose.yml) - так
# добавленные голоса переживают перезапуск/обновление контейнера.
DATA_DIR = Path(os.environ.get("COSYVOICE3_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOICES_JSON = DATA_DIR / "cosyvoice3_voices.json"
VOICES_DIR = DATA_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)

MODEL_DIR = os.environ.get("COSYVOICE3_MODEL_DIR", "/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")

cosyvoice_model = None
synth_sample_rate = 24000
voice_profiles: dict = {}

LOG_FILE = str(DATA_DIR / "cosyvoice3_rest_service.log")
logger = logging.getLogger("cosyvoice3_rest_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    from logging.handlers import RotatingFileHandler
    _console_handler = logging.StreamHandler()
    _file_handler = RotatingFileHandler(LOG_FILE, encoding="utf-8", maxBytes=5_000_000, backupCount=1)
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _console_handler.setFormatter(_formatter)
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)
    logger.propagate = False


@app.exception_handler(Exception)
async def _log_unhandled_exceptions(request: Request, exc: Exception):
    logger.error(f"Необработанное исключение на {request.method} {request.url.path}:\n"
                 + traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка сервера: {exc}"})


def _write_wav_bytes(audio_t: torch.Tensor, sample_rate: int) -> bytes:
    arr = audio_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


MAX_PROMPT_SECONDS = 10.0  # CosyVoice3 рекомендует короткие образцы (3-10 сек) -
# длиннее не улучшает клонирование, только замедляет синтез.


def _trim_prompt_if_needed(wav_path: Path, name: str):
    try:
        info = sf.info(str(wav_path))
        duration = info.frames / float(info.samplerate)
        if duration > MAX_PROMPT_SECONDS:
            arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            trimmed = arr[: int(MAX_PROMPT_SECONDS * sr)]
            sf.write(str(wav_path), trimmed, sr, format="WAV", subtype="PCM_16")
            logger.info(
                f"Образец голоса {name!r} был длиннее {MAX_PROMPT_SECONDS:.0f} сек "
                f"({duration:.1f} сек) — обрезан до первых {MAX_PROMPT_SECONDS:.0f} секунд."
            )
    except Exception:
        logger.warning(f"Не удалось проверить/обрезать длину образца {name!r} (оставляю как есть):\n"
                        + traceback.format_exc())


def _load_voice_profiles():
    global voice_profiles
    voice_profiles = {}
    if not VOICES_JSON.exists():
        logger.warning(f"{VOICES_JSON} не найден — ни одного профиля голоса не загружено.")
        return
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.error("Не удалось прочитать cosyvoice3_voices.json:\n" + traceback.format_exc())
        return
    for entry in entries:
        name = entry.get("name")
        wav_path = DATA_DIR / entry.get("wav", "")
        text = entry.get("text", "") or ""
        if not name or not wav_path.exists():
            logger.warning(f"Профиль {entry!r} пропущен — не найден файл {wav_path}")
            continue
        _trim_prompt_if_needed(wav_path, name)
        voice_profiles[name] = {"text": text, "wav_path": str(wav_path)}
        logger.info(f"Загружен профиль голоса: {name!r} (образец: {wav_path.name})")


def _load_cosyvoice3_model():
    global cosyvoice_model, synth_sample_rate
    sys.path.insert(0, "/opt/CosyVoice")
    sys.path.insert(0, "/opt/CosyVoice/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import AutoModel

    logger.info(f"Загружаю модель CosyVoice3 из {MODEL_DIR} ...")
    cosyvoice_model = AutoModel(model_dir=MODEL_DIR)
    synth_sample_rate = int(getattr(cosyvoice_model, "sample_rate", 24000) or 24000)
    logger.info(f"Модель CosyVoice3 загружена. Частота дискретизации: {synth_sample_rate} Гц. "
                f"GPU доступен: {torch.cuda.is_available()}.")


@app.on_event("startup")
async def startup_event():
    try:
        _load_cosyvoice3_model()
    except Exception:
        logger.error("Не удалось загрузить модель CosyVoice3:\n" + traceback.format_exc())
    _load_voice_profiles()


@app.get("/voices")
async def voices():
    return {"voices": list(voice_profiles.keys())}


@app.get("/health")
async def health():
    return {
        "model_loaded": cosyvoice_model is not None,
        "engine": "cosyvoice3" if cosyvoice_model is not None else None,
        "voices": list(voice_profiles.keys()),
        "service_version": SERVICE_VERSION,
    }


_whisper_asr_model = None


def _get_whisper_asr_model():
    global _whisper_asr_model
    if _whisper_asr_model is None:
        import whisper
        logger.info("Загружаю модель Whisper для распознавания речи (\"small\")...")
        _whisper_asr_model = whisper.load_model("small")
        logger.info("Модель Whisper для распознавания речи загружена.")
    return _whisper_asr_model


MAX_TRANSCRIBE_SECONDS = 30.0


def _load_wav_for_whisper(path_or_bytesio, max_seconds: float = MAX_TRANSCRIBE_SECONDS) -> np.ndarray:
    arr, sr = sf.read(path_or_bytesio, dtype="float32", always_2d=True)
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    else:
        arr = arr.reshape(-1)
    max_samples = int(max_seconds * sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    target_sr = 16000
    if sr != target_sr:
        wav_tensor = torch.from_numpy(arr).float().unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, target_sr)
        arr = wav_tensor.squeeze(0).numpy()
    return np.ascontiguousarray(arr.astype(np.float32))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    try:
        try:
            arr = _load_wav_for_whisper(io.BytesIO(data))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать аудиофайл: {e}") from e

        def _run():
            model = _get_whisper_asr_model()
            result = model.transcribe(arr, language="ru", fp16=torch.cuda.is_available())
            return result.get("text", "").strip()

        text = await run_in_threadpool(_run)
        return {"text": text}
    except Exception:
        logger.error("[/transcribe] Распознавание речи не удалось:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Не удалось распознать речь — см. лог сервиса")


@app.post("/add_voice")
async def add_voice(name: str = Form(...), text: str = Form(""), audio: UploadFile = File(...)):
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Пустое или недопустимое имя профиля")

    dest = VOICES_DIR / f"{safe_name}.wav"
    data = await audio.read()
    dest.write_bytes(data)
    _trim_prompt_if_needed(dest, safe_name)

    entries = []
    if VOICES_JSON.exists():
        try:
            entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("name") != safe_name]
    entries.append({"name": safe_name, "wav": f"voices/{dest.name}", "text": text})
    VOICES_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    _load_voice_profiles()
    if safe_name not in voice_profiles:
        raise HTTPException(status_code=400, detail="Не удалось загрузить добавленный образец — см. лог сервиса")
    return {"ok": True, "name": safe_name, "voices": list(voice_profiles.keys())}


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    if not VOICES_JSON.exists():
        raise HTTPException(status_code=404, detail=f"Профиль {name!r} не найден")
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.error("Не удалось прочитать cosyvoice3_voices.json:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Не удалось прочитать cosyvoice3_voices.json — см. лог сервиса")

    match = next((e for e in entries if e.get("name") == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Профиль {name!r} не найден")

    remaining = [e for e in entries if e.get("name") != name]
    VOICES_JSON.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")

    wav_path = DATA_DIR / match.get("wav", "")
    try:
        if wav_path.exists():
            wav_path.unlink()
    except Exception:
        logger.warning(f"Профиль {name!r} удалён из списка, но не удалось удалить файл {wav_path}:\n"
                        + traceback.format_exc())

    _load_voice_profiles()
    logger.info(f"Удалён профиль голоса: {name!r}")
    return {"ok": True, "name": name, "voices": list(voice_profiles.keys())}


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers_to_words_ru(text: str) -> str:
    """CosyVoice3 заявляет собственную нормализацию чисел (text_frontend=True
    при синтезе) - поэтому, в отличие от F5-TTS/XTTS/ESpeech в
    cosyvoice_rest_service.py, здесь num2words НЕ применяется по умолчанию.
    Оставлено на случай, если встроенная нормализация окажется хуже
    (см. _synthesize) - тогда легко включить, просто вызвав эту функцию."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def _repl(m):
        raw = m.group(0)
        try:
            if "," in raw or "." in raw:
                return num2words(float(raw.replace(",", ".")), lang="ru")
            return num2words(int(raw), lang="ru")
        except Exception:
            return raw

    return _NUMBER_RE.sub(_repl, text)


def _persist_profile_text(voice: str, text: str):
    if not VOICES_JSON.exists():
        return
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return
    changed = False
    for e in entries:
        if e.get("name") == voice and not e.get("text"):
            e["text"] = text
            changed = True
    if changed:
        VOICES_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_profile_text(voice: str, profile: dict) -> str:
    """CosyVoice3 (как и F5-TTS) для zero-shot клонирования использует текст
    образца (prompt_text) - если пользователь не указал его явно,
    распознаём через Whisper и запоминаем результат."""
    if profile.get("text"):
        return profile["text"]
    try:
        arr = _load_wav_for_whisper(profile["wav_path"])
        model = _get_whisper_asr_model()
        result = model.transcribe(arr, language="ru", fp16=torch.cuda.is_available())
        text = (result.get("text") or "").strip()
    except Exception:
        logger.warning(f"Не удалось автоматически распознать текст образца для {voice!r} - "
                        "синтез продолжится без него (клонирование может быть хуже):\n"
                        + traceback.format_exc())
        return ""
    if text:
        profile["text"] = text
        _persist_profile_text(voice, text)
        logger.info(f"Автоматически распознан текст образца для {voice!r}: {text}")
    return text


def _synthesize(text: str, voice: str) -> np.ndarray:
    profile = voice_profiles.get(voice)
    if profile is None:
        raise RuntimeError(f"Профиль голоса {voice!r} не найден (доступны: {list(voice_profiles.keys())})")

    prompt_wav_path = profile["wav_path"]
    ref_text = _ensure_profile_text(voice, profile) or " "

    chunks = []
    for out in cosyvoice_model.inference_zero_shot(
        text, ref_text, prompt_wav_path, stream=False, text_frontend=True,
    ):
        chunks.append(out["tts_speech"])
    if not chunks:
        raise RuntimeError("cosyvoice3 не вернул аудио (пустой результат)")
    speech = torch.cat(chunks, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if speech.size == 0:
        raise RuntimeError("cosyvoice3 не вернул аудио (пустой результат)")
    return speech


@app.get(
    "/getwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response,
)
async def getwav(text_to_speech: str, voice: str = "default", sample_rate: int = 24000):
    if cosyvoice_model is None:
        raise HTTPException(status_code=500, detail="Модель синтеза не загружена — см. лог сервиса")
    if voice not in voice_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Профиль голоса {voice!r} не найден. Доступны: {list(voice_profiles.keys())}",
        )

    logger.info(f"[/getwav] engine=cosyvoice3 voice={voice!r}, {len(text_to_speech)} симв.: {text_to_speech[:200]}")

    try:
        audio = await run_in_threadpool(_synthesize, text_to_speech, voice)
    except Exception:
        logger.error("[/getwav] Synthesis failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=400, detail="cosyvoice3 synthesis failed, see server log")

    model_sr = synth_sample_rate
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != model_sr:
        audio_t = torchaudio.functional.resample(audio_t, model_sr, sample_rate)

    wav_bytes = _write_wav_bytes(audio_t, sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("COSYVOICE3_PORT", "5012"))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Сервис остановлен пользователем (Ctrl+C)")
