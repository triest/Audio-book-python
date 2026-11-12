# -*- coding: utf-8 -*-
"""HTTP-сервис клонирования голоса на движке CosyVoice 3 (FunAudioLLM/CosyVoice,
Fun-CosyVoice3-0.5B) — запускается ПРЯМО НА WINDOWS, отдельным подпроцессом в
своём собственном окружении .venv_cosyvoice3 (install.bat клонирует туда же
репозиторий FunAudioLLM/CosyVoice - см. секцию CosyVoice3 в install.bat),
точно так же, как cosyvoice_rest_service.py работает в .venv_cosyvoice.

Почему ОТДЕЛЬНОЕ окружение, а не ещё один движок в cosyvoice_rest_service.py:
у CosyVoice3 жёстко закреплены версии зависимостей (torch==2.3.1 и т.д.),
несовместимые с тем, что уже стоит в .venv_cosyvoice для F5-TTS/XTTS/ESpeech
(там версии новее) - установка в общее окружение сломала бы остальные
движки.

(Изначально это планировалось запускать в Docker-контейнере для полной
изоляции - см. docker/cosyvoice3/ - но на практике Docker Desktop/WSL2 у
пользователя оказался слишком капризным, поэтому вместо контейнера - тот же
подход, что и для остального CosyVoice-стека: отдельный venv на голом
Windows. Часть зависимостей CosyVoice, которые официально поддерживаются
только на Linux (onnxruntime-gpu, deepspeed, TensorRT) - все опциональные
ускорители инференса, и requirements.txt сам ставит их только под
sys_platform == 'linux' - на Windows вместо них просто ставится обычный
onnxruntime и синтез идёт без них, чуть медленнее, но рабочий.)

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
SERVICE_VERSION = "2026-08-24.15-transcribe-window-fix"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("COSYVOICE3_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOICES_JSON = DATA_DIR / "cosyvoice3_voices.json"
VOICES_DIR = DATA_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# По умолчанию - рядом с этим файлом, внутри склонированного install.bat'ом
# репозитория CosyVoice (см. install.bat) - т.е. BASE_DIR и есть корень
# репозитория (cosyvoice3_rest_service.py копируется прямо туда).
MODEL_DIR = os.environ.get("COSYVOICE3_MODEL_DIR", str(BASE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"))

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


def _ensure_model_weights_downloaded():
    """Скачивает веса Fun-CosyVoice3-0.5B при первом запуске (кэш - в
    MODEL_DIR, переживает перезапуски программы, скачивается один раз) -
    раньше это делал entrypoint.sh контейнера, теперь (см. модуль-докстринг)
    сервис запускается прямо на Windows, поэтому делаем то же самое здесь."""
    model_dir = Path(MODEL_DIR)
    if model_dir.exists() and any(model_dir.iterdir()):
        return
    logger.info(f"Скачиваю веса Fun-CosyVoice3-0.5B в {model_dir} (один раз, ~1-2 ГБ)...")
    from huggingface_hub import snapshot_download
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=str(model_dir))
    logger.info("Веса Fun-CosyVoice3-0.5B скачаны.")


def _load_cosyvoice3_model():
    global cosyvoice_model, synth_sample_rate
    # BASE_DIR - корень склонированного install.bat'ом репозитория CosyVoice
    # (cosyvoice3_rest_service.py копируется прямо туда) - что в Docker-
    # контейнере (/opt/CosyVoice), что при обычном запуске на Windows
    # (CosyVoice3\ рядом с остальной программой).
    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    _ensure_model_weights_downloaded()

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


# ВАЖНО: раньше было 30.0, а сам голосовой образец обрезается до
# MAX_PROMPT_SECONDS (10 сек) в _trim_prompt_if_needed - то есть кнопка
# «Распознать текст» в GUI (POST /transcribe) распознавала на 20 секунд
# БОЛЬШЕ речи, чем реально попадало в обрезанный образец. Из-за этого
# сохранённый transcript систематически не соответствовал настоящему
# содержимому 10-секундного wav у КАЖДОГО добавленного через GUI голоса
# (не только у старых профилей с кэша до обрезки) - модель получала текст
# заведомо длиннее и "не про то" аудио, что и приводило к
# искажённому/зацикленному звуку при синтезе. Теперь /transcribe читает
# ровно столько же аудио, сколько реально останется в образце.
MAX_TRANSCRIBE_SECONDS = MAX_PROMPT_SECONDS


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


def _persist_profile_text(voice: str, text: str, overwrite: bool = False):
    if not VOICES_JSON.exists():
        return
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return
    changed = False
    for e in entries:
        if e.get("name") == voice and (overwrite or not e.get("text")):
            if e.get("text") != text:
                e["text"] = text
                changed = True
    if changed:
        VOICES_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


# При ~10 сек образца (MAX_PROMPT_SECONDS) и обычном темпе речи (~3
# слова/сек) в него физически не помещается больше ~35-40 слов - если
# сохранённый prompt_text длиннее, это почти наверняка "протухший" текст,
# распознанный ДО того, как .wav был обрезан до 10 сек (например, из
# старой версии сервиса без _trim_prompt_if_needed, или скопированный
# откуда-то ещё) - именно такой случай был у профиля "002 Пролог"
# (несколько абзацев текста на 10-секундный образец). Из-за этого
# CosyVoice3 на КАЖДОМ синтезе ругался "synthesis text ... too short than
# prompt text ... this may lead to bad performance" и в итоге падал в
# вокодере с "Kernel size can't be greater than actual input size" - даже
# короткие фрагменты оставались "слишком короткими" по сравнению с
# гигантским prompt_text.
_MAX_PROMPT_TEXT_WORDS = 40


def _ensure_profile_text(voice: str, profile: dict) -> str:
    """CosyVoice3 (как и F5-TTS) для zero-shot клонирования использует текст
    образца (prompt_text) - если пользователь не указал его явно,
    распознаём через Whisper и запоминаем результат."""
    cached = profile.get("text")
    if cached:
        words = cached.split()
        if len(words) <= _MAX_PROMPT_TEXT_WORDS:
            return cached
        truncated = " ".join(words[:_MAX_PROMPT_TEXT_WORDS])
        logger.warning(
            f"[_ensure_profile_text] Сохранённый текст образца для {voice!r} "
            f"подозрительно длинный ({len(words)} слов на "
            f"{MAX_PROMPT_SECONDS:.0f}-секундный образец) - вероятно, "
            "распознан до обрезки аудио. Обрезаю до "
            f"{_MAX_PROMPT_TEXT_WORDS} слов и пересохраняю."
        )
        profile["text"] = truncated
        _persist_profile_text(voice, truncated, overwrite=True)
        return truncated
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
    words = text.split()
    if len(words) > _MAX_PROMPT_TEXT_WORDS:
        text = " ".join(words[:_MAX_PROMPT_TEXT_WORDS])
    if text:
        profile["text"] = text
        _persist_profile_text(voice, text)
        logger.info(f"Автоматически распознан текст образца для {voice!r}: {text}")
    return text


def _run_inference_zero_shot(text: str, ref_text: str, prompt_wav_path) -> list:
    """НЕ вызывает публичный cosyvoice_model.inference_zero_shot() напрямую.

    Прочитал исходники CosyVoice3 (cosyvoice/cli/cosyvoice.py) прямо на
    компьютере пользователя - и там нашлась настоящая причина всех падений
    на коротких фрагментах ("Kernel size can't be greater...",
    "AssertionError: <|endofprompt|> not detected"), из-за которой ВСЕ
    предыдущие попытки (задваивание текста, дополнение словами из
    prompt_text) не работали:

        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, ...)):
            ...

    inference_zero_shot() САМ разбивает переданный text_to_speech обратно
    на предложения (split=True) перед синтезом - какой бы длинный текст мы
    сюда ни передали, CosyVoice3 всё равно резал его обратно на исходные
    короткие куски и синтезировал их по одному. Наше собственное
    дополнение текста просто отбрасывалось на этом шаге, поэтому фрагмент
    вроде "это Честь." падал одинаково что с паддингом, что без него.

    fb2_reader.py уже сам режет текст книги на предложения/фразы перед
    отправкой сюда - повторное разбиение внутри CosyVoice3 не нужно и
    только вредит. Поэтому здесь вызываются внутренние методы
    frontend/model напрямую, с split=False (см. frontend.text_normalize) -
    наш фрагмент идёт в модель ЦЕЛИКОМ, одним куском, без пересборки.

    НАСТОЯЩАЯ причина падений (после split=False проверил по логу: успешных
    /getwav не было НИ ОДНОГО, ни на длинных, ни на коротких фрагментах -
    значит короткие фрагменты были ни при чём): прочитал cosyvoice/llm/llm.py
    прямо на компьютере пользователя - LLM конкретно этого чекпоинта
    (CosyVoice3LM) жёстко требует, чтобы где-то в склеенных prompt_text+text
    токенах встречался специальный токен <|endofprompt|> (id 151646):

        text = torch.concat([prompt_text, text], dim=1)
        if self.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in text, '<|endofprompt|> not detected ...'

    Ни frontend_zero_shot, ни text_normalize сами этот маркер никуда не
    вставляют - его обязан передать вызывающий код (см. instruct_list в
    cosyvoice/utils/common.py - там ВСЕ примеры заканчиваются буквально
    "...<|endofprompt|>"). Обычный демо-код zero-shot (inference_zero_shot)
    для этого чекпоинта его не добавляет вообще - похоже, баг/недосмотр в
    самом upstream-репозитории применительно к обычному (не instruct)
    zero-shot клонированию голоса. Поэтому раньше падал АБСОЛЮТНО каждый
    вызов синтеза через CosyVoice3, а не только короткие фрагменты - просто
    короткие чаще встречались в тексте книги, вот и казалось, что дело в
    длине.

    НОВОЕ (после того как звук заработал, но начал повторять "не ту книгу" -
    выяснилось на реальном примере: образец = "Пролог", целевой текст =
    "Предисловие редактора", а на выходе звучал текст ПРОЛОГА): пробовал
    сначала переставлять/убирать маркер по догадке (транскрипт+маркер, потом
    просто маркер, потом короткий обрывок+маркер) - вторая и третья попытки
    даже сломали стабильность (AssertionError стал вылетать через раз).

    НАШЁЛ ОФИЦИАЛЬНЫЙ ПРИМЕР (README модели FunAudioLLM/Fun-CosyVoice3-0.5B
    на HuggingFace - до этого гадал по исходникам, теперь есть настоящий
    эталон использования от авторов):

        cosyvoice.inference_zero_shot(
            'Your text to synthesize',
            'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
            './asset/zero_shot_prompt.wav', stream=False)

    Формат prompt_text строго такой: "You are a helpful assistant.
    [необязательная короткая инструкция]<|endofprompt|>НАСТОЯЩИЙ ТРАНСКРИПТ
    ОБРАЗЦА". То есть маркер стоит МЕЖДУ фиксированной вводной фразой и
    настоящим транскриптом - а не ПОСЛЕ транскрипта (как было раньше) и не
    ВМЕСТО него (как было в прошлой попытке). Видимо, модель обучена именно
    на такой структуре: "you are a helpful assistant" + маркер как разделитель
    служебной части, после которого идёт РЕАЛЬНАЯ речь для клонирования - а
    когда транскрипт стоял ДО маркера, модель воспринимала его как левый
    контекст для продолжения (отсюда "не та книга"), а когда после маркера
    не было вообще ничего (пустой хвост) - токенизация/сама модель вела себя
    нестабильно (это и объясняет прошлые случайные AssertionError)."""
    frontend = cosyvoice_model.frontend
    ref_text_clean = ref_text.strip() if ref_text else ""
    prompt_text_marker = f"You are a helpful assistant.<|endofprompt|>{ref_text_clean}"
    prompt_text_norm = frontend.text_normalize(prompt_text_marker, split=False, text_frontend=True)
    text_norm = frontend.text_normalize(text, split=False, text_frontend=True)

    model_input = frontend.frontend_zero_shot(
        text_norm, prompt_text_norm, prompt_wav_path, cosyvoice_model.sample_rate, "",
    )
    chunks = []
    for out in cosyvoice_model.model.tts(**model_input, stream=False, speed=1.0):
        chunks.append(out["tts_speech"])
    return chunks


def _pad_text_to_match_prompt(text: str, ref_text: str, max_repeats: int = 30) -> str:
    """CosyVoice3 хочет, чтобы синтезируемый текст был не короче текста
    образца (prompt_text) - иначе печатает "too short than prompt text ...
    this may lead to bad performance" и нередко буквально предсказывает 0
    речевых токенов, что роняет вокодер. Книжные фрагменты (по предложению
    или по знаку препинания) почти всегда КОРОЧЕ произвольного prompt_text
    - поэтому раньше это стабильно падало почти на каждом фрагменте вне
    зависимости от длины конкретной фразы. Наращиваем текст повтором самого
    себя ДО попытки синтеза, а не только после падения - так "too short"
    вообще не должно возникать в обычном случае.

    ВАЖНО: предыдущая версия ограничивала число повторов до 6 - для
    короткого фрагмента ("это Честь." - 2 слова) и обычного prompt_text
    ~40 слов этого категорически не хватало (6 повторов = 12 слов < 40),
    поэтому "too short" продолжал появляться на каждом коротком фрагменте.
    Теперь число повторов считается по факту нужного соотношения слов, а
    max_repeats - это просто защитный потолок на случай совсем короткого
    текста (одно слово) с длинным prompt_text, чтобы не раздувать фразу до
    абсурда."""
    text_words = len(text.split())
    ref_words = len(ref_text.split())
    if ref_words <= 0 or text_words <= 0:
        return text
    if text_words >= ref_words:
        return text
    # ВАЖНО: изначально текст просто повторялся самим собой ("И, И, И,
    # ..."), но на практике ИМЕННО чистое повторение одного и того же
    # короткого куска (особенно однословных междометий с запятой) стабильно
    # вызывало отдельную, более серьёзную поломку глубоко внутри
    # токенизатора CosyVoice3 - "AssertionError: <|endofprompt|> not
    # detected in CosyVoice3 text or prompt_text" в фоновом потоке
    # llm_job, из-за чего генератор не выдавал ни одного токена и вокодер
    # падал так же, как при "too short". Чем больше повторов - тем хуже.
    # Поэтому вместо повторения текста добавляем к нему настоящие слова из
    # текста ОБРАЗЦА (ref_text) - они заведомо "нормальный" русский текст,
    # с которым модель уже работает (это её собственный prompt_text), и
    # никогда не превращаются в вырожденную последовательность повторов.
    ref_words_list = ref_text.split()
    filler_needed = min(ref_words - text_words, max_repeats * max(1, text_words))
    filler = " ".join(ref_words_list[:filler_needed])
    return f"{text} {filler}".strip()


def _synthesize(text: str, voice: str) -> np.ndarray:
    profile = voice_profiles.get(voice)
    if profile is None:
        raise RuntimeError(f"Профиль голоса {voice!r} не найден (доступны: {list(voice_profiles.keys())})")

    prompt_wav_path = profile["wav_path"]
    ref_text = _ensure_profile_text(voice, profile) or " "

    # Раньше здесь текст фрагмента принудительно "дополнялся" под длину
    # prompt_text (см. _pad_text_to_match_prompt, оставлена ниже только как
    # аварийный запасной вариант) - это было попыткой обойти
    # "AssertionError: <|endofprompt|> not detected", но настоящая причина
    # была не в длине текста (см. большой комментарий в
    # _run_inference_zero_shot - там теперь добавляется сам маркер
    # <|endofprompt|>). Раз причина устранена по существу, дополнять текст
    # для КАЖДОГО фрагмента больше не нужно - а лишний повтор/добавка была
    # слышна как отсебятина в озвучке. Используем исходный текст фрагмента
    # как есть.
    try:
        chunks = _run_inference_zero_shot(text, ref_text, prompt_wav_path)
    except RuntimeError as e:
        # Аварийный запасной вариант на случай, если вокодер всё же
        # споткнётся на каком-то патологически коротком/вырожденном
        # фрагменте ("Calculated padded input size per channel: (3).
        # Kernel size: (4)...") - пробуем один раз с текстом, дополненным
        # словами из образца, вместо того чтобы терять фрагмент целиком.
        if "Kernel size can't be greater than actual input size" not in str(e):
            raise
        logger.warning(
            f"[_synthesize] Ошибка вокодера CosyVoice3 на фрагменте "
            f"(voice={voice!r}, text={text!r}) - повторяю с текстом, "
            "дополненным словами образца, чтобы не терять фрагмент."
        )
        padded_text = _pad_text_to_match_prompt(text, ref_text)
        chunks = _run_inference_zero_shot(padded_text, ref_text, prompt_wav_path)

    if not chunks:
        raise RuntimeError("cosyvoice3 не вернул аудио (пустой результат)")
    speech = torch.cat(chunks, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if speech.size == 0:
        raise RuntimeError("cosyvoice3 не вернул аудио (пустой результат)")
    peak = float(np.abs(speech).max()) if speech.size else 0.0
    if peak < 1e-4:
        # Аудио физически есть (не пустой массив), но по громкости это
        # тишина - в получившемся .wav так и будет слышно "ничего". Раньше
        # это никак не логировалось, поэтому "файл создаётся, но тишина"
        # было не с чем разбираться. Самая вероятная причина - ref_text
        # (текст образца голоса, распознанный Whisper) оказался пустым:
        # ниже это подставляется как " ", а с таким "почти пустым" текстом
        # запроса CosyVoice3 может молча вернуть тишину вместо ошибки.
        logger.warning(
            f"[_synthesize] Синтез для voice={voice!r} вернул почти беззвучное "
            f"аудио (пиковая амплитуда {peak:.6f}). ref_text={ref_text!r}. "
            "Если ref_text пустой/пробел - Whisper не смог распознать образец "
            "голоса; удалите профиль и добавьте заново, либо укажите текст "
            "образца вручную."
        )
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
