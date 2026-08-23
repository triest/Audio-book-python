# -*- coding: utf-8 -*-
"""HTTP-сервис клонирования голоса — запускается ВНУТРИ отдельного окружения
.venv_cosyvoice.

ДВА ДВИЖКА, ВЫБИРАЕТСЯ ПЕРЕМЕННОЙ ОКРУЖЕНИЯ TTS_ENGINE:
  - "f5" (по умолчанию) — F5-TTS с чекпоинтом hotstone228/F5-TTS-Russian
    (https://huggingface.co/hotstone228/F5-TTS-Russian), дообученным именно
    на русской речи (Common Voice, SOVA RUDevices, SberDevices Golos).
    Ударения и произношение заметно естественнее, чем у общей
    мультиязычной модели, потому что модель реально училась на русском, а
    не угадывает по паттернам из смеси 17 языков.
  - "xtts" — XTTS-v2 (Coqui TTS, пакет coqui-tts). Официально мультиязычная
    (17 языков, включая русский), но без специализации под русский —
    ударения из-за этого иногда невпопад (частично компенсируется
    RUAccent, см. _add_stress_marks ниже). Оставлена как запасной вариант
    (TTS_ENGINE=xtts в переменных окружения сервиса) на случай, если
    F5-TTS-Russian по каким-то причинам не подойдёт (например, если её
    некоммерческая лицензия CC-BY-NC-SA не устроит для вашего случая).

  Раньше здесь была модель CosyVoice2 — от неё отказались полностью: она
  официально не обучена на русском языке и даже при правильном
  zero-shot-клонировании выдавала невнятную "кашу", похожую на
  китайскую/японскую речь, и это ограничение самой модели, а не баг
  сервиса.

Программа (fb2_reader.py, режим --mode cosyvoice / run_cosyvoice) обращается
к этому сервису по HTTP так же, как раньше — GET /getwav и GET /voices ниже
(имя режима/переменных в программе осталось "cosyvoice" по историческим
причинам, но по факту это F5-TTS или XTTS).

ВАЖНО про голоса: используется клонирование по короткому образцу (в идеале
6-15 сек чистой речи одного голоса, WAV). Текст образца (транскрипт) нужен
движку F5-TTS для качественного клонирования (см. _ensure_profile_text —
если текст не указан явно, сервис сам распознаёт его через Whisper при
первом синтезе этим голосом и запоминает результат); движку XTTS текст не
обязателен вовсе. Профили голоса хранятся в cosyvoice_voices.json рядом с
этим файлом (плюс сами WAV-файлы в папке voices/). Добавить голос можно
через POST /add_voice (см. ниже) — проще всего это делает сама программа
fb2_reader_gui.py (кнопка "Добавить голос" в настройках режима CosyVoice).
"""
import io
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path


class _SafeStream:
    """Обёртка вокруг sys.stdout/sys.stderr, которая проглатывает ошибки
    при write()/flush() вместо падения всего процесса.

    Сервис запускается программой (fb2_reader_gui.py) без отдельного окна
    консоли — весь его вывод перенаправляется в канал (subprocess.PIPE),
    который программа читает и показывает в своём журнале. На некоторых
    системах в таком режиме sys.stderr.flush() иногда падает с
    "OSError: [Errno 22] Invalid argument" (не наша ошибка — так ведёт
    себя сама Windows-консоль в этом режиме). Библиотека tqdm (прогресс-
    бар, используется внутри CosyVoice при синтезе) вызывает flush() на
    каждой итерации — без этой обёртки любой синтез падал с этой ошибкой,
    даже если сам текст и голос были в полном порядке."""

    def __init__(self, orig):
        self._orig = orig

    def write(self, s):
        try:
            return self._orig.write(s)
        except Exception:
            return len(s) if isinstance(s, str) else 0

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._orig, name)


sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)

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
# (fb2_reader_gui.py) сверяет эту версию через /health и, если она не
# совпадает с ожидаемой, считает уже запущенный сервис устаревшим и сам
# перезапускает его. Без этого уже работающий (старый) процесс сервиса
# продолжал бы обслуживать запросы своим старым кодом сколько угодно долго
# после того, как программу обновили - новые эндпоинты/исправления просто
# не были бы видны, пока пользователь вручную не перезапустит компьютер.
SERVICE_VERSION = "2026-08-23.6"

BASE_DIR = Path(__file__).resolve().parent

VOICES_JSON = BASE_DIR / "cosyvoice_voices.json"
VOICES_DIR = BASE_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# XTTS-v2 требует согласия с лицензией (Coqui Public Model License) при
# первой загрузке весов - без этой переменной coqui-tts пытается задать
# интерактивный вопрос (y/n) в консоли, а у сервиса, запущенного программой
# без отдельного окна консоли (см. _SafeStream выше), спросить не у кого -
# процесс просто зависает молча на загрузке модели. Ставим согласие заранее.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

ENGINE = os.environ.get("TTS_ENGINE", "f5").strip().lower()
if ENGINE not in ("f5", "xtts"):
    ENGINE = "f5"

# --- F5-TTS-Russian ---
F5_MODEL_REPO = os.environ.get("F5_MODEL_REPO", "hotstone228/F5-TTS-Russian")
F5_CKPT_FILENAME = "model_last.safetensors"  # меньше и быстрее грузится, чем .pt
F5_VOCAB_FILENAME = "vocab.txt"
F5_ARCH = "F5TTS_Base"  # архитектура, на которой основан этот чекпоинт (см. его setting.json)
F5_MODEL_DIR = BASE_DIR / "pretrained_models" / "f5-tts-russian"

# --- XTTS-v2 (запасной движок, TTS_ENGINE=xtts) ---
XTTS_MODEL_NAME = os.environ.get("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_LANGUAGE = os.environ.get("XTTS_LANGUAGE", "ru")

cosyvoice_model = None  # объект F5TTS или coqui TTS, в зависимости от ENGINE -
# имя переменной оставлено как есть с прошлых версий сервиса, чтобы не
# переписывать все обращения к ней по файлу.
synth_sample_rate = 24000  # уточняется после загрузки модели в startup_event
# name -> {"text": str, "wav_path": str}
voice_profiles: dict = {}

LOG_FILE = "cosyvoice_rest_service.log"
logger = logging.getLogger("cosyvoice_rest_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    from logging.handlers import RotatingFileHandler
    _console_handler = logging.StreamHandler()
    # Растёт быстро (полный traceback на каждую неудачную фразу) - без
    # ротации разрастался до многих мегабайт за один сеанс озвучки книги.
    # maxBytes=5MB, храним один архивный файл (.1) в дополнение к текущему.
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
    """Сохраняет тензор в WAV-байты через soundfile вместо torchaudio.save():
    начиная с недавних версий torchaudio, для сохранения/декодирования
    аудио ему нужен отдельно установленный FFmpeg (через пакет torchcodec)
    — а на компьютере пользователя его обычно нет, из-за чего
    torchaudio.save() падает с ошибками загрузки libtorchcodec_core*.dll.
    soundfile обходится своей собственной библиотекой (libsndfile),
    которая идёт прямо внутри Python-пакета, поэтому системный FFmpeg ей
    не нужен."""
    arr = audio_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


MAX_PROMPT_SECONDS = 15.0  # с запасом хватает и F5-TTS, и XTTS; дольше — только медленнее


def _trim_prompt_if_needed(wav_path: Path, name: str):
    """Ни F5-TTS, ни XTTS не ставят жёсткого лимита в 30 сек, как раньше
    CosyVoice, но рекомендованная длина референсного образца — единицы-десятки секунд:
    более длинные образцы не улучшают клонирование, зато заметно замедляют
    синтез (весь образец прогоняется через кодировщик голоса на каждый
    вызов). Обрезаем на месте — и при добавлении нового голоса, и при
    загрузке уже сохранённых (на случай очень длинных файлов, добавленных
    раньше)."""
    try:
        info = sf.info(str(wav_path))
        duration = info.frames / float(info.samplerate)
        if duration > MAX_PROMPT_SECONDS:
            arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            trimmed = arr[: int(MAX_PROMPT_SECONDS * sr)]
            sf.write(str(wav_path), trimmed, sr, format="WAV", subtype="PCM_16")
            logger.info(
                f"Образец голоса {name!r} был длиннее {MAX_PROMPT_SECONDS:.0f} сек "
                f"({duration:.1f} сек) — обрезан до первых {MAX_PROMPT_SECONDS:.0f} секунд "
                "(для клонирования достаточно, а синтез так быстрее)."
            )
    except Exception:
        logger.warning(f"Не удалось проверить/обрезать длину образца {name!r} "
                        "(оставляю как есть):\n" + traceback.format_exc())


def _load_voice_profiles():
    """Читает cosyvoice_voices.json и проверяет, что референсные WAV-файлы
    каждого профиля существуют. Держим здесь только путь к файлу — сам
    движок (F5-TTS или XTTS) читает и обрабатывает референсное аудио
    внутри своего вызова синтеза."""
    global voice_profiles
    voice_profiles = {}
    if not VOICES_JSON.exists():
        logger.warning(f"{VOICES_JSON} не найден — ни одного профиля голоса не загружено. "
                        "Запустите install_cosyvoice.bat ещё раз или добавьте голос через /add_voice.")
        return
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.error("Не удалось прочитать cosyvoice_voices.json:\n" + traceback.format_exc())
        return
    for entry in entries:
        name = entry.get("name")
        wav_path = BASE_DIR / entry.get("wav", "")
        text = entry.get("text", "") or ""
        if not name or not wav_path.exists():
            logger.warning(f"Профиль {entry!r} пропущен — не найден файл {wav_path}")
            continue
        # На случай голосов, добавленных ДО того, как здесь появилась
        # автоматическая обрезка длинных образцов (см. _trim_prompt_if_needed
        # в /add_voice) - подчищаем и уже сохранённые файлы тоже, иначе они
        # так и остаются "битыми" (валятся с AssertionError при синтезе)
        # даже после обновления программы, пока их не пересохранят вручную.
        _trim_prompt_if_needed(wav_path, name)
        voice_profiles[name] = {"text": text, "wav_path": str(wav_path)}
        logger.info(f"Загружен профиль голоса: {name!r} (образец: {wav_path.name})")


def _load_f5_model():
    """Скачивает (при первом запуске, в F5_MODEL_DIR - подпадает под
    **/pretrained_models/ в .gitignore) чекпоинт F5-TTS-Russian с
    HuggingFace и загружает его через официальный класс F5TTS."""
    global cosyvoice_model, synth_sample_rate
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    logger.info(f"Скачиваю/проверяю чекпоинт F5-TTS-Russian ({F5_MODEL_REPO}) - "
                "при первом запуске это ~1.3 ГБ, один раз...")
    ckpt_path = hf_hub_download(repo_id=F5_MODEL_REPO, filename=F5_CKPT_FILENAME,
                                 local_dir=str(F5_MODEL_DIR))
    vocab_path = hf_hub_download(repo_id=F5_MODEL_REPO, filename=F5_VOCAB_FILENAME,
                                  local_dir=str(F5_MODEL_DIR))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Загружаю модель F5-TTS-Russian на устройство {device} ...")
    cosyvoice_model = F5TTS(model=F5_ARCH, ckpt_file=ckpt_path, vocab_file=vocab_path, device=device)
    synth_sample_rate = int(getattr(cosyvoice_model, "target_sample_rate", 24000) or 24000)
    logger.info(f"Модель F5-TTS-Russian загружена. Частота дискретизации: {synth_sample_rate} Гц. "
                f"GPU доступен: {torch.cuda.is_available()} (устройство: {device}).")


def _load_xtts_model():
    global cosyvoice_model, synth_sample_rate
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Загружаю модель XTTS-v2 ({XTTS_MODEL_NAME}) на устройство {device} ... "
                "при первом запуске модель (~2 ГБ) скачивается один раз, это может занять "
                "несколько минут.")
    cosyvoice_model = TTS(XTTS_MODEL_NAME).to(device)
    try:
        synth_sample_rate = int(cosyvoice_model.synthesizer.output_sample_rate)
    except Exception:
        synth_sample_rate = 24000
    logger.info(f"Модель XTTS-v2 загружена. Частота дискретизации: {synth_sample_rate} Гц. "
                f"GPU доступен: {torch.cuda.is_available()} (устройство: {device}).")


@app.on_event("startup")
async def startup_event():
    global cosyvoice_model, ENGINE
    loaders = {"f5": _load_f5_model, "xtts": _load_xtts_model}
    order = [ENGINE] + [e for e in loaders if e != ENGINE]  # пробуем выбранный первым, второй - запасной
    for engine_name in order:
        try:
            loaders[engine_name]()
            ENGINE = engine_name
            break
        except Exception:
            logger.error(f"Не удалось загрузить модель ({engine_name}):\n" + traceback.format_exc())
            cosyvoice_model = None
    else:
        logger.error("Не удалось загрузить ни один из движков синтеза (f5, xtts) - "
                      "см. ошибки выше по логу.")

    _load_voice_profiles()


@app.get("/voices")
async def voices():
    return {"voices": list(voice_profiles.keys())}


@app.get("/health")
async def health():
    """Позволяет клиенту (fb2_reader_gui.py) отличить "сервис отвечает, но
    модель не загрузилась при старте" от "сервис реально готов к работе" —
    обычный пинг любого URL (например /docs) отвечает 200/404 в обоих
    случаях, а /getwav в первом случае всегда возвращает 500. Клиент
    использует это, чтобы понять, что старый зависший процесс с
    незагруженной моделью нужно перезапустить, а не считать его рабочим."""
    return {
        "model_loaded": cosyvoice_model is not None,
        "engine": ENGINE if cosyvoice_model is not None else None,
        "voices": list(voice_profiles.keys()),
        "service_version": SERVICE_VERSION,
    }


_whisper_asr_model = None  # загружается лениво, при первом /transcribe


def _get_whisper_asr_model():
    """Загружает (при первом вызове) отдельную полную модель Whisper для
    распознавания речи — используется только для авто-заполнения поля
    "Текст записи" на вкладке добавления голоса (кнопка "Распознать текст").
    Это НЕ та часть whisper, что CosyVoice использует внутри себя для
    извлечения признаков (там используется только log_mel_spectrogram, без
    полноценного распознавания) - здесь нужна именно ASR-модель с
    transcribe(). Модель ("small", ~500 МБ) скачивается один раз при первом
    использовании этой кнопки и дальше берётся из кэша."""
    global _whisper_asr_model
    if _whisper_asr_model is None:
        import whisper
        logger.info("Загружаю модель Whisper для распознавания речи (\"small\", "
                    "один раз, при первом использовании кнопки \"Распознать текст\")...")
        _whisper_asr_model = whisper.load_model("small")
        logger.info("Модель Whisper для распознавания речи загружена.")
    return _whisper_asr_model


MAX_TRANSCRIBE_SECONDS = 30.0  # текст образца всё равно нужен только для
# клонирования (см. MAX_PROMPT_SECONDS) — гонять Whisper по всему файлу
# (может быть многие минуты) незачем и на CPU это очень медленно.


def _load_wav_for_whisper(path_or_bytesio, max_seconds: float = MAX_TRANSCRIBE_SECONDS) -> np.ndarray:
    """Декодирует WAV и ресемплирует в 16 кГц вручную через soundfile/torch,
    возвращая готовый numpy-массив для model.transcribe(). Используется
    ВЕЗДЕ, где нужен Whisper (и /transcribe, и автораспознавание текста
    образца для F5-TTS в _ensure_profile_text) — если вместо этого передать
    в whisper.transcribe() путь к файлу, она внутри себя попытается вызвать
    системный ffmpeg (whisper/audio.py: load_audio), которого на машине
    может не быть, и упадёт с FileNotFoundError [WinError 2]. Так мы вообще
    не зависим от ffmpeg."""
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
    """Распознаёт речь в начале загруженного аудиофайла через Whisper и
    возвращает текст — используется кнопкой "Распознать текст" на вкладке
    добавления голоса, чтобы не печатать транскрипт образца вручную.

    Важно: аудио декодируется и ресемплируется в 16 кГц вручную через
    soundfile/torch, а в Whisper передаётся уже готовый numpy-массив, а НЕ
    путь к файлу — иначе whisper.transcribe() внутри себя пытается вызвать
    системный ffmpeg (whisper/audio.py: load_audio), которого на машине
    может не быть, и падает с FileNotFoundError [WinError 2]. Так мы вообще
    не зависим от ffmpeg. Кроме того, распознаём только первые
    MAX_TRANSCRIBE_SECONDS секунд — этого достаточно, т.к. на клонирование
    всё равно идёт (и урезается) только начало образца, а прогонять Whisper
    по всему файлу (может быть многие минуты) не нужно и на CPU очень
    медленно."""
    data = await audio.read()
    try:
        try:
            arr = _load_wav_for_whisper(io.BytesIO(data))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать аудиофайл: {e}") from e

        def _run():
            model = _get_whisper_asr_model()
            result = model.transcribe(arr, language="ru", fp16=False)
            return result.get("text", "").strip()

        text = await run_in_threadpool(_run)
        return {"text": text}
    except Exception:
        logger.error("[/transcribe] Распознавание речи не удалось:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Не удалось распознать речь — см. лог сервиса")


@app.post("/add_voice")
async def add_voice(name: str = Form(...), text: str = Form(""), audio: UploadFile = File(...)):
    """Добавляет новый профиль голоса из загруженного аудио (6-15 секунд
    чистой речи одного голоса, без музыки/шума звучат лучше всего). text —
    расшифровка того, что сказано в образце; движку F5-TTS она нужна для
    качественного клонирования (если не указать - сервис распознает её
    сам через Whisper при первом синтезе, см. _ensure_profile_text), движку
    XTTS не обязательна вовсе."""
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Пустое или недопустимое имя профиля")

    dest = VOICES_DIR / f"{safe_name}.wav"
    data = await audio.read()
    dest.write_bytes(data)

    # Слишком длинный образец не улучшает клонирование, зато сильно
    # замедляет синтез (см. MAX_PROMPT_SECONDS) - обрезаем сразу.
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
    """Удаляет ранее добавленный профиль голоса — и запись в
    cosyvoice_voices.json, и сам WAV-файл в voices/. Используется вкладкой
    «Голоса CosyVoice» в программе (кнопка «Удалить выбранный голос»)."""
    if not VOICES_JSON.exists():
        raise HTTPException(status_code=404, detail=f"Профиль {name!r} не найден")
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        logger.error("Не удалось прочитать cosyvoice_voices.json:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Не удалось прочитать cosyvoice_voices.json — см. лог сервиса")

    match = next((e for e in entries if e.get("name") == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Профиль {name!r} не найден")

    remaining = [e for e in entries if e.get("name") != name]
    VOICES_JSON.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")

    wav_path = BASE_DIR / match.get("wav", "")
    try:
        if wav_path.exists():
            wav_path.unlink()
    except Exception:
        # Запись из cosyvoice_voices.json мы уже удалили - это главное
        # (голос перестанет быть доступен для синтеза); если сам файл не
        # удалился (например, занят чем-то), просто предупреждаем в лог,
        # но не считаем это ошибкой всей операции.
        logger.warning(f"Профиль {name!r} удалён из списка, но не удалось удалить файл {wav_path}:\n"
                        + traceback.format_exc())

    _load_voice_profiles()
    logger.info(f"Удалён профиль голоса: {name!r}")
    return {"ok": True, "name": name, "voices": list(voice_profiles.keys())}


_ruaccent = None
_ruaccent_load_failed = False


def _get_ruaccent():
    """Лениво загружает RUAccent (https://github.com/Den4ikAI/ruaccent) —
    расставляет ударения в русском тексте символом "+" перед ударной
    гласной (например "зам+ок") прямо перед тем, как текст уходит в XTTS.
    У XTTS-v2 нет отдельного модуля для расстановки ударений в русском (в
    отличие от английского), из-за чего она сама иногда ставит их не туда;
    русские TTS-корпуса, на которых её обучали, размечены именно в этом
    "+"-формате — поэтому предварительная разметка текста этим же символом
    заметно помогает произношению. Если библиотека не установлена или не
    загрузилась — тихо синтезируем без разметки (не блокируем синтез)."""
    global _ruaccent, _ruaccent_load_failed
    if _ruaccent_load_failed:
        return None
    if _ruaccent is None:
        try:
            from ruaccent import RUAccent
            logger.info("Загружаю RUAccent (расстановка ударений для русского текста)...")
            accentizer = RUAccent()
            accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True)
            _ruaccent = accentizer
            logger.info("RUAccent загружен.")
        except Exception:
            logger.warning("Не удалось загрузить RUAccent - буду синтезировать без "
                            "расстановки ударений (см. install.bat, пакет ruaccent):\n"
                            + traceback.format_exc())
            _ruaccent_load_failed = True
            return None
    return _ruaccent


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers_to_words_ru(text: str) -> str:
    """Заменяет числа словами (num2words) - ни F5-TTS, ни XTTS не умеют
    сами разворачивать цифры в русский текст (в отличие, например, от
    Yandex SpeechKit): цифры либо молча пропускаются моделью, либо
    произносятся невнятно/неправильно. Ловит и числа, слипшиеся с
    буквами/знаками ("20-летие", "5%", "3,5") - заменяется только
    цифровая часть. Если num2words не установлен в этом окружении -
    возвращает текст без изменений (не блокирует синтез)."""
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


def _add_stress_marks(text: str) -> str:
    if XTTS_LANGUAGE != "ru":
        return text
    accentizer = _get_ruaccent()
    if accentizer is None:
        return text
    try:
        return accentizer.process_all(text)
    except Exception:
        logger.warning("RUAccent не смог обработать этот текст - использую как есть:\n"
                        + traceback.format_exc())
        return text


def _persist_profile_text(voice: str, text: str):
    """Записывает автоматически распознанный текст образца обратно в
    cosyvoice_voices.json, чтобы Whisper не гонялся по этому образцу заново
    при каждом синтезе — только один раз, при первом использовании
    голоса."""
    try:
        entries = json.loads(VOICES_JSON.read_text(encoding="utf-8")) if VOICES_JSON.exists() else []
        for e in entries:
            if e.get("name") == voice:
                e["text"] = text
        VOICES_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning(f"Не удалось сохранить распознанный текст образца для {voice!r} в "
                        "cosyvoice_voices.json (не критично - распознается заново в следующий "
                        "раз):\n" + traceback.format_exc())


def _ensure_profile_text(voice: str, profile: dict) -> str:
    """F5-TTS для качественного клонирования нужен текст (транскрипт) того,
    что сказано в референсном аудио - в отличие от XTTS, ей текст образца не
    обязателен. Если пользователь его не указал явно при добавлении голоса
    (и не нажал "Распознать текст" в программе) - распознаём его сами тем
    же Whisper, что использует /transcribe, и запоминаем результат, чтобы
    не распознавать заново при каждой фразе."""
    if profile.get("text"):
        return profile["text"]
    try:
        arr = _load_wav_for_whisper(profile["wav_path"])
        model = _get_whisper_asr_model()
        result = model.transcribe(arr, language="ru", fp16=False)
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
    """Синтезирует текст выбранным профилем голоса через F5-TTS или XTTS
    (см. ENGINE), возвращает float32 numpy-массив (моно, частота —
    synth_sample_rate)."""
    profile = voice_profiles.get(voice)
    if profile is None:
        raise RuntimeError(f"Профиль голоса {voice!r} не найден (доступны: {list(voice_profiles.keys())})")

    prompt_wav_path = profile["wav_path"]
    text = _numbers_to_words_ru(text)

    if ENGINE == "f5":
        # Если распознать текст образца не удалось (см. _ensure_profile_text)
        # - "" туда НЕ передаём: сама библиотека f5-tts, получив пустой
        # ref_text, пытается распознать его САМА через свой внутренний
        # ASR-пайплайн (transformers), который тоже требует системный
        # ffmpeg и падает точно так же - т.е. без этой подстраховки один
        # сбой Whisper на нашей стороне тянет за собой второй, уже не
        # перехватываемый нами. Один пробел - не пустая строка, поэтому
        # f5-tts не станет распознавать сама, а просто чуть хуже выровняет
        # клонирование (в этом крайнем случае текста всё равно ни у кого нет).
        ref_text = _ensure_profile_text(voice, profile) or " "
        wav, _sr, _spec = cosyvoice_model.infer(
            ref_file=prompt_wav_path, ref_text=ref_text, gen_text=text,
            remove_silence=False, seed=None,
        )
        speech = np.asarray(wav, dtype=np.float32)
    else:  # xtts
        # RUAccent-разметка ударений ("+" перед ударной гласной) помогает
        # только XTTS - F5-TTS-Russian обучена на обычном тексте без такой
        # разметки (Common Voice и т.п. её не содержат), добавлять её туда
        # было бы контрпродуктивно, поэтому марки ставим только для XTTS.
        marked_text = _add_stress_marks(text)
        wav = cosyvoice_model.tts(
            text=marked_text, speaker_wav=prompt_wav_path, language=XTTS_LANGUAGE, split_sentences=True
        )
        speech = np.asarray(wav, dtype=np.float32)

    if speech.size == 0:
        raise RuntimeError(f"{ENGINE} не вернул аудио (пустой результат)")
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

    logger.info(f"[/getwav] engine={ENGINE} voice={voice!r}, {len(text_to_speech)} симв.: {text_to_speech[:200]}")

    try:
        # Синтез на GPU занимает реальное время (секунды) — выполняем в
        # отдельном потоке, чтобы не блокировать событийный цикл uvicorn.
        audio = await run_in_threadpool(_synthesize, text_to_speech, voice)
    except Exception:
        logger.error("[/getwav] Synthesis failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"{ENGINE} synthesis failed, see server log")

    model_sr = synth_sample_rate
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != model_sr:
        audio_t = torchaudio.functional.resample(audio_t, model_sr, sample_rate)

    wav_bytes = _write_wav_bytes(audio_t, sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("COSYVOICE_PORT", "5011"))
    try:
        uvicorn.run("cosyvoice_rest_service:app", host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Сервис остановлен пользователем (Ctrl+C)")
    except SystemExit:
        raise
    except Exception:
        logger.error("Процесс сервиса аварийно завершился:\n" + traceback.format_exc())
        sys.exit(1)
