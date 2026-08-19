from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool
import uvicorn
import torch
import numpy as np
from ruaccent import RUAccent
import os
import re
import sys
import logging
import traceback
from num2words import num2words  # Библиотека для преобразования чисел в слова
from xml.sax.saxutils import escape as xml_escape, unescape as xml_unescape
 
app = FastAPI()
 
version = "1.4"
model = None
accentizer = None
 
# --------------------------------------------------------------------------
# Логирование: всё пишем и в консоль, и в файл silero_rest_service.log —
# чтобы при "много пропусков" можно было посмотреть, что реально пошло не
# так (полный traceback, а не только короткое сообщение).
#
# ВАЖНО: при запуске "python3 silero_rest_service.py" модуль импортируется
# дважды — один раз как __main__, второй раз как "silero_rest_service"
# (когда uvicorn.run("silero_rest_service:app", ...) сам его импортирует).
# Без защиты ниже это задваивает все строки лога. Проверяем logger.handlers,
# чтобы обработчики добавлялись только один раз.
# --------------------------------------------------------------------------
LOG_FILE = "silero_rest_service.log"
logger = logging.getLogger("silero_rest_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _console_handler.setFormatter(_formatter)
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)
    logger.propagate = False
 
 
@app.exception_handler(Exception)
async def _log_unhandled_exceptions(request: Request, exc: Exception):
    """Ловит вообще любое необработанное исключение в любом эндпоинте и
    пишет полный traceback в лог, вместо того чтобы уронить процесс без
    единой записи (именно так выглядели "пропуски без причины" — сервис
    падал/перезапускался, а в логе не оставалось ни строки)."""
    logger.error(f"Необработанное исключение на {request.method} {request.url.path}:\n"
                 + traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка сервера: {exc}"})
 
# v5_5_ru — актуальная модель Silero для русского: умеет автоматически
# расставлять ударения, разрешать омографы и строить вопросительную
# интонацию без дополнительной разметки (в отличие от v4_ru). API
# идентичен v4 (те же голоса aidar/baya/kseniya/xenia/eugene,
# те же model.save_wav/apply_tts), поэтому это прямая замена.
# Имя файла отличается от старого silero_model.pt специально — чтобы
# не подхватить случайно закэшированную старую модель v4.
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
MODEL_LOCAL_FILE = "silero_model_v5_5_ru.pt"
 
 
def _patch_ruaccent_token_type_ids(accentizer_obj):
    """
    Чинит несовместимость установленной версии пакета ruaccent с
    onnxruntime/transformers: ONNX-модель расстановки ударений
    (ruaccent/accent_model.py, метод put_accent -> session.run) требует
    на входе token_type_ids, а код ruaccent передаёт только input_ids и
    attention_mask. Из-за этого КАЖДЫЙ вызов accentizer.process_all(...)
    падал с:
      ValueError: Required inputs (['token_type_ids']) are missing from
      input feed (['input_ids', 'attention_mask']).
    что и приводило к HTTP 500 на 100% фрагментов (и SSML, и обычный
    текст — оба пути идут через один и тот же accentizer).
 
    Патчим не сам пакет (его файлы на диске трогать нельзя — слетит при
    обновлении), а конкретный ONNX InferenceSession уже загруженной
    модели: оборачиваем session.run так, чтобы он сам подставлял нулевой
    token_type_ids той же формы, что input_ids, если модель его требует,
    а его не передали.
    """
    try:
        # Внутри RUAccent.accent_model лежит объект с полем .session —
        # это onnxruntime.InferenceSession для put_accent().
        inner = getattr(accentizer_obj, "accent_model", None)
        if inner is None or not hasattr(inner, "session"):
            logger.warning(
                "Не удалось найти accent_model.session у RUAccent — "
                "пропускаю патч token_type_ids (возможно, изменилась "
                "внутренняя структура пакета ruaccent)."
            )
            return
 
        session = inner.session
        required_inputs = {i.name for i in session.get_inputs()}
        if "token_type_ids" not in required_inputs:
            # Установленная модель не требует token_type_ids — патч не нужен.
            return
 
        original_run = session.run
 
        def patched_run(output_names, input_feed, run_options=None):
            if "token_type_ids" not in input_feed and "input_ids" in input_feed:
                input_feed = dict(input_feed)
                input_feed["token_type_ids"] = np.zeros_like(input_feed["input_ids"])
            return original_run(output_names, input_feed, run_options)
 
        session.run = patched_run
        logger.info(
            "Применён патч token_type_ids для ONNX-модели ударений ruaccent "
            "(иначе синтез падал бы на каждом фрагменте)."
        )
    except Exception:
        logger.warning(
            "Не удалось применить патч token_type_ids для ruaccent — "
            "ошибки 'Required inputs ([token_type_ids])' могут повториться:\n"
            + traceback.format_exc()
        )
 
 
@app.on_event("startup")
async def startup_event():
    global model, accentizer
 
    device = torch.device('cpu')
    torch.set_num_threads(4)
 
    if not os.path.isfile(MODEL_LOCAL_FILE):
        logger.info(f"Downloading Silero TTS model ({MODEL_LOCAL_FILE})...")
        torch.hub.download_url_to_file(MODEL_URL, MODEL_LOCAL_FILE)
 
    try:
        model = torch.package.PackageImporter(MODEL_LOCAL_FILE).load_pickle("tts_models", "model")
        model.to(device)
        logger.info("TTS Model loaded successfully")
    except Exception:
        logger.error("Failed to load TTS model:\n" + traceback.format_exc())
        model = None
 
    try:
        accentizer = RUAccent()
        # 'turbo' больше не существует как значение omograph_model_size в
        # актуальных версиях ruaccent — актуальные варианты: tiny, tiny2,
        # tiny2.1, turbo2, turbo3, turbo3.1, big_poetry. Оставляем turbo3.1
        # (лучшее качество омографов) — сама проблема с token_type_ids была
        # не в omograph-модели, а в ГЛАВНОЙ модели ударений ruaccent
        # (accent_model.py: put_accent), которая используется всегда,
        # независимо от omograph_model_size. Исправляем её ниже patch'ем.
        accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)
        _patch_ruaccent_token_type_ids(accentizer)
        logger.info("RUAccent model loaded successfully")
    except Exception:
        logger.error("Failed to load RUAccent model:\n" + traceback.format_exc())
        accentizer = None
 
 
def preprocess_text(text):
    """Преобразует цифры в текстовый формат."""
    words = text.split()
    processed_words = []
    for word in words:
        if word.isdigit():
            try:
                word = num2words(int(word), lang='ru')
            except Exception:
                logger.warning(f"Failed to convert number {word!r} to words:\n" + traceback.format_exc())
        processed_words.append(word)
    return " ".join(processed_words)
 
 
@app.get(
    "/getwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response
)
async def getwav(text_to_speech: str, speaker: str = "xenia", sample_rate: int = 24000):
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded")
 
    preprocessed_text = preprocess_text(text_to_speech)
    accented_text = accentizer.process_all(preprocessed_text) if accentizer else preprocessed_text
    logger.info(f"[/getwav] Text after accent processing: {accented_text[:200]}")
 
    try:
        # model.save_wav — тяжёлая синхронная CPU-операция (секунды на
        # фрагмент); выполняем в отдельном потоке, чтобы не блокировать
        # событийный цикл uvicorn на всё это время (иначе параллельные
        # запросы/health-check зависают, что тоже выглядит как "сервис
        # завис").
        path = await run_in_threadpool(model.save_wav, text=accented_text,
                                        speaker=speaker, sample_rate=sample_rate)
    except Exception:
        logger.error("[/getwav] Synthesis failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=400, detail="TTS synthesis failed, see server log")
 
    with open(path, "rb") as in_file:
        data = in_file.read()
 
    return Response(content=data, media_type="audio/wav")
 
 
# --------------------------------------------------------------------------
# SSML: интонационные паузы, вопросы/восклицания, границы предложений/абзацев
# --------------------------------------------------------------------------
 
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SOFT_PAUSE_RE = re.compile(r"([,;:]|—|--)\s+")
_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_PROSODY_TAG_RE = re.compile(r"</?prosody[^>]*>")
 
 
def _accent_plain(text: str) -> str:
    """Прогоняет обычный (не-SSML) текст через preprocess_text + ударения."""
    preprocessed = preprocess_text(text)
    return accentizer.process_all(preprocessed) if accentizer else preprocessed
 
 
def accentize_ssml_text_nodes(ssml: str) -> str:
    """Расставляет ударения (через RUAccent) только в текстовых узлах SSML,
    не трогая сами теги <speak>/<p>/<s>/<break .../>/<prosody ...>.
 
    Нужно и для SSML, построенного самим сервисом (raw_ssml=false), и для
    готового SSML, присланного клиентом (raw_ssml=true, как это делает
    fb2_reader.py) — в обоих случаях без этого шага модель могла бы не
    получить + -ударений и хуже справляться с омографами (замо́к/за́мок и т.п.).
    """
    if accentizer is None:
        return ssml
    parts = _TAG_SPLIT_RE.split(ssml)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            out.append(part)
        else:
            out.append(_accent_plain(part))
    return "".join(out)
 
 
def strip_ssml_to_plain_text(ssml: str) -> str:
    """Убирает все SSML-теги и раскрывает XML-сущности (&amp; и т.п.),
    оставляя обычный читаемый текст — используется как аварийный вариант,
    когда синтез самого SSML не удаётся."""
    text = _ANY_TAG_RE.sub(" ", ssml)
    text = xml_unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def _prosody_wrap(escaped_sentence: str, original_sentence: str,
                   emphasize: bool) -> str:
    """Слегка усиливает интонацию вопросительных и восклицательных
    предложений через <prosody pitch=... rate=...>, поверх того, что модель
    и так делает по самому знаку "?"/"!" — помогает на репликах в кавычках
    и коротких восклицаниях, где интонация иначе может звучать плоско."""
    if not emphasize:
        return f"<s>{escaped_sentence}</s>"
    stripped = original_sentence.rstrip()
    if stripped.endswith("?"):
        return f'<s><prosody pitch="high">{escaped_sentence}</prosody></s>'
    if stripped.endswith("!"):
        return f'<s><prosody pitch="high" rate="fast">{escaped_sentence}</prosody></s>'
    return f"<s>{escaped_sentence}</s>"
 
 
def build_ssml(text: str, sentence_break_ms: int = 320,
                paragraph_break_ms: int = 550, comma_break_ms: int = 180,
                emphasize: bool = True) -> str:
    """Строит SSML-документ для Silero из обычного текста.
 
    * Каждый абзац -> <p>, между абзацами длинная пауза.
    * Каждое предложение -> <s>, между предложениями пауза покороче;
      вопросительный/восклицательный знак сохраняется как есть, а сверху
      (если emphasize=True) добавляется <prosody> с повышенным pitch —
      усиливает вопросительную/восклицательную интонацию.
    * Внутри предложения после запятых/тире/двоеточий/точек с запятой —
      небольшая пауза <break/>, имитирующая естественную интонационную
      паузу при чтении.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
 
    p_chunks = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        s_chunks = []
        for sent in sentences:
            escaped = xml_escape(sent)
            # небольшие паузы внутри предложения на знаках препинания
            escaped = _SOFT_PAUSE_RE.sub(
                lambda m: f'{xml_escape(m.group(1))}<break time="{comma_break_ms}ms"/> ',
                escaped,
            )
            s_chunks.append(_prosody_wrap(escaped, sent, emphasize))
        if s_chunks:
            joiner = f'<break time="{sentence_break_ms}ms"/>'
            p_chunks.append("<p>" + joiner.join(s_chunks) + "</p>")
 
    joiner = f'<break time="{paragraph_break_ms}ms"/>'
    body = joiner.join(p_chunks)
    return f"<speak>{body}</speak>"
 
 
async def _synthesize_with_fallbacks(ssml: str, speaker: str, sample_rate: int) -> tuple:
    """Пытается синтезировать речь по SSML в несколько шагов, с понижением
    сложности разметки, чтобы в итоге всегда что-то сгенерировалось:
 
      1) исходный SSML как есть;
      2) тот же SSML, но без тегов <prosody> (на случай, если конкретная
         сборка модели не понимает pitch/rate — тогда теряется только
         усиленная интонация, паузы и структура предложений остаются);
      3) обычный текст без SSML вообще (полностью убираем разметку) —
         последний рубеж, чтобы аудиофайл в любом случае получился.
 
    Возвращает (путь_к_wav, использованный_уровень: 1/2/3).
    """
    attempts = [
        ("as-is", ssml),
        ("no-prosody", _PROSODY_TAG_RE.sub("", ssml)),
    ]
    last_exc = None
    for label, candidate_ssml in attempts:
        try:
            path = await run_in_threadpool(model.save_wav, ssml_text=candidate_ssml,
                                            speaker=speaker, sample_rate=sample_rate)
            if label != "as-is":
                logger.warning(f"SSML synthesis succeeded only after fallback '{label}'")
            return path, label
        except Exception as e:
            last_exc = e
            logger.error(f"SSML synthesis attempt '{label}' failed:\n" + traceback.format_exc())
 
    # Последний рубеж — совсем без SSML, обычным текстом.
    try:
        plain = strip_ssml_to_plain_text(ssml)
        path = await run_in_threadpool(model.save_wav, text=plain,
                                        speaker=speaker, sample_rate=sample_rate)
        logger.warning("SSML synthesis failed completely, fell back to plain text")
        return path, "plain-text-fallback"
    except Exception:
        logger.error("Plain-text fallback synthesis also failed:\n" + traceback.format_exc())
        raise last_exc
 
 
@app.get(
    "/getssmlwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response
)
async def getssmlwav(text_to_speech: str, speaker: str = "xenia", sample_rate: int = 24000,
                      sentence_break_ms: int = 320, paragraph_break_ms: int = 550,
                      comma_break_ms: int = 180, raw_ssml: bool = False,
                      emphasize: bool = True):
    """Синтез речи с интонационными паузами и вопросительной/восклицательной
    интонацией. Всегда пытается вернуть аудио, даже если сложная SSML-
    разметка не синтезируется — см. _synthesize_with_fallbacks.
 
    - text_to_speech: обычный текст (будет автоматически превращён в SSML
      с паузами на знаках препинания и границах предложений/абзацев), либо
      уже готовый SSML-документ, если raw_ssml=true.
    - raw_ssml: если true, text_to_speech передаётся моделью как SSML
      (ожидается валидный <speak>...</speak>); ударения всё равно
      расставляются сервером перед синтезом (см. accentize_ssml_text_nodes).
    - emphasize: усиливать ли pitch/rate у вопросительных и восклицательных
      предложений (только для raw_ssml=false — при raw_ssml=true разметку
      уже задаёт клиент).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded")
 
    if raw_ssml:
        ssml = text_to_speech
    else:
        ssml = build_ssml(
            text_to_speech,
            sentence_break_ms=sentence_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
            emphasize=emphasize,
        )
 
    ssml = accentize_ssml_text_nodes(ssml)
 
    logger.info(f"SSML for synthesis ({len(ssml)} chars): {ssml[:300]}{'...' if len(ssml) > 300 else ''}")
 
    try:
        path, level = await _synthesize_with_fallbacks(ssml, speaker, sample_rate)
    except Exception as e:
        logger.error("All synthesis attempts (including plain-text fallback) failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"SSML synthesis failed even with fallbacks: {e}")
 
    with open(path, "rb") as in_file:
        data = in_file.read()
 
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"X-Synthesis-Level": level},
    )
 
 
if __name__ == "__main__":
    try:
        uvicorn.run("silero_rest_service:app", host="0.0.0.0", port=5010, log_level="info")
    except KeyboardInterrupt:
        logger.info("Сервис остановлен пользователем (Ctrl+C)")
    except SystemExit:
        raise
    except Exception:
        logger.error("Процесс сервиса аварийно завершился:\n" + traceback.format_exc())
        sys.exit(1)
    # Если в логе НЕТ ни одной из строк выше ("остановлен пользователем" /
    # "аварийно завершился"), а сервис всё равно перестал отвечать — значит
    # процесс убило что-то за пределами Python (например, антивирус,
    # нехватка памяти, или сбой в нативном коде torch), traceback для
    # такого не напечатать. В этом случае надёжнее всего запускать сервис
    # через простой скрипт-обёртку, которая перезапускает его при падении,
    # например run_service_forever.bat (см. README.md).