from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool
import uvicorn
import torch
import numpy as np
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
 
from silero_config import DEFAULT_SILERO_MODEL, load_silero_package

# Модель по умолчанию — v5_6_ru (последняя; прежняя v5_5_ru). Переопределить: set SILERO_MODEL=v5_4_ru
MODEL_ID = os.environ.get("SILERO_MODEL", DEFAULT_SILERO_MODEL)
 
 
@app.on_event("startup")
async def startup_event():
    global model
 
    device = torch.device('cpu')
    torch.set_num_threads(4)

    try:
        logger.info(f"Loading Silero TTS model {MODEL_ID}...")
        model = load_silero_package(MODEL_ID, device)
        logger.info(f"TTS Model {MODEL_ID} loaded successfully")
    except Exception:
        logger.error("Failed to load TTS model:\n" + traceback.format_exc())
        model = None
 
# A leading "-" is a minus sign only when it is not glued to a word/number ("5-6" is a range).
_NUMBER_RE = re.compile(r"(?P<neg>(?<![\w-])-)?(?P<num>\d+(?:[.,]\d+)?)(?P<hy>-(?=[^\W\d_]))?")


def preprocess_text(text):
    """Преобразует цифры в словесную форму (num2words). Заменяется только цифровая часть
    токенов ("20-летие", "5%", "3,5"). Слова ВСЕГДА отделяются пробелами от соседних
    букв/цифр ("10лет" -> "десять лет"): без отступов Silero склеивает число с соседним
    словом и "проглатывает" его. Дефис между числом и буквами ("20-летие") заменяется пробелом."""
    def _repl(m):
        raw = m.group("num")
        try:
            if "," in raw or "." in raw:
                words = num2words(float(raw.replace(",", ".")), lang="ru")
            else:
                words = num2words(int(raw), lang="ru")
        except Exception:
            logger.warning(f"Failed to convert number {raw!r} to words:\n" + traceback.format_exc())
            return m.group(0)
        if m.group("neg"):
            words = "минус " + words
        s, a, b = m.string, m.start(), m.end()
        left = " " if a > 0 and re.match(r"\w", s[a - 1]) else ""
        right = " " if m.group("hy") or (b < len(s) and re.match(r"\w", s[b])) else ""
        return left + words + right
    return _NUMBER_RE.sub(_repl, text)
 
 
@app.get(
    "/getwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response
)
async def getwav(text_to_speech: str, speaker: str = "xenia", sample_rate: int = 24000,
                 strip_mono: bool = True, server_accent: bool = True):
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded")
 
    preprocessed_text = preprocess_text(text_to_speech)
    accented_text = _finalize_text(preprocessed_text, strip_mono)
    logger.info(f"[/getwav] Text sent to Silero: {accented_text[:200]}")
 
    try:
        # model.save_wav — тяжёлая синхронная CPU-операция (секунды на
        # фрагмент); выполняем в отдельном потоке, чтобы не блокировать
        # событийный цикл uvicorn на всё это время (иначе параллельные
        # запросы/health-check зависают, что тоже выглядит как "сервис
        # завис").
        path = await run_in_threadpool(model.save_wav, text=accented_text,
                                        speaker=speaker, sample_rate=sample_rate,
                                        **_silero_own_guess(server_accent))
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
 
 
# --------------------------------------------------------------------------
# The service does NOT accentuate. The client (fb2_reader.py) places every "+" itself
# (user dictionary -> silero-stress -> grammar gate) and sends the marked text; the service
# only turns digits into words, strips useless marks from one-vowel words and synthesizes
# exactly what it was given. server_accent=true means "let Silero's own put_accent/put_yo guess"
# (used when the client could not accentuate, e.g. silero-stress is not installed).
# --------------------------------------------------------------------------

# Односложные слова, которым метка нужна: без неё Silero читает «лет» как «лёт».
_MONO_KEEP_STRESS = frozenset({"лет"})
_MONO_STRESS_RE = re.compile(r"[А-Яа-яЁё+]*\+[А-Яа-яЁё+]*")


def _strip_monosyllable_stress(text: str) -> str:
    """Removes "+" from one-vowel words ("+он", "н+ет", "д+ва"): the stress is
    unambiguous there and the extra mark makes Silero stretch/emphasise the
    word (mirrors fb2_reader.strip_monosyllable_stress). Words with "ё" keep their mark
    ("вс+ё", "ч+ём") — "ё" is always stressed."""
    def repl(m: "re.Match") -> str:
        w = m.group(0)
        plain = w.replace("+", "")
        if "ё" in plain.lower():
            return w
        if plain.lower() in _MONO_KEEP_STRESS:
            return w
        if len(re.findall(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", plain)) <= 1:
            return plain
        return w
    return _MONO_STRESS_RE.sub(repl, text)


def _finalize_text(text: str, strip_mono: bool = True) -> str:
    """Pass-through for the client's marks; only one-vowel words lose a useless "+"."""
    return _strip_monosyllable_stress(text) if strip_mono else text


def _prepare_plain(text: str, strip_mono: bool = True) -> str:
    """Plain (non-SSML) text: digits -> words, then the pass-through."""
    return _finalize_text(preprocess_text(text), strip_mono)
 
 
def prepare_ssml_text_nodes(ssml: str, strip_mono: bool = True) -> str:
    """Digits -> words in the text nodes of an SSML document only (tags <speak>/<p>/<s>/<break .../>/
    <prosody ...> are untouched); the client's "+" marks pass through unchanged. Works for the SSML
    built by the service itself (raw_ssml=false) and for client SSML (raw_ssml=true).
    Digits must ALWAYS become words here: Silero swallows raw digits."""
    parts = _TAG_SPLIT_RE.split(ssml)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            out.append(part)
        else:
            out.append(_prepare_plain(part, strip_mono))
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
        return f'<s><prosody pitch="+15%">{escaped_sentence}</prosody></s>'
    if stripped.endswith("!"):
        return f'<s><prosody pitch="+10%" rate="107%">{escaped_sentence}</prosody></s>'
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
 
 
def _silero_own_guess(server_accent: bool) -> dict:
    """server_accent=False: the client already placed every "+" (user dictionary / silero-stress /
    grammar gate), so Silero's built-in guessing is switched OFF and it only executes our marks."""
    return {} if server_accent else {"put_accent": False, "put_yo": False}


async def _synthesize_with_fallbacks(ssml: str, speaker: str, sample_rate: int,
                                     server_accent: bool = True) -> tuple:
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
                                            speaker=speaker, sample_rate=sample_rate,
                                            **_silero_own_guess(server_accent))
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
                                        speaker=speaker, sample_rate=sample_rate,
                                        **_silero_own_guess(server_accent))
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
                      emphasize: bool = True, strip_mono: bool = True, server_accent: bool = True):
    """Синтез речи с интонационными паузами и вопросительной/восклицательной
    интонацией. Всегда пытается вернуть аудио, даже если сложная SSML-
    разметка не синтезируется — см. _synthesize_with_fallbacks.
 
    - text_to_speech: обычный текст (будет автоматически превращён в SSML
      с паузами на знаках препинания и границах предложений/абзацев), либо
      уже готовый SSML-документ, если raw_ssml=true.
    - raw_ssml: если true, text_to_speech передаётся моделью как SSML
      (ожидается валидный <speak>...</speak>); ударения всё равно
      расставляются сервером перед синтезом (см. prepare_ssml_text_nodes).
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
 
    ssml = prepare_ssml_text_nodes(ssml, strip_mono)
 
    logger.info(f"SSML for synthesis ({len(ssml)} chars): {ssml[:300]}{'...' if len(ssml) > 300 else ''}")
 
    try:
        path, level = await _synthesize_with_fallbacks(ssml, speaker, sample_rate, server_accent)
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