#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fb2_reader.py — озвучивание книг в формате FB2 на русском языке.

Возможности:
  * извлечение текста из .fb2 (и .fb2.zip) по главам;
  * озвучка четырьмя способами:
      1) silero      — (рекомендуется, если torch не нужен на этой машине)
                       нейросетевой синтез Silero TTS локально через torch.hub.
                       Живая интонация, естественные паузы на знаках
                       препинания, ударения расставляются автоматически.
                       Модель скачивается один раз (~50-100 МБ), дальше
                       работает офлайн. Требует torch.
      2) silero_rest — синтез через удалённый/локальный Silero-REST-Service
                       (https://github.com/Flokss/Silero-REST-Service).
                       Текст главы автоматически превращается в SSML с
                       интонационными паузами на запятых/тире/двоеточиях,
                       более длинными паузами между предложениями и
                       абзацами, и с сохранением интонации вопросительных
                       и восклицательных предложений. Требует запущенного
                       сервиса (см. --rest-url) и пакет requests.
                       Для пауз на сервисе должен быть эндпоинт /getssmlwav
                       (см. патч в этом репозитории).
      3) online      — через gTTS (Google Text-to-Speech), нужен интернет
                       на каждый запуск, голос неплохой, но менее выразительный.
      4) offline     — через pyttsx3 и системный TTS (espeak и т.п.), самое
                       низкое качество, зато совсем без интернета и без torch.

Установка зависимостей:
  # для локального режима silero (лучшее качество голоса):
  pip install silero torch torchaudio omegaconf numpy
  # silero — официальный пакет; без него модель загрузится через torch.hub

  # для режима silero_rest (клиент к REST-сервису):
  pip install requests numpy

  # для остальных режимов:
  pip install gTTS pyttsx3 pygame lxml pydub

  # необязательно, но рекомендуется — красивый прогресс-бар по фрагментам
  # главы вместо простого текстового счётчика (без него используется
  # встроенная упрощённая замена):
  pip install tqdm

Для offline-режима (pyttsx3) на Linux дополнительно нужен espeak-ng:
  sudo apt install espeak-ng

Повторный запуск: если для главы уже есть готовый .mp3/.wav файл,
озвученный с теми же параметрами (голос, частота дискретизации, паузы и
сам текст главы не изменились), она пропускается — файл не переозвучивается
заново. Признак хранится в скрытом файле рядом с аудио (например,
"001_Глава 1.wav.meta.json"). Если поменять --speaker, --sample-rate,
паузы или сам текст книги — соответствующая глава переозвучится заново.

Примеры запуска:
  # Лучшее качество: нейросетевой голос Silero (локально), сохранить в wav
  python3 fb2_reader.py book.fb2 --mode silero --speaker xenia --outdir audiobook

  # То же самое, но сразу слушать по мере озвучки
  python3 fb2_reader.py book.fb2 --mode silero --play

  # Через Silero-REST-Service с интонационными паузами (SSML), сервис
  # запущен на localhost:5010 (см. install.sh из Silero-REST-Service)
  python3 fb2_reader.py book.fb2 --mode silero_rest --rest-url http://localhost:5010 \
      --speaker xenia --outdir audiobook

  # Озвучить книгу через gTTS и сразу проигрывать
  python3 fb2_reader.py book.fb2 --mode online --play

  # Офлайн pyttsx3-режим, начиная с 3-й главы
  python3 fb2_reader.py book.fb2 --mode offline --start 3

  # Просто посмотреть список глав, ничего не озвучивая
  python3 fb2_reader.py book.fb2 --list

  # Графический интерфейс (любой из способов):
  python3 fb2_reader.py              # без аргументов — откроется окно
  python3 fb2_reader.py --gui
  python3 fb2_reader.py --gui book.fb2
  python3 fb2_reader_gui.py
  python3 fb2_reader_gui.py book.fb2

  # Командная строка (как раньше):
  python3 fb2_reader.py book.fb2 --mode silero --speaker xenia --outdir audiobook

Доступные модели Silero (--model): v5_5_ru (последняя, по умолчанию),
v5_4_ru, v5_ru, v4_ru (устаревшая, поддерживает random).
Голоса (--speaker): aidar (муж.), baya (жен.),
kseniya (жен.), xenia (жен.), eugene (муж.), random (только v4_ru).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Мини-замена tqdm на случай, если пакет не установлен —
        выводит прогресс вида 'Глава 3: 5/12' в одну строку."""

        def __init__(self, iterable=None, total=None, desc="", unit="", leave=True):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None else None)
            self.desc = desc
            self.unit = unit
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()

        def update(self, n=1):
            self.n += n
            suffix = f"/{self.total}" if self.total else ""
            print(f"\r  {self.desc}: {self.n}{suffix} {self.unit}".rstrip(), end="", flush=True)

        def close(self):
            print()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

try:
    from lxml import etree
    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAVE_LXML = False


FB2_NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}

from silero_config import (
    DEFAULT_SILERO_MODEL,
    SILERO_MODELS,
    load_silero_model,
    speaker_choices,
    speakers_for_model,
)

# Голоса Silero TTS для последней модели (используются в CLI и GUI)
SILERO_SPEAKERS = speaker_choices(DEFAULT_SILERO_MODEL)

TTS_MODES = {
    "silero": "Silero (локально, лучшее качество)",
    "silero_rest": "Silero REST (через сервис, SSML-паузы)",
    "cosyvoice": "CosyVoice (локально, GPU, клонирование голоса)",
    "piper": "Piper (локально, CPU, быстро, без клонирования)",
    "yandex": "Yandex SpeechKit (облако, платно, нужен API-ключ)",
    "online": "Google TTS (нужен интернет)",
    "offline": "Системный TTS (pyttsx3, без интернета)",
}

# Голоса Piper TTS — быстрый локальный CPU-движок без клонирования голоса
# (в отличие от cosyvoice), заметно легче и быстрее Silero/F5.
#
# Раньше здесь была попытка починить ударения через сторонний патч словаря
# espeak-ng + голос "mari" (github.com/mitrokun/espeak-ng-data) — откатил:
# на практике голос и версия словаря оказались рассинхронизированы (автор
# репозитория сам предупреждает, что дважды переделывал фонемные правила и
# это "сломало синтез на существующих моделях"; какой версии словаря
# соответствует именно чекпоинт mari-medium_epoch6399 — не выяснить
# надёжно без живого прослушивания каждой комбинации), в результате чего
# синтез превращался в неразборчивую кашу — хуже, чем просто неточные
# ударения. Вернулись к обычным голосам rhasspy/piper-voices без патча:
# ударения местами будут не там, зато речь разборчива.
PIPER_VOICES = {
    "irina": "Ирина (женский)",
    "denis": "Денис (мужской)",
    "dmitri": "Дмитрий (мужской)",
    "ruslan": "Руслан (мужской)",
}
PIPER_DEFAULT_VOICE = "irina"
PIPER_VOICES_REPO = "rhasspy/piper-voices"
PIPER_SAMPLE_RATE = 22050  # частота дискретизации у всех "medium"-голосов Piper

# CosyVoice — отдельный сервис (см. cosyvoice_rest_service.py, ставится
# отдельным install_cosyvoice.bat в свою .venv_cosyvoice, несовместимую по
# версиям torch/numpy с обычным .venv) — программа обращается к нему по
# HTTP, как к Silero REST. У CosyVoice нет готовых голосов "из коробки":
# вместо этого сервис хранит именованные "профили голоса" (короткий образец
# аудио + его текст), которыми можно клонировать любой голос — список
# профилей запрашивается у сервиса через /voices.
COSYVOICE_DEFAULT_REST_URL = "http://localhost:5011"

# Голоса Yandex SpeechKit (v1) на момент добавления — актуальный список
# стоит сверять в консоли Yandex Cloud, он время от времени пополняется
# новыми дикторами. Почти все голоса — ru-RU, но "lera" — украинский
# диктор и требует lang=uk-UK; если отправить его с lang=ru-RU, Yandex
# отвечает HTTP 400 (несовпадение voice/lang — самая частая причина этой
# ошибки, когда с ключом и балансом всё в порядке). Поэтому lang для
# запроса выбирается автоматически по голосу (см. YANDEX_VOICE_LANGS),
# а не задаётся вручную.
YANDEX_VOICES = {
    "alena": "Алёна (жен., нейтральный)",
    "filipp": "Филипп (муж., нейтральный)",
    "ermil": "Ермил (муж., с эмоциями good/neutral)",
    "jane": "Джейн (жен., с эмоциями good/neutral/evil)",
    "madirus": "Мадирус (муж.)",
    "omazh": "Омаж (жен.)",
    "zahar": "Захар (муж.)",
    "lera": "Лера (жен., укр.)",
}

# lang, обязательный для каждого голоса (Yandex требует точное совпадение
# voice/lang — иначе 400 Bad Request). У всех голосов, кроме lera, это
# ru-RU; для lera — uk-UK (украинский).
YANDEX_VOICE_LANGS = {
    "lera": "uk-UK",
}
YANDEX_DEFAULT_LANG = "ru-RU"


def yandex_lang_for_voice(voice: str) -> str:
    return YANDEX_VOICE_LANGS.get(voice, YANDEX_DEFAULT_LANG)


# --------------------------------------------------------------------------
# Извлечение текста из FB2
# --------------------------------------------------------------------------

def load_fb2_bytes(path: Path) -> bytes:
    """Читает содержимое .fb2 файла, поддерживает .fb2.zip архивы."""
    if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".fb2")]
            if not names:
                raise ValueError("В архиве не найден файл .fb2")
            return z.read(names[0])
    return path.read_bytes()


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def iter_text(element) -> str:
    """Рекурсивно собирает текст элемента, вставляя переносы строк
    между абзацами/заголовками."""
    parts = []
    block_tags = {"p", "title", "subtitle", "empty-line", "v", "cite",
                  "epigraph", "poem", "stanza"}

    def walk(el):
        tag = strip_ns(el.tag) if hasattr(el.tag, "split") else None
        if el.text and el.text.strip():
            parts.append(el.text.strip())
        for child in el:
            walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        if tag in block_tags:
            parts.append("\n")

    walk(element)
    text = " ".join(p for p in parts if p != "\n" or True)
    # аккуратно склеиваем: заменяем маркеры \n на реальные переносы
    text = re.sub(r"\s*\n\s*", "\n", " ".join(parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def parse_fb2(path: Path):
    """Возвращает (заголовок_книги, [(заголовок_главы, текст_главы), ...])."""
    data = load_fb2_bytes(path)
    parser = etree.XMLParser(recover=True) if HAVE_LXML else None
    root = etree.fromstring(data, parser=parser) if HAVE_LXML else etree.fromstring(data)

    def find_all(el, tagname):
        # работает и с lxml (namespace-aware), и с ElementTree (без namespaces)
        results = []
        for child in el.iter():
            if strip_ns(child.tag) == tagname:
                results.append(child)
        return results

    # Название книги
    book_title = None
    for title_info in find_all(root, "title-info"):
        for bt in find_all(title_info, "book-title"):
            book_title = "".join(bt.itertext()).strip()
            break
        break
    if not book_title:
        book_title = path.stem

    # Главы: ищем body -> section (верхнего уровня)
    bodies = [c for c in root if strip_ns(c.tag) == "body"]
    chapters = []

    def section_title(section):
        for child in section:
            if strip_ns(child.tag) == "title":
                return " ".join("".join(child.itertext()).split())
        return None

    def collect_sections(el, depth=0):
        found = []
        for child in el:
            if strip_ns(child.tag) == "section":
                found.append(child)
            elif strip_ns(child.tag) not in ("title",):
                found.extend(collect_sections(child, depth + 1))
        return found

    for body in bodies:
        # пропускаем body с notes/комментариями, если их несколько
        name_attr = body.get("name", "")
        if name_attr and name_attr.lower() in ("notes", "comments"):
            continue
        top_sections = [c for c in body if strip_ns(c.tag) == "section"]
        if not top_sections:
            top_sections = collect_sections(body)
        if not top_sections:
            text = iter_text(body)
            if text:
                chapters.append(("Текст", text))
            continue
        for i, sec in enumerate(top_sections, 1):
            title = section_title(sec) or f"Глава {i}"
            text = iter_text(sec)
            if text:
                chapters.append((title, text))

    if not chapters:
        raise ValueError("Не удалось извлечь текст из книги — возможно, файл повреждён.")

    return book_title, chapters


# --------------------------------------------------------------------------
# Озвучка
# --------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip()[:80] or "chapter"


try:
    from num2words import num2words
except ImportError:
    num2words = None

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def numbers_to_words_ru(text: str) -> str:
    """Заменяет числа словами (например "2026" -> "две тысячи двадцать
    шесть"). Без этого TTS-движки, у которых нет собственной нормализации
    чисел (локальный Silero, CosyVoice/F5-TTS/XTTS), либо молча пропускают
    цифры, либо читают их как попало по одной цифре. Ловит и числа,
    слипшиеся с буквами/знаками ("20-летие", "5%", "3,5") — заменяется
    только цифровая часть, остальное остаётся как было. Если пакет
    num2words не установлен — возвращает текст без изменений (не падает)."""
    if num2words is None or not text:
        return text

    def _repl(m: "re.Match") -> str:
        raw = m.group(0)
        try:
            if "," in raw or "." in raw:
                return num2words(float(raw.replace(",", ".")), lang="ru")
            return num2words(int(raw), lang="ru")
        except Exception:
            return raw

    return _NUMBER_RE.sub(_repl, text)


_HAS_LETTER_RE = re.compile(r"\w", re.UNICODE)


def _text_has_speakable_content(text: str) -> bool:
    """True, если в тексте есть хотя бы одна буква/цифра — т.е. есть что
    озвучивать. Фрагменты вроде "* * *" (разделитель сцен, часто
    встречается в fb2-книгах между сценами внутри главы) состоят только из
    астерисков/пробелов — движки TTS (особенно Silero) на таком тексте
    падают с непонятной ошибкой ("not enough values to unpack" и т.п.),
    хотя по сути там просто нечего произносить. Такие фрагменты вместо
    отправки в TTS сразу заменяются на короткую тишину — без ошибок и
    лишних сетевых запросов."""
    return bool(_HAS_LETTER_RE.search(text))


# --------------------------------------------------------------------------
# Пропуск уже озвученных глав (если файл уже сгенерирован с теми же
# параметрами — не переозвучиваем заново)
# --------------------------------------------------------------------------

def _meta_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".meta.json")


def _params_fingerprint(text: str, **params) -> dict:
    """Собирает "отпечаток" параметров озвучки главы: сам текст (хэш) плюс
    все параметры синтеза. Если хоть один параметр изменился (голос,
    частота, паузы и т.п.) или изменился текст главы — считаем, что файл
    нужно переозвучить."""
    fingerprint = dict(params)
    fingerprint["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return fingerprint


def _is_already_done(out_path: Path, fingerprint: dict) -> bool:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False
    meta_path = _meta_path(out_path)
    if not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return saved == fingerprint


def _save_fingerprint(out_path: Path, fingerprint: dict) -> None:
    _meta_path(out_path).write_text(
        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def synth_online(text: str, out_path: Path, lang="ru", desc: str = ""):
    from gtts import gTTS
    # gTTS ограничивает длину — разбиваем на куски по предложениям
    max_len = 4500
    chunks = split_text(text, max_len)
    if len(chunks) == 1:
        with tqdm(total=1, desc=desc or "синтез", unit="фрагм.") as bar:
            gTTS(text=chunks[0], lang=lang).save(str(out_path))
            bar.update(1)
        return
    # склеиваем несколько mp3 через pydub, если он есть; иначе сохраняем по частям
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        tmp_files = []
        for i, chunk in enumerate(tqdm(chunks, desc=desc or "синтез", unit="фрагм.")):
            tmp = out_path.with_suffix(f".part{i}.mp3")
            gTTS(text=chunk, lang=lang).save(str(tmp))
            combined += AudioSegment.from_mp3(str(tmp))
            tmp_files.append(tmp)
        combined.export(str(out_path), format="mp3")
        for t in tmp_files:
            t.unlink(missing_ok=True)
    except ImportError:
        # Без pydub — сохраняем части отдельными файлами
        for i, chunk in enumerate(tqdm(chunks, desc=desc or "синтез", unit="фрагм.")):
            part_path = out_path.with_name(f"{out_path.stem}_part{i+1}{out_path.suffix}")
            gTTS(text=chunk, lang=lang).save(str(part_path))
        print("  (pydub не установлен — глава сохранена частями *_partN.mp3)")


def split_text(text: str, max_len: int):
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_len:
            if cur:
                chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur:
        chunks.append(cur.strip())
    return chunks


_DIALOGUE_LINE_RE = re.compile(r"^[\-–—]\s")  # -, – или — в начале абзаца


def _split_dialogue_paragraphs(text: str):
    """Делит текст главы на абзацы (по \\n — так парсер fb2 разделяет
    <p>/<v>/... — см. iter_text) и помечает, какие из них похожи на
    реплику прямой речи: в русской прозе реплика почти всегда начинается
    с тире ("— Пойдём, — сказал он."). Это не точное определение "кто
    говорит" (для этого нужен полноценный NLP-анализ), а простая эвристика
    для разбивки: пусть хотя бы соседние реплики звучат разными голосами,
    а не одним и тем же монотонным диктором все 20+ часов книги."""
    paras = [p for p in text.split("\n") if p.strip()]
    return [(p, bool(_DIALOGUE_LINE_RE.match(p))) for p in paras]


def _group_paragraphs_by_voice(text: str, main_voice, dialogue_voices):
    """Делит текст главы на блоки (голос, текст_блока).

    Если dialogue_voices пуст/не задан — весь текст один блок с main_voice
    (без изменений).

    Если задан — реплики (абзацы, начинающиеся с тире) по очереди
    озвучиваются голосами из dialogue_voices (первая реплика — первым
    голосом из списка, вторая — вторым, и так по кругу), а весь остальной
    текст (авторская речь) — main_voice. Соседние абзацы одного и того же
    голоса (повествование) склеиваются в один блок, чтобы не плодить
    лишние обращения к TTS; каждая реплика — отдельный блок (даже если
    голос совпал бы со следующей), чтобы между репликами оставался
    естественный разрыв.

    Используется и для Yandex (там же режется на max_chars), и для Silero
    (там же режется на предложения/паузы через _segments_with_pauses) —
    сама логика разбивки на "чья это реплика" не зависит от режима TTS."""
    if not dialogue_voices:
        return [(main_voice, text)]

    paras = _split_dialogue_paragraphs(text)
    grouped = []  # [[voice, [paragraphs], is_dialogue], ...]
    di = 0
    for para, is_dialogue in paras:
        if is_dialogue:
            voice = dialogue_voices[di % len(dialogue_voices)]
            di += 1
            grouped.append([voice, [para], True])
        else:
            if grouped and grouped[-1][0] == main_voice and not grouped[-1][2]:
                grouped[-1][1].append(para)
            else:
                grouped.append([main_voice, [para], False])

    return [(voice, "\n".join(para_list)) for voice, para_list, _is_dialogue in grouped]


def _chunk_voice_groups(groups, max_chars: int):
    """Режет уже построенные группы (голос, текст_группы) на куски по
    max_chars — используется для Yandex/gTTS, где ограничение просто на
    длину запроса, в отличие от Silero, где куски ещё и по паузам на
    знаках препинания (см. run_silero)."""
    result = []
    for voice, group_text in groups:
        for chunk in split_text(group_text, max_chars):
            if chunk.strip():
                result.append((chunk, voice))
    return result


# --------------------------------------------------------------------------
# Атрибуция реплик по говорящему через LLM (Claude API) — необязательная,
# платная надстройка над простым чередованием голосов из
# _group_paragraphs_by_voice: вместо "первая реплика первым голосом, вторая
# вторым и по кругу" здесь модель читает главу и определяет, ПЕРСОНАЖ
# говорит каждую реплику, а голос закрепляется за именем персонажа на всю
# книгу (через сохраняемый на диске словарь), а не только на одну главу.
# --------------------------------------------------------------------------

# Два варианта поставщика атрибуции — платный (Anthropic Claude) и
# бесплатный (Google Gemini, есть щедрый бесплатный уровень без привязки
# карты — см. ATTRIBUTION_PROVIDERS ниже и подсказку в GUI). По умолчанию
# используется бесплатный Gemini.
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
GEMINI_GENERATE_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
YANDEXGPT_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

ATTRIBUTION_PROVIDERS = {
    "yandexgpt": {
        "title": "YandexGPT (бесплатный лимит, тот же ключ/каталог, что и SpeechKit)",
        "default_model": "yandexgpt-lite",
        "key_hint": "Работает из России без VPN, доступен грант на бесплатные запросы в "
                     "Yandex Cloud. Можно использовать тот же API-ключ и Folder ID, что и "
                     "для Yandex SpeechKit выше (скопируйте их сюда, если ещё не заполнено) "
                     "— консоль: yandex.cloud/ru/docs/ai-studio/quickstart.",
    },
    "gemini": {
        "title": "Google Gemini (бесплатно, ключ на aistudio.google.com)",
        "default_model": "gemini-2.5-flash-lite",
        "key_hint": "Бесплатный ключ: aistudio.google.com/apikey (карта не нужна). Из России "
                     "доступ у Google часто нестабилен/заблокирован из-за санкций — если не "
                     "работает, попробуйте YandexGPT. На бесплатном уровне Google может "
                     "использовать запросы для улучшения своих продуктов.",
    },
    "anthropic": {
        "title": "Anthropic Claude (платно, ключ на console.anthropic.com)",
        "default_model": "claude-haiku-4-5",
        "key_hint": "Платный ключ: console.anthropic.com/settings/keys (нужна привязанная "
                     "иностранная карта — из России оплатить напрямую обычно нельзя).",
    },
}
DEFAULT_ATTRIBUTION_PROVIDER = "yandexgpt"
DEFAULT_ATTRIBUTION_MODEL = ATTRIBUTION_PROVIDERS[DEFAULT_ATTRIBUTION_PROVIDER]["default_model"]

_ATTRIBUTION_TOOL_SCHEMA = {
    "name": "report_speakers",
    "description": "Сообщает, кто произносит каждую реплику прямой речи главы, по порядку.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Имя говорящего для каждой реплики прямой речи, в порядке "
                                "появления в тексте. Одно каноничное имя на персонажа "
                                "(например, всегда 'Иван Петров', не то 'Иван', то 'он'). "
                                "'unknown', если по контексту не понятно, кто говорит.",
            },
        },
        "required": ["results"],
    },
}


def _attribution_dialogue_count(text: str) -> int:
    paras = _split_dialogue_paragraphs(text)
    return sum(1 for _p, is_d in paras if is_d)


def _attribution_prompt(text: str, dialogue_count: int) -> str:
    return (
        "Ниже — глава книги на русском языке. Определи, кто произносит каждую реплику "
        "прямой речи (абзацы, начинающиеся с тире «—», «-» или «–»).\n\n"
        "Верни ответ вызовом инструмента report_speakers с полем results — списком имён "
        "говорящих строго в том порядке, в котором реплики встречаются в тексте (первая "
        "реплика — первый элемент списка). Количество элементов должно ТОЧНО совпадать с "
        f"количеством реплик в тексте ({dialogue_count} шт.) — не пропускай и не добавляй лишних.\n\n"
        "Для одного и того же персонажа всегда используй одно и то же каноничное имя "
        "(например, всегда 'Иван Петров', а не то 'Иван', то 'он', то 'Петров') — это "
        "нужно, чтобы закрепить за персонажем один голос на всю книгу. Если по контексту "
        "невозможно понять, кто говорит — используй значение 'unknown'.\n\n"
        "--- ТЕКСТ ГЛАВЫ ---\n" + text
    )


def _normalize_attribution_results(results, dialogue_count: int, log_fn=None) -> list:
    results = [str(r).strip() or "unknown" for r in (results or [])]
    if len(results) != dialogue_count:
        if log_fn:
            log_fn(f"Внимание: модель вернула {len(results)} имён вместо {dialogue_count} "
                   "реплик — выравниваю список (лишнее обрезаю/недостающее заполняю 'unknown').")
        if len(results) < dialogue_count:
            results = results + ["unknown"] * (dialogue_count - len(results))
        else:
            results = results[:dialogue_count]
    return results


def attribute_speakers_anthropic(text: str, api_key: str, model: str, log_fn=None) -> list:
    """Спрашивает у Anthropic Claude API, кто произносит каждую реплику
    прямой речи в тексте главы. Возвращает список имён — по одному на
    каждую реплику, в порядке их появления в тексте.

    Требует отдельный платный API-ключ Anthropic (console.anthropic.com) —
    это не тот же ключ, что использует сама программа Claude/Cowork, и не
    связан с Yandex SpeechKit."""
    import requests
    import json as _json

    dialogue_count = _attribution_dialogue_count(text)
    if dialogue_count == 0:
        return []

    if not api_key:
        raise ValueError(
            "Не указан API-ключ Anthropic для атрибуции говорящих. Получить его можно на "
            "https://console.anthropic.com/settings/keys — это отдельный (платный) ключ, не "
            "связанный с самой программой Claude."
        )

    prompt = _attribution_prompt(text, dialogue_count)
    if log_fn:
        log_fn(f"Атрибуция говорящих через Anthropic {model}: {dialogue_count} реплик, "
               f"{len(text)} симв. текста главы...")

    resp = requests.post(
        ANTHROPIC_MESSAGES_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "tools": [_ATTRIBUTION_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "report_speakers"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        body = resp.text[:1000]
        if log_fn:
            log_fn(f"Anthropic API ответил HTTP {resp.status_code}: {body}")
        raise RuntimeError(
            f"Anthropic API вернул ошибку HTTP {resp.status_code}: {body[:500]}\n"
            "  Проверьте API-ключ и баланс на https://console.anthropic.com/ . "
            f"Если ошибка про модель {model!r} — актуальные названия моделей "
            "смотрите на https://docs.claude.com/en/docs/about-claude/models и "
            "укажите вручную в настройках атрибуции."
        )

    data = resp.json()
    results = None
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "report_speakers":
            results = (block.get("input") or {}).get("results")
            break

    if not isinstance(results, list):
        raise RuntimeError(
            f"Anthropic API не вернул ожидаемый список говорящих (ответ: {_json.dumps(data)[:500]})"
        )

    return _normalize_attribution_results(results, dialogue_count, log_fn)


def attribute_speakers_gemini(text: str, api_key: str, model: str, log_fn=None) -> list:
    """То же самое, но через бесплатный уровень Google Gemini API
    (aistudio.google.com/apikey — ключ бесплатный, карта не нужна).
    Использует function calling (аналог tool_use у Anthropic) с той же
    схемой report_speakers, чтобы результат был структурированным JSON, а
    не текстом, который надо парсить руками."""
    import requests
    import json as _json

    dialogue_count = _attribution_dialogue_count(text)
    if dialogue_count == 0:
        return []

    if not api_key:
        raise ValueError(
            "Не указан API-ключ Google Gemini для атрибуции говорящих. Получить бесплатный "
            "ключ можно на https://aistudio.google.com/apikey — карта не требуется."
        )

    prompt = _attribution_prompt(text, dialogue_count)
    if log_fn:
        log_fn(f"Атрибуция говорящих через Gemini {model}: {dialogue_count} реплик, "
               f"{len(text)} симв. текста главы...")

    gemini_schema = {
        "name": "report_speakers",
        "description": _ATTRIBUTION_TOOL_SCHEMA["description"],
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": _ATTRIBUTION_TOOL_SCHEMA["input_schema"]["properties"]["results"]["description"],
                },
            },
            "required": ["results"],
        },
    }

    resp = requests.post(
        GEMINI_GENERATE_URL_TMPL.format(model=model),
        params={"key": api_key},
        headers={"content-type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"function_declarations": [gemini_schema]}],
            "tool_config": {"function_calling_config": {"mode": "ANY",
                                                          "allowed_function_names": ["report_speakers"]}},
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        body = resp.text[:1000]
        if log_fn:
            log_fn(f"Gemini API ответил HTTP {resp.status_code}: {body}")
        raise RuntimeError(
            f"Gemini API вернул ошибку HTTP {resp.status_code}: {body[:500]}\n"
            "  Проверьте API-ключ на https://aistudio.google.com/apikey (и не превышена ли "
            "дневная бесплатная квота — она сбрасывается раз в сутки). "
            f"Если ошибка про модель {model!r} — актуальные названия моделей "
            "смотрите на https://ai.google.dev/gemini-api/docs/models и "
            "укажите вручную в настройках атрибуции."
        )

    data = resp.json()
    results = None
    for cand in data.get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            fc = part.get("functionCall")
            if fc and fc.get("name") == "report_speakers":
                results = (fc.get("args") or {}).get("results")
                break
        if results is not None:
            break

    if not isinstance(results, list):
        raise RuntimeError(
            f"Gemini API не вернул ожидаемый список говорящих (ответ: {_json.dumps(data)[:500]})"
        )

    return _normalize_attribution_results(results, dialogue_count, log_fn)


def _extract_json_array(raw_text: str):
    """Достаёт из текстового ответа модели JSON-массив строк, даже если
    модель обернула его в markdown-код (```json ... ```) или добавила
    пояснения до/после. Нужен для YandexGPT, у которого нет отдельного
    режима принудительного вызова инструмента (в отличие от Anthropic/
    Gemini) — там приходится просить вернуть JSON текстом и парсить его."""
    import json as _json
    import re as _re

    m = _re.search(r"\[.*\]", raw_text, flags=_re.DOTALL)
    if not m:
        return None
    try:
        parsed = _json.loads(m.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def attribute_speakers_yandexgpt(text: str, api_key: str, model: str, folder_id: str = "",
                                  log_fn=None) -> list:
    """То же самое через YandexGPT (Yandex Cloud AI Studio) — работает из
    России без VPN, есть бесплатный грант на запросы; можно использовать
    тот же API-ключ и Folder ID, что и для Yandex SpeechKit. У YandexGPT
    нет строгого function calling, поэтому просим вернуть JSON текстом и
    парсим его сами (см. _extract_json_array)."""
    import requests
    import json as _json

    dialogue_count = _attribution_dialogue_count(text)
    if dialogue_count == 0:
        return []

    if not api_key:
        raise ValueError(
            "Не указан API-ключ YandexGPT для атрибуции говорящих. Можно использовать тот же "
            "ключ, что и для Yandex SpeechKit выше, либо получить отдельный на "
            "https://yandex.cloud/ru/docs/ai-studio/quickstart"
        )
    if not folder_id:
        raise ValueError(
            "Не указан Folder ID для YandexGPT — можно использовать тот же Folder ID, что и "
            "для Yandex SpeechKit выше."
        )

    prompt = (
        _attribution_prompt(text, dialogue_count) +
        "\n\nВерни ОТВЕТ СТРОГО в виде JSON-массива строк, без каких-либо пояснений, "
        "markdown-разметки или текста до/после — только сам массив, например: "
        '["Иван Петров", "unknown", "Мария"]'
    )
    if log_fn:
        log_fn(f"Атрибуция говорящих через YandexGPT {model}: {dialogue_count} реплик, "
               f"{len(text)} симв. текста главы...")

    model_uri = f"gpt://{folder_id}/{model}"
    resp = requests.post(
        YANDEXGPT_COMPLETION_URL,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "content-type": "application/json",
        },
        json={
            "modelUri": model_uri,
            "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": "4000"},
            "messages": [{"role": "user", "text": prompt}],
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        body = resp.text[:1000]
        if log_fn:
            log_fn(f"YandexGPT API ответил HTTP {resp.status_code}: {body}")
        raise RuntimeError(
            f"YandexGPT API вернул ошибку HTTP {resp.status_code}: {body[:500]}\n"
            "  Проверьте API-ключ и Folder ID (тот же, что и для SpeechKit, либо отдельный "
            "с ролью ai.languageModels.user) на https://console.yandex.cloud/ . "
            f"Если ошибка про модель {model!r} — актуальные названия моделей смотрите на "
            "https://yandex.cloud/ru/docs/ai-studio/concepts/generation/models"
        )

    data = resp.json()
    try:
        raw_text = data["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"YandexGPT API не вернул ожидаемый текст ответа (ответ: {_json.dumps(data)[:500]})"
        )

    results = _extract_json_array(raw_text)
    if results is None:
        raise RuntimeError(
            f"YandexGPT вернул ответ, из которого не удалось извлечь список говорящих "
            f"(ответ модели: {raw_text[:500]})"
        )

    return _normalize_attribution_results(results, dialogue_count, log_fn)


def attribute_speakers_llm(text: str, api_key: str, model: str = DEFAULT_ATTRIBUTION_MODEL,
                            provider: str = DEFAULT_ATTRIBUTION_PROVIDER, log_fn=None,
                            folder_id: str = "") -> list:
    """Диспетчер: вызывает attribute_speakers_yandexgpt (по умолчанию,
    работает из РФ без VPN), attribute_speakers_gemini (бесплатно, но
    часто недоступен из РФ) или attribute_speakers_anthropic (платно), в
    зависимости от provider — см. ATTRIBUTION_PROVIDERS."""
    if provider == "yandexgpt":
        return attribute_speakers_yandexgpt(text, api_key, model, folder_id=folder_id, log_fn=log_fn)
    if provider == "anthropic":
        return attribute_speakers_anthropic(text, api_key, model, log_fn=log_fn)
    return attribute_speakers_gemini(text, api_key, model, log_fn=log_fn)


def _make_file_logger(outdir: Path, filename: str):
    """Возвращает log(message) — пишет и в консоль/GUI (через print,
    перехватываемый TextRedirector), и в файл рядом с аудио, чтобы можно
    было потом посмотреть подробности (например, ответы LLM-атрибуции)."""
    import time as _time
    log_path = outdir / filename

    def log(message: str):
        line = f"{_time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(f"  {message}")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    return log


def _character_map_path(outdir: Path) -> Path:
    return outdir / "dialogue_characters.json"


def _load_character_voice_map(outdir: Path) -> dict:
    """Словарь {имя_персонажа: голос}, сохраняемый рядом с аудио — так
    один и тот же персонаж получает один и тот же голос во всех главах
    книги, а не только внутри одной главы. Можно открыть и поправить
    руками между запусками (например, если атрибуция ошиблась с полом
    голоса для персонажа)."""
    path = _character_map_path(outdir)
    if not path.exists():
        return {}
    try:
        import json as _json
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_character_voice_map(outdir: Path, mapping: dict):
    path = _character_map_path(outdir)
    try:
        import json as _json
        path.write_text(_json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _group_paragraphs_by_speaker_names(text: str, main_voice, dialogue_voices,
                                        speaker_names: list, character_voice_map: dict):
    """Как _group_paragraphs_by_voice, но голос реплики выбирается не по
    круговому чередованию, а по имени говорящего (speaker_names[i] — имя
    для i-й по счёту реплики в тексте, см. attribute_speakers_llm):
    каждому новому имени закрепляется следующий свободный голос из
    dialogue_voices (и запоминается в character_voice_map — на будущие
    главы и перезапуски), 'unknown'/непонятные реплики просто чередуются
    между собой отдельным счётчиком, чтобы не путать с реальными именами."""
    paras = _split_dialogue_paragraphs(text)
    grouped = []  # [[voice, [paragraphs], is_dialogue], ...]
    di = 0
    unknown_i = 0
    for para, is_dialogue in paras:
        if is_dialogue:
            name = speaker_names[di] if di < len(speaker_names) else "unknown"
            di += 1
            name_key = name.strip().lower()
            if not name_key or name_key == "unknown":
                voice = dialogue_voices[unknown_i % len(dialogue_voices)]
                unknown_i += 1
            elif name_key in character_voice_map:
                voice = character_voice_map[name_key]
            else:
                voice = dialogue_voices[len(character_voice_map) % len(dialogue_voices)]
                character_voice_map[name_key] = voice
            grouped.append([voice, [para], True])
        else:
            if grouped and grouped[-1][0] == main_voice and not grouped[-1][2]:
                grouped[-1][1].append(para)
            else:
                grouped.append([main_voice, [para], False])

    return [(voice, "\n".join(para_list)) for voice, para_list, _is_dialogue in grouped]


def resolve_voice_groups(text: str, main_voice, dialogue_voices, outdir: Path,
                          attribution=None, log_fn=None):
    """Единая точка входа для всех run_*: строит список (голос, текст
    группы) для главы. Без dialogue_voices — весь текст одним main_voice.
    С dialogue_voices, но без attribution — простое чередование реплик по
    кругу (см. _group_paragraphs_by_voice), как раньше. С attribution —
    {"api_key":..., "model":...} — реплики атрибутируются через Claude API
    (attribute_speakers_llm), и один и тот же персонаж получает один и тот
    же голос по всей книге (словарь сохраняется в outdir/dialogue_characters.json).
    При любой ошибке атрибуции (нет ключа, сеть, лимиты) — тихо откатывается
    на простое чередование, чтобы не срывать всю озвучку из-за LLM."""
    if not dialogue_voices:
        return [(main_voice, text)]
    if not attribution:
        return _group_paragraphs_by_voice(text, main_voice, dialogue_voices)

    try:
        speaker_names = attribute_speakers_llm(
            text, attribution.get("api_key", ""), attribution.get("model", DEFAULT_ATTRIBUTION_MODEL),
            provider=attribution.get("provider", DEFAULT_ATTRIBUTION_PROVIDER),
            folder_id=attribution.get("folder_id", ""),
            log_fn=log_fn,
        )
    except Exception as e:
        if log_fn:
            log_fn(f"Атрибуция говорящих не удалась ({e}) — использую обычное чередование голосов.")
        return _group_paragraphs_by_voice(text, main_voice, dialogue_voices)

    character_voice_map = _load_character_voice_map(outdir)
    groups = _group_paragraphs_by_speaker_names(
        text, main_voice, dialogue_voices, speaker_names, character_voice_map
    )
    _save_character_voice_map(outdir, character_voice_map)
    if log_fn:
        named = ", ".join(sorted(character_voice_map.keys()))
        log_fn(f"Персонажи с закреплённым голосом на сейчас: {named or '(пока нет)'}")
    return groups


def play_file(path: Path):
    """Проигрывает готовый аудиофайл.

    Раньше для этого всегда требовался pygame — но на новых версиях
    Python (например 3.14) у pygame ещё может не быть готового wheel под
    Windows, и pip пытается собрать его из исходников, что почти всегда
    падает без установленного компилятора Visual Studio. Поэтому сначала
    используются средства, которые есть в системе без установки чего бы
    то ни было: winsound для .wav на Windows (входит в стандартную
    библиотеку Python), иначе — открытие файла проигрывателем по
    умолчанию (os.startfile на Windows, afplay на macOS, xdg-open на
    Linux). pygame используется только как запасной вариант, если он
    всё-таки установлен, а системные способы не сработали.
    """
    if sys.platform == "win32":
        if path.suffix.lower() == ".wav":
            try:
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return
            except Exception:
                pass
        try:
            os.startfile(str(path))  # noqa: S606 — открывает системный проигрыватель по умолчанию
            return
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess
            subprocess.run(["afplay", str(path)], check=True)
            return
        except Exception:
            pass
    else:
        try:
            import subprocess
            subprocess.run(["xdg-open", str(path)], check=True)
            return
        except Exception:
            pass

    # запасной вариант, если ни один системный способ не сработал
    try:
        import pygame
    except ImportError:
        raise RuntimeError(
            "Не удалось проиграть файл: не нашёл ни системного проигрывателя, "
            f"ни pygame. Откройте файл вручную: {path}"
        )
    pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def _parse_chapters_spec(spec: str):
    """Разбирает строку вида "1,3,5-7" в список номеров глав (1-based).
    Пустая строка -> None (значит, использовать --start как раньше)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if a > b:
                a, b = b, a
            result.update(range(a, b + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _select_chapters(chapters, start: int = 1, chapter_indices=None, char_ranges=None):
    """Отбирает главы для озвучки — либо все, начиная с --start (старое
    поведение по умолчанию), либо только перечисленные в chapter_indices
    (1-based номера глав, в любом порядке — результат всё равно идёт по
    порядку книги; start в этом случае игнорируется). Это то, что стоит
    за выбором конкретных глав в GUI (можно выделить несколько
    несоседних глав вместо диапазона).

    char_ranges — необязательный словарь {номер_главы: (от_символа,
    до_символа)}, чтобы озвучить не главу целиком, а кусок её текста
    (например, только середину — если в остальном всё уже устраивает).
    Название такой главы дополняется пометкой "(фрагмент)", чтобы файл
    не путался с озвучкой главы целиком и не перезаписывал её.

    Возвращает список (номер_главы, заголовок, текст) в порядке книги —
    именно по нему потом идёт прогресс-бар (его длина — это "всего глав"
    для текущего запуска)."""
    if chapter_indices:
        wanted = sorted({i for i in chapter_indices if 1 <= i <= len(chapters)})
    else:
        wanted = [i for i in range(1, len(chapters) + 1) if i >= start]

    result = []
    for idx in wanted:
        title, text = chapters[idx - 1]
        if char_ranges and idx in char_ranges:
            a, b = char_ranges[idx]
            a = max(0, min(a, len(text)))
            b = max(a, min(b, len(text)))
            if a > 0 or b < len(text):
                text = text[a:b]
                title = f"{title} (фрагмент)"
        result.append((idx, title, text))
    return result


YANDEX_TTS_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
YANDEX_MAX_CHARS = 4900  # лимит SpeechKit — 5000 символов на запрос, берём с запасом


def synth_yandex_chunk(text: str, api_key: str, folder_id: str, voice: str, lang: str,
                        speed: float, emotion: str, audio_format: str, log_fn=None) -> bytes:
    """Один запрос к Yandex SpeechKit (REST), возвращает содержимое аудио
    (mp3 по умолчанию). Поднимает исключение с понятным текстом при ошибке
    (неверный ключ, кончились деньги/лимит и т.п.).

    Folder ID необязателен: если API-ключ выпущен для сервисного аккаунта
    (обычный случай для этого скрипта — см. README), SpeechKit сам знает
    каталог, в котором создан аккаунт, и folderId можно не передавать.
    Указывать его нужно только для другого, более редкого способа
    авторизации (IAM-токен пользовательского аккаунта), который этот
    скрипт не использует.

    log_fn(message), если передан, получает подробности каждого запроса
    (все параметры, кроме самого ключа — он в логе не пишется) и полный
    ответ сервера при ошибке — используется, чтобы разобраться в причине
    HTTP 400/403 и т.п., когда простого сообщения об ошибке недостаточно."""
    import urllib.request
    import urllib.parse
    import urllib.error

    if not api_key:
        raise ValueError(
            "Не указан API-ключ Yandex SpeechKit. Получить его можно в консоли "
            "Yandex Cloud (см. README, раздел про Yandex SpeechKit)."
        )

    params = {
        "text": text,
        "lang": lang,
        "voice": voice,
        "format": audio_format,
        "speed": str(speed),
    }
    if folder_id:
        params["folderId"] = folder_id
    if emotion:
        params["emotion"] = emotion

    if log_fn:
        # По просьбе пользователя пишем полный запрос целиком, включая
        # API-ключ, — для диагностики. ВНИМАНИЕ: значит, файл лога
        # (yandex_client.log) содержит секретный ключ в открытом виде —
        # не отправляйте его никому и не коммитьте в git (он уже добавлен
        # в .gitignore рядом с fb2_reader_settings.json).
        full_params = dict(params)
        full_params["text"] = f"[{len(text)} симв.] {text[:80]!r}…" if len(text) > 80 else text
        log_fn(f"Запрос к Yandex SpeechKit: {full_params} "
               f"(Authorization: Api-Key {api_key})")

    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        YANDEX_TTS_URL,
        data=data,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    import http.client
    import socket
    import time as _time

    retries = 3
    last_incomplete = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (http.client.IncompleteRead, socket.timeout, ConnectionError) as e:
            # Соединение оборвалось/протухло не долетев до конца ответа
            # (обрыв сети, антивирус/прокси режет длинные ответы,
            # Dropbox/файрвол вмешивается и т.п.) — сам сервис Yandex тут
            # обычно ни при чём, это НЕ ошибка синтеза. Пробуем ещё раз с
            # той же главой.
            last_incomplete = e
            if isinstance(e, http.client.IncompleteRead):
                got = len(e.partial) if e.partial else 0
                detail = f"получено {got} байт из {e.expected or '?'}"
            else:
                detail = f"{type(e).__name__}: {e}"
            if log_fn:
                log_fn(f"Обрыв соединения при получении ответа ({detail}), "
                       f"попытка {attempt}/{retries}...")
            if attempt < retries:
                _time.sleep(3.0)
                continue
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if log_fn:
                log_fn(f"Yandex SpeechKit ответил HTTP {e.code}, полный текст ответа: {body}")
            raise RuntimeError(
                f"Yandex SpeechKit вернул ошибку HTTP {e.code}: {body[:500]}\n"
                "  Частые причины: неверный/просроченный API-ключ, не подключён "
                "платёжный аккаунт, закончилась пробная квота, либо у сервисного "
                "аккаунта нет роли ai.speechkit-tts.user. Проверить ключ, баланс и "
                "роли можно в консоли Yandex Cloud: https://console.yandex.cloud/ "
                "(раздел Billing — баланс/лимиты, IAM — роли сервисного аккаунта). "
                "Список кодов ошибок SpeechKit: "
                "https://yandex.cloud/ru/docs/speechkit/tts/request\n"
                f"  Полный текст ответа записан в лог рядом с аудио (yandex_client.log)."
            ) from e
        except urllib.error.URLError as e:
            if log_fn:
                log_fn(f"Не удалось связаться с Yandex SpeechKit: {e.reason}")
            raise RuntimeError(f"Не удалось связаться с Yandex SpeechKit: {e.reason}") from e

    # Все retries исчерпаны, каждый раз обрыв на IncompleteRead
    raise RuntimeError(
        f"Не удалось получить полный ответ от Yandex SpeechKit после {retries} попыток "
        f"(соединение обрывается на середине ответа: {last_incomplete}). Обычно это "
        "сеть/антивирус/прокси, а не сам SpeechKit — проверьте интернет-соединение, "
        "или, если антивирус агрессивно проверяет трафик, добавьте исключение для "
        "python.exe/этой программы."
    ) from last_incomplete


def run_yandex(chapters, outdir: Path, start: int, play: bool, api_key: str, folder_id: str,
               voice: str = "alena", lang: str = "", speed: float = 1.0,
               emotion: str = "", on_progress=None, chapter_indices=None, char_ranges=None,
               dialogue_voices=None, attribution=None, should_stop=None, play_fn=None):
    """Озвучка через облачный Yandex SpeechKit. Требует интернет, ключ и
    Folder ID на каждый запуск, платный после пробного периода (см.
    README). Текст режется на куски по YANDEX_MAX_CHARS (лимит SpeechKit —
    5000 символов на запрос) и склеивается через pydub, если он
    установлен, иначе сохраняется частями.

    chapter_indices/char_ranges — см. _select_chapters: позволяют
    озвучить только выбранные главы (не обязательно подряд) и/или кусок
    конкретной главы вместо неё целиком.

    dialogue_voices — необязательный список голосов (ключи YANDEX_VOICES),
    которыми по очереди озвучиваются реплики прямой речи (абзацы,
    начинающиеся с тире), чтобы диалоги не звучали одним и тем же
    монотонным голосом на протяжении всей книги — см. _build_voiced_segments.
    Остальной текст (авторская речь) по-прежнему звучит голосом voice."""
    import time as _time

    # lang должен точно соответствовать голосу (см. YANDEX_VOICE_LANGS) —
    # если передан явный lang, отличающийся от нужного для voice, это,
    # скорее всего, ошибка настройки (например, украинский голос lera с
    # lang=ru-RU), которая иначе приводит к неочевидной HTTP 400 —
    # используем правильный lang для голоса всегда, чтобы не наступать на
    # эти грабли.
    correct_lang = yandex_lang_for_voice(voice)
    if lang and lang != correct_lang:
        print(f"Внимание: голосу {voice!r} нужен lang={correct_lang!r}, "
              f"а не {lang!r} — использую {correct_lang!r}, иначе Yandex "
              f"ответит ошибкой HTTP 400 (несовпадение voice/lang).")
    lang = correct_lang

    outdir.mkdir(parents=True, exist_ok=True)
    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1
    audio_format = "mp3"

    log_path = outdir / "yandex_client.log"

    def log(message: str):
        line = f"{_time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(f"  {message}")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.mp3"
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="yandex", voice=voice, lang=lang, speed=speed,
            emotion=emotion, format=audio_format,
            dialogue_voices=",".join(dialogue_voices) if dialogue_voices else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (Yandex SpeechKit): {title} -> {fname}")

        voice_groups = resolve_voice_groups(text, voice, dialogue_voices, outdir,
                                             attribution=attribution, log_fn=log)
        segments = _chunk_voice_groups(voice_groups, YANDEX_MAX_CHARS)
        chunks_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, chunks_total)

        chunk_bytes = []
        for chunk_no, (chunk, seg_voice) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if chunk.strip() and _text_has_speakable_content(chunk):
                seg_lang = yandex_lang_for_voice(seg_voice)
                data = synth_yandex_chunk(
                    chunk, api_key, folder_id, seg_voice, seg_lang, speed, emotion, audio_format,
                    log_fn=log,
                )
                chunk_bytes.append(data)
            if on_progress:
                on_progress(pos, total, chunk_no, chunks_total)

        if not chunk_bytes:
            continue

        if len(chunk_bytes) == 1:
            out_path.write_bytes(chunk_bytes[0])
        else:
            try:
                from pydub import AudioSegment
                import io as _io
                combined = AudioSegment.empty()
                for data in chunk_bytes:
                    combined += AudioSegment.from_file(_io.BytesIO(data), format="mp3")
                combined.export(str(out_path), format="mp3")
            except ImportError:
                for i, data in enumerate(chunk_bytes):
                    part_path = out_path.with_name(f"{out_path.stem}_part{i + 1}{out_path.suffix}")
                    part_path.write_bytes(data)
                print("  (pydub не установлен — глава сохранена частями *_partN.mp3)")

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


def run_online(chapters, outdir: Path, play: bool, start: int, voice_lang: str, on_progress=None,
               chapter_indices=None, char_ranges=None, dialogue_voices=None, should_stop=None,
               play_fn=None):
    """voice_lang — язык для gTTS (Google Translate TTS). dialogue_voices
    здесь принимается только для единообразия сигнатуры с другими
    режимами — у gTTS нет отдельных русских голосов (только один голос на
    язык), поэтому реально разные голоса для диалогов он дать не может;
    если параметр передан непустым, просто печатается предупреждение."""
    if dialogue_voices:
        print("Внимание: Google TTS (режим online) не поддерживает несколько "
              "разных русских голосов — вся книга озвучится одним голосом, "
              "как обычно. Для разных голосов на диалогах используйте режим "
              "Silero (локально, бесплатно) или Yandex SpeechKit.")
    outdir.mkdir(parents=True, exist_ok=True)
    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1
    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.mp3"
        out_path = outdir / fname
        fingerprint = _params_fingerprint(text, mode="online", lang=voice_lang)
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue
        print(f"[{idx}/{len(chapters)}] Озвучиваю: {title} -> {fname}")
        if on_progress:
            on_progress(pos, total, 0, 1)
        synth_online(text, out_path, lang=voice_lang, desc=f"Гл.{idx}")
        _save_fingerprint(out_path, fingerprint)
        if on_progress:
            on_progress(pos, total, 1, 1)
        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)


_LOCAL_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_LOCAL_COMMA_SPLIT_RE = re.compile(r"(?<=[,;:—-])\s+")


def _segments_with_pauses(text: str, sentence_break_ms: int, paragraph_break_ms: int,
                           comma_break_ms: int, max_chars: int = 350):
    """Делит текст главы на короткие озвучиваемые куски по знакам
    препинания (абзац -> предложение -> часть предложения по запятым/тире),
    и для каждого куска сразу вычисляет длину паузы (мс), которая должна
    идти ПОСЛЕ него — короткая на запятой/тире/двоеточии, побольше на
    границе предложений, самая длинная между абзацами. Так локальный режим
    silero тоже получает интонационные паузы по пунктуации, а не только
    silero_rest. Длинные куски дополнительно режутся по max_chars (это
    ограничение самого Silero на длину текста за один вызов), с небольшой
    паузой между такими техническими обрезками.

    Возвращает список (текст_куска, пауза_после_мс).
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()] or [text.strip()]
    segments: list[tuple[str, int]] = []

    for pi, para in enumerate(paragraphs):
        is_last_para = pi == len(paragraphs) - 1
        sentences = [s.strip() for s in _LOCAL_SENT_SPLIT_RE.split(para) if s.strip()] or [para]
        for si, sent in enumerate(sentences):
            is_last_sentence = si == len(sentences) - 1
            comma_parts = [c.strip() for c in _LOCAL_COMMA_SPLIT_RE.split(sent) if c.strip()] or [sent]
            for ci, part in enumerate(comma_parts):
                is_last_comma = ci == len(comma_parts) - 1
                if not is_last_comma:
                    pause = comma_break_ms
                elif not is_last_sentence:
                    pause = sentence_break_ms
                elif not is_last_para:
                    pause = paragraph_break_ms
                else:
                    pause = 0

                if len(part) <= max_chars:
                    segments.append((part, pause))
                else:
                    # слишком длинный кусок для одного вызова Silero — режем
                    # дальше по предложениям/словам, между техническими
                    # обрезками ставим небольшую паузу (comma_break_ms)
                    sub_chunks = split_text(part, max_chars)
                    for j, sub in enumerate(sub_chunks):
                        sub_pause = pause if j == len(sub_chunks) - 1 else min(comma_break_ms, 120)
                        segments.append((sub, sub_pause))

    return segments


def run_silero(chapters, outdir: Path, start: int, speaker: str, sample_rate: int, play: bool,
               model_id: str = DEFAULT_SILERO_MODEL, sentence_break_ms: int = 320,
               paragraph_break_ms: int = 550, comma_break_ms: int = 180,
               put_accent: bool = True, put_yo: bool = True, on_progress=None,
               chapter_indices=None, char_ranges=None, dialogue_speakers=None, attribution=None,
               should_stop=None, play_fn=None):
    """Озвучка через Silero TTS — нейросетевой русский голос, локально.
    По умолчанию v5_5_ru (последняя модель: ударения, омографы, вопросы).

    Текст делится на короткие куски по знакам препинания (см.
    _segments_with_pauses), между кусками вставляется пауза нужной длины —
    короткая на запятых/тире, побольше между предложениями, самая длинная
    между абзацами. Ударения (put_accent) и буква "ё" (put_yo) расставляются
    моделью автоматически.

    Если синтез фрагмента не удаётся — не пропускается молча: сначала
    делается повторная попытка, затем (если фрагмент состоит из нескольких
    слов) он делится пополам и пробуется по частям, и только если ничего не
    помогло — на его место вставляется короткая тишина (чтобы не потерять
    место в тексте и не сломать порядок остальных фрагментов), с явным
    сообщением в лог. Раньше такие фрагменты просто выбрасывались, из-за
    чего в готовой озвучке появлялись заметные пропуски.

    on_progress(chapter_pos, chapters_total, fragment_done, fragments_total),
    если передан, вызывается после каждого фрагмента (и один раз в начале
    главы) — используется GUI для настоящего прогресс-бара вместо
    "бегающей" неопределённой полоски.

    dialogue_speakers — необязательный список голосов Silero (из
    speakers_for_model(model_id)), которыми по очереди озвучиваются реплики
    прямой речи (абзацы, начинающиеся с тире) — см. _group_paragraphs_by_voice.
    Остальной текст (авторская речь) звучит голосом speaker, как обычно.
    """
    import numpy as np
    import wave

    allowed = speakers_for_model(model_id)
    if speaker not in allowed:
        print(f"Голос {speaker!r} недоступен для {model_id}, использую xenia")
        speaker = "xenia" if "xenia" in allowed else allowed[0]
    if dialogue_speakers:
        bad = [s for s in dialogue_speakers if s not in allowed]
        dialogue_speakers = [s for s in dialogue_speakers if s in allowed]
        if bad:
            print(f"Голоса {bad} недоступны для {model_id}, пропускаю их в чередовании диалогов.")
        if not dialogue_speakers:
            print("Не осталось ни одного голоса для диалогов после проверки — "
                  "диалоги озвучиваются основным голосом, как обычно.")

    print(f"Загружаю модель Silero TTS {model_id} (при первом запуске — скачивание)...")
    model = load_silero_model(model_id)

    outdir.mkdir(parents=True, exist_ok=True)

    max_chars = 350

    def synth_one(part_text: str, part_speaker: str):
        """Один вызов модели. Бросает исключение при неудаче."""
        # Локальный Silero (в отличие от silero_rest) не умеет сам
        # разворачивать числа в слова — без этого цифры либо пропускаются,
        # либо звучат странно.
        part_text = numbers_to_words_ru(part_text)
        return model.apply_tts(
            text=part_text,
            speaker=part_speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo,
        ).numpy()

    def synth_with_fallback(part_text: str, idx: int, title: str, part_speaker: str) -> "np.ndarray":
        """Синтезирует один фрагмент с повтором и делением пополам при
        ошибке; в самом крайнем случае возвращает тишину вместо исключения."""
        if not _text_has_speakable_content(part_text):
            # разделитель сцен вроде "* * *" — нечего озвучивать, просто
            # короткая пауза вместо него, без обращения к TTS
            return np.zeros(int(sample_rate * 0.4), dtype=np.float32)
        try:
            return synth_one(part_text, part_speaker)
        except Exception as e1:
            print(f"  [Гл.{idx} «{title}»] ошибка синтеза фрагмента ({e1}), повторяю…")
            try:
                return synth_one(part_text, part_speaker)
            except Exception as e2:
                words = part_text.split()
                if len(words) > 3:
                    mid = len(words) // 2
                    left, right = " ".join(words[:mid]), " ".join(words[mid:])
                    print(f"  [Гл.{idx} «{title}»] повтор не помог ({e2}), делю фрагмент пополам и пробую снова…")
                    try:
                        left_audio = synth_with_fallback(left, idx, title, part_speaker)
                        right_audio = synth_with_fallback(right, idx, title, part_speaker)
                        gap = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
                        return np.concatenate([left_audio, gap, right_audio])
                    except Exception:
                        pass
                silence_seconds = max(0.4, min(6.0, len(part_text) / 15))
                print(f"  [Гл.{idx} «{title}»] ОШИБКА: не удалось синтезировать фрагмент "
                      f"({e2}). Вставляю тишину ({silence_seconds:.1f} с) вместо него: "
                      f"{part_text[:80]!r}…")
                return np.zeros(int(sample_rate * silence_seconds), dtype=np.float32)

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="silero", model=model_id, speaker=speaker,
            sample_rate=sample_rate, max_chars=max_chars,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms, put_accent=put_accent, put_yo=put_yo,
            dialogue_speakers=",".join(dialogue_speakers) if dialogue_speakers else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено с теми же параметрами): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю: {title} -> {fname}")

        # Группируем по голосу (диалоги/повествование, см. dialogue_speakers,
        # и/или LLM-атрибуция по персонажам, см. attribution), затем каждую
        # группу — как раньше, на куски по знакам препинания с паузами.
        # Результат — плоский список (текст, пауза, голос).
        voice_groups = resolve_voice_groups(
            text, speaker, dialogue_speakers, outdir, attribution=attribution,
            log_fn=_make_file_logger(outdir, "silero_client.log"),
        )
        segments = []
        for group_voice, group_text in voice_groups:
            for part_text, pause_ms in _segments_with_pauses(
                group_text, sentence_break_ms=sentence_break_ms,
                paragraph_break_ms=paragraph_break_ms, comma_break_ms=comma_break_ms,
                max_chars=max_chars,
            ):
                segments.append((part_text, pause_ms, group_voice))

        audio_parts = []
        segs_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, segs_total)

        for seg_i, (part_text, pause_ms, seg_speaker) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if not part_text.strip():
                if on_progress:
                    on_progress(pos, total, seg_i, segs_total)
                continue
            audio = synth_with_fallback(part_text, idx, title, seg_speaker)
            audio_parts.append(audio)
            if pause_ms > 0:
                audio_parts.append(np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32))
            if on_progress:
                on_progress(pos, total, seg_i, segs_total)

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


# --------------------------------------------------------------------------
# SSML: расстановка интонационных пауз, вопросов/восклицаний
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SOFT_PAUSE_RE = re.compile(r"([,;:]|—|--)\s+")


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _prosody_wrap(escaped_sentence: str, original_sentence: str, emphasize: bool) -> str:
    """Слегка усиливает интонацию вопросительных/восклицательных предложений
    через <prosody pitch=.../rate=...> поверх того, что модель и так делает
    по самому знаку "?"/"!" — особенно заметно на репликах в кавычках и
    коротких восклицаниях."""
    if not emphasize:
        return f"<s>{escaped_sentence}</s>"
    stripped = original_sentence.rstrip()
    if stripped.endswith("?"):
        return f'<s><prosody pitch="high">{escaped_sentence}</prosody></s>'
    if stripped.endswith("!"):
        return f'<s><prosody pitch="high" rate="fast">{escaped_sentence}</prosody></s>'
    return f"<s>{escaped_sentence}</s>"


def text_to_ssml(text: str, sentence_break_ms: int = 320,
                  paragraph_break_ms: int = 550, comma_break_ms: int = 180,
                  emphasize: bool = True) -> str:
    """Превращает обычный текст главы в SSML-документ для Silero.

    * абзацы -> <p>, между ними длинная пауза (paragraph_break_ms);
    * предложения -> <s>, между ними пауза покороче (sentence_break_ms);
      знаки "?" и "!" сохраняются как есть — по ним Silero строит
      вопросительную/восклицательную интонацию, а если emphasize=True —
      дополнительно оборачиваются в <prosody pitch="high"[...]"> для более
      выраженной интонации;
    * внутри предложения после запятых/тире/двоеточий/точек с запятой
      добавляется короткая пауза <break/> (comma_break_ms), имитирующая
      естественную интонационную паузу при чтении.

    Ударения (RUAccent) в готовый SSML не добавляются здесь — это делает
    сервер (silero_rest_service.py) при получении запроса, в том числе для
    уже готового SSML, который присылает этот клиент.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()] or [text.strip()]

    p_chunks = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        s_chunks = []
        for sent in sentences:
            escaped = _xml_escape(sent)
            escaped = _SOFT_PAUSE_RE.sub(
                lambda m: f'{_xml_escape(m.group(1))}<break time="{comma_break_ms}ms"/> ',
                escaped,
            )
            s_chunks.append(_prosody_wrap(escaped, sent, emphasize))
        if s_chunks:
            joiner = f'<break time="{sentence_break_ms}ms"/>'
            p_chunks.append("<p>" + joiner.join(s_chunks) + "</p>")

    joiner = f'<break time="{paragraph_break_ms}ms"/>'
    return "<speak>" + joiner.join(p_chunks) + "</speak>"


def _split_paragraphs_for_ssml(text: str, max_len: int):
    """Делит текст главы на куски по абзацам/предложениям так, чтобы
    длина обычного текста каждого куска не превышала max_len (SSML-разметка
    добавляется уже поверх каждого куска отдельно)."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks, cur = [], []
    cur_len = 0
    for para in paragraphs:
        if cur_len + len(para) + 1 > max_len and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        if len(para) > max_len:
            # длинный абзац без переносов — режем по предложениям
            for sent in split_text(para, max_len):
                if cur_len + len(sent) + 1 > max_len and cur:
                    chunks.append("\n".join(cur))
                    cur, cur_len = [], 0
                cur.append(sent)
                cur_len += len(sent) + 1
        else:
            cur.append(para)
            cur_len += len(para) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [text]


def run_silero_rest(chapters, outdir: Path, start: int, speaker: str, sample_rate: int,
                     play: bool, rest_url: str, sentence_break_ms: int, paragraph_break_ms: int,
                     comma_break_ms: int, emphasize: bool = True, max_len: int = 700,
                     model_id: str = DEFAULT_SILERO_MODEL, on_progress=None,
                     chapter_indices=None, char_ranges=None, dialogue_speakers=None, attribution=None,
                     should_stop=None, play_fn=None):
    """Озвучка через Silero-REST-Service (см. https://github.com/Flokss/Silero-REST-Service).

    Текст каждой главы автоматически превращается в SSML с интонационными
    паузами (см. text_to_ssml) и отправляется на эндпоинт /getssmlwav
    (нужен патч сервиса, добавляющий поддержку ssml_text/SSML — обычный
    /getwav SSML не понимает).

    Синтез каждого фрагмента проходит по каскаду вариантов, чтобы почти
    всегда что-то сгенерировалось, а не пропускалось молча:
      1) SSML с паузами и усиленной интонацией (как обычно);
      2) тот же текст через /getwav (обычный текст, без SSML) — на случай,
         если конкретный SSML сервис не смог разобрать (сервис сам внутри
         себя тоже пробует упрощённые варианты, см. silero_rest_service.py);
      3) если и это не удалось (например, сервис вообще недоступен) — в
         файл вставляется короткая тишина вместо фрагмента, чтобы длина
         и порядок остальных фрагментов не съезжали, а не пропускается
         совсем без следа.
    Все ошибки подробно пишутся в лог-файл рядом с аудио (см. LOG_FILE).
    """
    import io
    import time
    import traceback as tb_module
    import requests
    import numpy as np
    import wave as wave_mod

    if dialogue_speakers:
        allowed = speakers_for_model(model_id)
        bad = [s for s in dialogue_speakers if s not in allowed]
        dialogue_speakers = [s for s in dialogue_speakers if s in allowed]
        if bad:
            print(f"Голоса {bad} недоступны для {model_id}, пропускаю их в чередовании диалогов.")

    rest_url = rest_url.rstrip("/")
    ssml_endpoint = f"{rest_url}/getssmlwav"
    plain_endpoint = f"{rest_url}/getwav"
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / "silero_rest_client.log"

    def log(message: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(f"  {message}")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _error_detail(exc: Exception) -> str:
        """Достаёт понятное сообщение об ошибке: если сервер ответил
        HTTPException с текстом (detail), берём его — там обычно и есть
        настоящая причина сбоя, а не просто код 400."""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                return f"HTTP {resp.status_code}: {resp.json().get('detail', resp.text)[:500]}"
            except Exception:
                return f"HTTP {resp.status_code}: {resp.text[:500]}"
        return f"{type(exc).__name__}: {exc}"

    def _get_with_retry(url: str, params: dict, retries: int = 3, backoff_s: float = 5.0):
        """Повторяет запрос при обрыве соединения (например, сервис как раз
        перезапускается) — именно так выглядела ситуация в логах: сервис
        ненадолго "падал"/перезапускался, и все запросы в этом окне
        получали ConnectionRefused. HTTP-ошибки (4xx/5xx, т.е. сервис
        ответил, но не смог синтезировать) не повторяем — там причина не в
        временной недоступности, и retry всё равно даст ту же ошибку."""
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=300)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
                if attempt < retries:
                    log(f"сервис недоступен ({type(e).__name__}), попытка {attempt}/{retries}, "
                        f"жду {backoff_s:.0f} сек и пробую снова...")
                    time.sleep(backoff_s)
            except requests.exceptions.HTTPError as e:
                raise e
        raise last_exc

    def synth_via_ssml(plain_text_chunk: str, chunk_speaker: str) -> bytes:
        ssml = text_to_ssml(
            plain_text_chunk,
            sentence_break_ms=sentence_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
            emphasize=emphasize,
        )
        resp = _get_with_retry(ssml_endpoint, {
            "text_to_speech": ssml,
            "speaker": chunk_speaker,
            "sample_rate": sample_rate,
            "raw_ssml": "true",
        })
        level = resp.headers.get("X-Synthesis-Level", "as-is")
        if level != "as-is":
            log(f"сервис использовал упрощённый вариант синтеза: {level}")
        return resp.content

    def synth_via_plain_text(plain_text_chunk: str, chunk_speaker: str) -> bytes:
        resp = _get_with_retry(plain_endpoint, {
            "text_to_speech": plain_text_chunk,
            "speaker": chunk_speaker,
            "sample_rate": sample_rate,
        })
        return resp.content

    def synth_chunk(plain_text_chunk: str, chunk_no: int, idx: int, title: str,
                     chunk_speaker: str) -> "np.ndarray":
        if not _text_has_speakable_content(plain_text_chunk):
            # разделитель сцен вроде "* * *" — нечего озвучивать, просто
            # короткая пауза вместо него, без обращения к сервису
            return np.zeros(int(sample_rate * 0.4), dtype=np.float32)
        try:
            wav_bytes = synth_via_ssml(plain_text_chunk, chunk_speaker)
        except Exception as e_ssml:
            log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] SSML-синтез не удался "
                f"({_error_detail(e_ssml)}), пробую обычный текст без SSML...")
            try:
                wav_bytes = synth_via_plain_text(plain_text_chunk, chunk_speaker)
                log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] синтез обычным текстом удался")
            except Exception as e_plain:
                log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] ОШИБКА: не удалось синтезировать "
                    f"даже обычным текстом ({_error_detail(e_plain)}). "
                    f"Вставляю тишину вместо фрагмента, чтобы не потерять место в главе.")
                log("Полный traceback последней ошибки:\n" + tb_module.format_exc())
                # тишина длиной пропорционально длине текста (примерно как
                # если бы его прочитали) — чтобы не выпадать из ритма главы
                silence_seconds = max(1.0, min(8.0, len(plain_text_chunk) / 15))
                return np.zeros(int(sample_rate * silence_seconds), dtype=np.float32)

        with wave_mod.open(io.BytesIO(wav_bytes), "rb") as wf:
            n = wf.getnframes()
            pcm = wf.readframes(n)
            return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="silero_rest", model=model_id, speaker=speaker, sample_rate=sample_rate,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms, emphasize=emphasize, max_len=max_len,
            dialogue_speakers=",".join(dialogue_speakers) if dialogue_speakers else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено с теми же параметрами): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (silero_rest): {title} -> {fname}")

        # Группируем по голосу (диалоги/повествование и/или LLM-атрибуция
        # по персонажам, см. attribution), затем каждую группу — как
        # раньше, на куски под SSML-лимит max_len.
        voice_groups = resolve_voice_groups(text, speaker, dialogue_speakers, outdir,
                                             attribution=attribution, log_fn=log)
        chunks = []  # [(текст, голос), ...]
        for group_voice, group_text in voice_groups:
            for chunk in _split_paragraphs_for_ssml(group_text, max_len):
                if chunk.strip():
                    chunks.append((chunk, group_voice))

        pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
        audio_parts = []
        chunks_total = len(chunks) or 1
        if on_progress:
            on_progress(pos, total, 0, chunks_total)

        for chunk_no, (chunk, chunk_voice) in enumerate(
            tqdm(chunks, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if not chunk.strip():
                if on_progress:
                    on_progress(pos, total, chunk_no, chunks_total)
                continue
            audio = synth_chunk(chunk, chunk_no, idx, title, chunk_voice)
            audio_parts.append(audio)
            audio_parts.append(pause)
            if on_progress:
                on_progress(pos, total, chunk_no, chunks_total)

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        with wave_mod.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")
    print(f"Подробный лог ошибок (если были): {log_path.resolve()}")


def cosyvoice_list_voices(rest_url: str) -> list:
    """Запрашивает у сервиса cosyvoice_rest_service.py список загруженных
    профилей голоса (см. cosyvoice_rest_service.py: POST /add_voice их
    добавляет, GET /voices — перечисляет). Возвращает [] и не бросает
    исключение, если сервис недоступен — вызывающий код (GUI) сам решает,
    как это показать пользователю."""
    import requests
    try:
        resp = requests.get(f"{rest_url.rstrip('/')}/voices", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("voices", [])
    except Exception:
        return []


def run_cosyvoice(chapters, outdir: Path, start: int, voice: str, sample_rate: int,
                   play: bool, rest_url: str, sentence_break_ms: int = 250,
                   paragraph_break_ms: int = 450, comma_break_ms: int = 120,
                   max_len: int = 400, on_progress=None,
                   chapter_indices=None, char_ranges=None, dialogue_voices=None, attribution=None,
                   should_stop=None, play_fn=None):
    """Озвучка через локальный сервис cosyvoice_rest_service.py (CosyVoice2,
    работает на GPU) — см. install_cosyvoice.bat/README для установки в
    отдельное окружение .venv_cosyvoice. В отличие от Silero/Yandex, у
    CosyVoice нет готовых голосов "из коробки" — вместо этого сервис хранит
    именованные "профили голоса" (короткий образец аудио + его текст),
    voice — имя одного из них (см. cosyvoice_list_voices/эндпоинт /voices).

    dialogue_voices здесь — список ИМЁН профилей (а не голосов Silero/
    Yandex), по которым чередуются реплики диалогов — работает точно так
    же, как dialogue_speakers/dialogue_voices в других режимах, включая
    LLM-атрибуцию по персонажам (attribution).

    Паузы между предложениями/абзацами/запятыми (sentence_break_ms,
    paragraph_break_ms, comma_break_ms) — CosyVoice, в отличие от
    silero_rest, не понимает SSML, так что паузы вставляются вручную:
    текст режется на куски по пунктуации через _segments_with_pauses (как
    в локальном режиме silero), и после каждого куска в готовое аудио
    добавляется настоящая тишина нужной длины — а не фиксированные 0.3 сек
    между любыми кусками, как было раньше."""
    import io
    import time
    import traceback as tb_module
    import requests
    import numpy as np
    import wave as wave_mod

    rest_url = rest_url.rstrip("/")
    endpoint = f"{rest_url}/getwav"
    outdir.mkdir(parents=True, exist_ok=True)

    # Какой движок сейчас реально обслуживает сервис (f5/xtts) - попадает в
    # "отпечаток" параметров ниже, чтобы при переключении движка (или при
    # смене версии сервиса, где менялась логика синтеза - см. историю
    # cosyvoice_rest_service.py) уже "готовые" главы не считались готовыми
    # молча навсегда. Без этого, например, глава, синтезированная в момент,
    # когда сервис падал на каждой фразе и вместо звука вставлял тишину,
    # так и осталась бы отмеченной как "уже озвучено" даже после того, как
    # сама причина ошибки исправлена - до этой правки так и произошло.
    engine_tag = "unknown"
    try:
        health = requests.get(f"{rest_url}/health", timeout=5).json()
        engine_tag = f"{health.get('engine')}:{health.get('service_version')}"
    except Exception:
        pass

    log_path = outdir / "cosyvoice_client.log"

    def log(message: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(f"  {message}")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _error_detail(exc: Exception) -> str:
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                return f"HTTP {resp.status_code}: {resp.json().get('detail', resp.text)[:500]}"
            except Exception:
                return f"HTTP {resp.status_code}: {resp.text[:500]}"
        return f"{type(exc).__name__}: {exc}"

    def _get_with_retry(params: dict, retries: int = 3, backoff_s: float = 5.0):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                # CosyVoice-синтез заметно медленнее Silero даже на GPU
                # (секунды на фразу) — таймаут увеличен с запасом.
                resp = requests.get(endpoint, params=params, timeout=600)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
                if attempt < retries:
                    log(f"сервис недоступен ({type(e).__name__}), попытка {attempt}/{retries}, "
                        f"жду {backoff_s:.0f} сек и пробую снова...")
                    time.sleep(backoff_s)
            except requests.exceptions.HTTPError as e:
                raise e
        raise last_exc

    def synth_chunk(plain_text_chunk: str, chunk_no: int, idx: int, title: str,
                     chunk_voice: str) -> "np.ndarray":
        if not _text_has_speakable_content(plain_text_chunk):
            return np.zeros(int(sample_rate * 0.1), dtype=np.float32)
        try:
            resp = _get_with_retry({
                "text_to_speech": plain_text_chunk,
                "voice": chunk_voice,
                "sample_rate": sample_rate,
            })
        except Exception as e:
            log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] ОШИБКА: не удалось синтезировать "
                f"({_error_detail(e)}). Вставляю тишину вместо фрагмента, чтобы не потерять "
                f"место в главе — ЭТО НЕ НОРМАЛЬНО, проверьте логи сервиса CosyVoice "
                f"(cosyvoice_rest_service.log в папке CosyVoice) — обычно причина в том, "
                f"что модель или голосовой профиль не загрузились при старте сервиса.")
            log("Полный traceback последней ошибки:\n" + tb_module.format_exc())
            silence_seconds = max(1.0, min(8.0, len(plain_text_chunk) / 15))
            return np.zeros(int(sample_rate * silence_seconds), dtype=np.float32)

        with wave_mod.open(io.BytesIO(resp.content), "rb") as wf:
            n = wf.getnframes()
            pcm = wf.readframes(n)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        if audio.size == 0 or float(np.abs(audio).max()) < 1e-4:
            log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] ПРЕДУПРЕЖДЕНИЕ: сервис вернул "
                f"пустой или практически беззвучный WAV (без ошибки HTTP) — если это "
                f"повторяется на каждом фрагменте, скорее всего профиль голоса "
                f"{chunk_voice!r} не загрузился на сервисе (см. cosyvoice_rest_service.log).")
        return audio

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="cosyvoice", voice=voice, sample_rate=sample_rate, max_len=max_len,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms, engine=engine_tag,
            dialogue_voices=",".join(dialogue_voices) if dialogue_voices else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено с теми же параметрами): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (CosyVoice): {title} -> {fname}")

        voice_groups = resolve_voice_groups(text, voice, dialogue_voices, outdir,
                                             attribution=attribution, log_fn=log)
        # Как и в локальном режиме silero: режем на куски по пунктуации и
        # для каждого куска сразу знаем длину паузы (мс) ПОСЛЕ него.
        segments = []  # [(текст, пауза_мс, профиль_голоса), ...]
        for group_voice, group_text in voice_groups:
            for part_text, pause_ms in _segments_with_pauses(
                group_text, sentence_break_ms=sentence_break_ms,
                paragraph_break_ms=paragraph_break_ms, comma_break_ms=comma_break_ms,
                max_chars=max_len,
            ):
                if part_text.strip():
                    segments.append((part_text, pause_ms, group_voice))

        audio_parts = []
        segs_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, segs_total)

        for seg_i, (part_text, pause_ms, seg_voice) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            audio = synth_chunk(part_text, seg_i, idx, title, seg_voice)
            audio_parts.append(audio)
            if pause_ms > 0:
                audio_parts.append(np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32))
            if on_progress:
                on_progress(pos, total, seg_i, segs_total)

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        with wave_mod.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")
    print(f"Подробный лог ошибок (если были): {log_path.resolve()}")


def list_offline_voices():
    """Возвращает список системных голосов pyttsx3: [{id, name, is_russian}, ...]."""
    import pyttsx3
    engine = pyttsx3.init()
    result = []
    for v in engine.getProperty("voices") or []:
        vid = v.id or ""
        vname = getattr(v, "name", "") or vid
        langs = getattr(v, "languages", [])
        lang_str = " ".join(str(l) for l in langs).lower()
        is_russian = (
            "ru" in vid.lower()
            or "russian" in vname.lower()
            or "ru" in lang_str
        )
        result.append({"id": vid, "name": vname, "is_russian": is_russian})
    try:
        engine.stop()
    except Exception:
        pass
    result.sort(key=lambda x: (not x["is_russian"], x["name"].lower()))
    return result


_PIPER_VOICE_CACHE: dict = {}


def _piper_voice_dir() -> Path:
    d = Path(__file__).resolve().parent / "piper_voices"
    d.mkdir(exist_ok=True)
    return d


def _load_piper_voice(name: str):
    """Скачивает (при первом обращении, в piper_voices/ рядом со скриптом —
    см. .gitignore) .onnx + .onnx.json нужного голоса Piper с HuggingFace
    (rhasspy/piper-voices) и загружает его через PiperVoice.load(). Голоса
    кэшируются в памяти на весь процесс — не грузим заново на каждую фразу
    и не держим больше одной копии голоса в памяти одновременно."""
    if name in _PIPER_VOICE_CACHE:
        return _PIPER_VOICE_CACHE[name]

    import requests
    from piper import PiperVoice

    voice_dir = _piper_voice_dir()
    fname_onnx = f"ru_RU-{name}-medium.onnx"
    fname_json = f"ru_RU-{name}-medium.onnx.json"
    base_url = f"https://huggingface.co/{PIPER_VOICES_REPO}/resolve/main/ru/ru_RU/{name}/medium"

    for fname in (fname_onnx, fname_json):
        path = voice_dir / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Скачиваю голос Piper {name!r} ({fname})…")
        resp = requests.get(f"{base_url}/{fname}?download=true", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)

    voice = PiperVoice.load(str(voice_dir / fname_onnx))
    _PIPER_VOICE_CACHE[name] = voice
    return voice


def run_piper(chapters, outdir: Path, start: int, voice: str, play: bool,
              sentence_break_ms: int = 250, paragraph_break_ms: int = 450,
              comma_break_ms: int = 120, on_progress=None,
              chapter_indices=None, char_ranges=None, dialogue_voices=None, attribution=None,
              should_stop=None, play_fn=None):
    """Озвучка через Piper TTS — маленький и быстрый (в отличие от Silero и
    тем более cosyvoice) локальный CPU-движок без клонирования голоса,
    четыре готовых русских голоса (см. PIPER_VOICES). Синтезирует заметно
    быстрее Silero, но качество/выразительность речи скромнее (нет
    отдельной расстановки ударений/интонации — только то, что заложено в
    сам голос). Текст режется на куски с паузами так же, как в silero/
    cosyvoice (см. _segments_with_pauses), voice — ключ из PIPER_VOICES.

    dialogue_voices — необязательный список ключей PIPER_VOICES, которыми
    по очереди озвучиваются реплики прямой речи (как в других режимах, см.
    resolve_voice_groups)."""
    import io
    import wave
    import numpy as np

    if voice not in PIPER_VOICES:
        print(f"Голос {voice!r} недоступен для Piper, использую {PIPER_DEFAULT_VOICE!r}")
        voice = PIPER_DEFAULT_VOICE
    if dialogue_voices:
        bad = [v for v in dialogue_voices if v not in PIPER_VOICES]
        dialogue_voices = [v for v in dialogue_voices if v in PIPER_VOICES]
        if bad:
            print(f"Голоса {bad} недоступны для Piper, пропускаю их в чередовании диалогов.")

    outdir.mkdir(parents=True, exist_ok=True)
    max_chars = 400

    def synth_one(part_text: str, part_voice: str) -> "np.ndarray":
        piper_voice = _load_piper_voice(part_voice)
        part_text = numbers_to_words_ru(part_text)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            piper_voice.synthesize_wav(part_text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    def synth_with_fallback(part_text: str, idx: int, title: str, part_voice: str) -> "np.ndarray":
        if not _text_has_speakable_content(part_text):
            return np.zeros(int(PIPER_SAMPLE_RATE * 0.4), dtype=np.float32)
        try:
            return synth_one(part_text, part_voice)
        except Exception as e:
            print(f"  [Гл.{idx} «{title}»] ОШИБКА: не удалось синтезировать фрагмент Piper "
                  f"({e}). Вставляю тишину вместо него: {part_text[:80]!r}…")
            silence_seconds = max(0.4, min(6.0, len(part_text) / 15))
            return np.zeros(int(PIPER_SAMPLE_RATE * silence_seconds), dtype=np.float32)

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="piper", voice=voice, max_chars=max_chars,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
            dialogue_voices=",".join(dialogue_voices) if dialogue_voices else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        if _is_already_done(out_path, fingerprint):
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено с теми же параметрами): {title} -> {fname}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(out_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (Piper): {title} -> {fname}")

        voice_groups = resolve_voice_groups(
            text, voice, dialogue_voices, outdir, attribution=attribution,
            log_fn=_make_file_logger(outdir, "piper_client.log"),
        )
        segments = []
        for group_voice, group_text in voice_groups:
            for part_text, pause_ms in _segments_with_pauses(
                group_text, sentence_break_ms=sentence_break_ms,
                paragraph_break_ms=paragraph_break_ms, comma_break_ms=comma_break_ms,
                max_chars=max_chars,
            ):
                segments.append((part_text, pause_ms, group_voice))

        audio_parts = []
        segs_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, segs_total)

        for seg_i, (part_text, pause_ms, seg_voice) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if not part_text.strip():
                if on_progress:
                    on_progress(pos, total, seg_i, segs_total)
                continue
            audio = synth_with_fallback(part_text, idx, title, seg_voice)
            audio_parts.append(audio)
            if pause_ms > 0:
                audio_parts.append(np.zeros(int(PIPER_SAMPLE_RATE * pause_ms / 1000), dtype=np.float32))
            if on_progress:
                on_progress(pos, total, seg_i, segs_total)

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PIPER_SAMPLE_RATE)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


def run_offline(chapters, start: int, rate: int, voice_hint: str, voice_id: str = "", on_progress=None,
                 chapter_indices=None, char_ranges=None, dialogue_voice_ids=None, attribution=None,
                 outdir: Path = None, should_stop=None):
    """outdir здесь используется только для сохранения словаря "персонаж ->
    голос" при dialogue_voice_ids + attribution (offline-режим ничего не
    пишет на диск сам по себе — говорит вслух сразу) — если не передан,
    берётся текущая папка."""
    outdir = outdir or Path(".")
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)

    chosen = voice_id or None
    if not chosen:
        for v in engine.getProperty("voices") or []:
            vid = (v.id or "").lower()
            vname = (getattr(v, "name", "") or "").lower()
            langs = getattr(v, "languages", [])
            lang_str = " ".join(str(l) for l in langs).lower()
            if "ru" in vid or "russian" in vname or "ru" in lang_str or (voice_hint and voice_hint.lower() in vid):
                chosen = v.id
                break
    if chosen:
        engine.setProperty("voice", chosen)
        print(f"Использую голос: {chosen}")
    else:
        print("ВНИМАНИЕ: русский голос не найден в системе — "
              "звучать будет на голосе по умолчанию (может звучать неразборчиво).\n"
              "На Linux установите: sudo apt install espeak-ng espeak-ng-data")

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1
    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        print(f"\n[{idx}/{len(chapters)}] {title}")
        if not dialogue_voice_ids:
            if on_progress:
                on_progress(pos, total, 0, 1)
            engine.say(text)
            engine.runAndWait()
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        # Разные голоса для диалогов: группируем по голосу (см.
        # resolve_voice_groups — простое чередование и/или LLM-атрибуция по
        # персонажам) и переключаем голос движка перед каждой группой —
        # say()/runAndWait() читает синхронно, поэтому можно спокойно
        # менять voice между вызовами.
        voice_groups = resolve_voice_groups(text, chosen or "", dialogue_voice_ids, outdir,
                                             attribution=attribution,
                                             log_fn=_make_file_logger(outdir, "offline_client.log"))
        groups_total = len(voice_groups) or 1
        if on_progress:
            on_progress(pos, total, 0, groups_total)
        for g_i, (group_voice, group_text) in enumerate(voice_groups, 1):
            try:
                engine.setProperty("voice", group_voice or chosen)
            except Exception as e:
                print(f"  Не удалось переключить голос на {group_voice!r} ({e}), "
                      f"использую текущий.")
            engine.say(group_text)
            engine.runAndWait()
            if on_progress:
                on_progress(pos, total, g_i, groups_total)
        # возвращаем основной голос на случай следующей главы без диалогов
        try:
            engine.setProperty("voice", chosen)
        except Exception:
            pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Озвучивание FB2-книг на русском языке")
    ap.add_argument("book", type=Path, nargs="?", help="путь к .fb2 или .fb2.zip файлу")
    ap.add_argument("--gui", action="store_true", help="запустить графический интерфейс")
    ap.add_argument("--mode", choices=["online", "offline", "silero", "silero_rest", "cosyvoice", "piper", "yandex"],
                     default="silero",
                     help="silero = нейросетевой голос локально через torch.hub; "
                          "silero_rest = синтез через Silero-REST-Service с интонационными паузами "
                          "(SSML: паузы на запятых/тире, границах предложений и абзацев, интонация "
                          "вопросов/восклицаний); cosyvoice = локальный сервис CosyVoice (GPU, "
                          "клонирование голоса по образцу — см. install_cosyvoice.bat); "
                          "piper = Piper TTS (локально, CPU, быстро, без клонирования — "
                          "см. --piper-voice); "
                          "yandex = облачный Yandex SpeechKit (платно, нужны "
                          "--yandex-api-key и --yandex-folder-id); online = gTTS (интернет); "
                          "offline = pyttsx3 (без интернета)")
    ap.add_argument("--outdir", type=Path, default=Path("audiobook_output"),
                     help="папка для сохранения аудио (online и silero режимы)")
    ap.add_argument("--play", action="store_true",
                     help="сразу проигрывать главы после озвучки (online и silero режимы)")
    ap.add_argument("--start", type=int, default=1, help="с какой главы начать (1 = с начала; "
                     "игнорируется, если указан --chapters)")
    ap.add_argument("--chapters", type=str, default="",
                     help="озвучить только перечисленные главы вместо диапазона от --start, "
                          "например: \"1,3,5-7\" (номера с 1, можно вперемешку с диапазонами)")
    ap.add_argument("--rate", type=int, default=170, help="скорость речи для offline-режима (слов/мин)")
    ap.add_argument("--voice", type=str, default="", help="подсказка имени голоса для offline-режима")
    ap.add_argument("--speaker", type=str, default="xenia",
                     help="голос Silero: aidar, baya, kseniya, xenia, eugene "
                          "(random — только для --model v4_ru)")
    ap.add_argument("--dialogue-speakers", type=str, default="",
                     help="список голосов Silero через запятую (например: aidar,eugene,baya) — "
                          "если задан, реплики диалогов (абзацы, начинающиеся с тире) будут по "
                          "очереди озвучены этими голосами вместо основного --speaker (режимы "
                          "silero и silero_rest); по умолчанию не задано — один голос на книгу")
    ap.add_argument("--dialogue-voice-ids", type=str, default="",
                     help="список системных голосов (id, через запятую) для диалогов в "
                          "offline-режиме — см. --voice/список системных голосов в GUI")
    ap.add_argument("--model", type=str, default=DEFAULT_SILERO_MODEL,
                     choices=list(SILERO_MODELS.keys()),
                     help="модель Silero для silero/silero_rest "
                          f"(по умолчанию {DEFAULT_SILERO_MODEL} — последняя)")
    ap.add_argument("--sample-rate", type=int, default=48000,
                     help="частота дискретизации для silero-режима (8000/24000/48000)")
    ap.add_argument("--list", action="store_true", help="только показать список глав и выйти")
    ap.add_argument("--rest-url", type=str, default="http://localhost:5010",
                     help="адрес Silero-REST-Service для режима silero_rest")
    ap.add_argument("--cosyvoice-rest-url", type=str, default=COSYVOICE_DEFAULT_REST_URL,
                     help="адрес сервиса cosyvoice_rest_service.py для режима cosyvoice "
                          f"(по умолчанию {COSYVOICE_DEFAULT_REST_URL})")
    ap.add_argument("--cosyvoice-voice", type=str, default="default",
                     help="имя профиля голоса CosyVoice (см. GET /voices сервиса) — по умолчанию "
                          "'default', встроенный образец, который ставит install_cosyvoice.bat")
    ap.add_argument("--cosyvoice-dialogue-voices", type=str, default="",
                     help="список имён профилей голоса CosyVoice через запятую — если задан, "
                          "реплики диалогов по очереди озвучиваются этими профилями вместо "
                          "--cosyvoice-voice (режим cosyvoice)")
    ap.add_argument("--cosyvoice-max-len", type=int, default=400,
                     help="макс. длина куска текста за один вызов CosyVoice, символов "
                          "(по умолчанию 400 — CosyVoice медленнее Silero, куски короче удобнее "
                          "для прогресса/повторов при сбое)")
    ap.add_argument("--piper-voice", type=str, default=PIPER_DEFAULT_VOICE,
                     choices=list(PIPER_VOICES.keys()),
                     help=f"голос Piper (по умолчанию {PIPER_DEFAULT_VOICE!r}) — режим piper")
    ap.add_argument("--piper-dialogue-voices", type=str, default="",
                     help="список голосов Piper через запятую — если задан, реплики диалогов по "
                          "очереди озвучиваются этими голосами вместо --piper-voice (режим piper)")
    ap.add_argument("--sentence-break-ms", type=int, default=320,
                     help="пауза между предложениями в silero/silero_rest/cosyvoice-режимах (мс)")
    ap.add_argument("--paragraph-break-ms", type=int, default=550,
                     help="пауза между абзацами в silero/silero_rest-режимах (мс)")
    ap.add_argument("--comma-break-ms", type=int, default=180,
                     help="пауза на запятых/тире/двоеточиях в silero/silero_rest-режимах (мс)")
    ap.add_argument("--no-emphasis", action="store_true",
                     help="не усиливать интонацию вопросительных/восклицательных "
                          "предложений через <prosody> в silero_rest-режиме "
                          "(по умолчанию усиление включено)")
    ap.add_argument("--no-accent", action="store_true",
                     help="не расставлять ударения автоматически в silero-режиме "
                          "(по умолчанию расставляются)")
    ap.add_argument("--no-yo", action="store_true",
                     help="не заменять 'е' на 'ё' там, где нужно, в silero-режиме "
                          "(по умолчанию заменяется)")
    ap.add_argument("--yandex-api-key", type=str, default="",
                     help="API-ключ Yandex SpeechKit (режим yandex); можно также задать "
                          "переменной окружения YANDEX_API_KEY")
    ap.add_argument("--yandex-folder-id", type=str, default="",
                     help="Folder ID каталога в Yandex Cloud (режим yandex); можно также задать "
                          "переменной окружения YANDEX_FOLDER_ID")
    ap.add_argument("--yandex-voice", type=str, default="alena",
                     choices=list(YANDEX_VOICES.keys()),
                     help="голос Yandex SpeechKit (режим yandex)")
    ap.add_argument("--yandex-emotion", type=str, default="",
                     help="эмоция для голосов, которые её поддерживают (ermil, jane): "
                          "good, neutral, evil")
    ap.add_argument("--yandex-speed", type=float, default=1.0,
                     help="скорость речи Yandex SpeechKit, от 0.1 до 3.0 (по умолчанию 1.0)")
    ap.add_argument("--yandex-dialogue-voices", type=str, default="",
                     help="список голосов через запятую (например: jane,filipp,zahar) — если "
                          "задан, реплики диалогов (абзацы, начинающиеся с тире) будут по "
                          "очереди озвучены этими голосами вместо голоса --yandex-voice, "
                          "чтобы диалоги не звучали одним монотонным голосом; по умолчанию "
                          "не задано — вся книга одним голосом, как раньше")
    ap.add_argument("--check-model-updates", action="store_true",
                     help="проверить на GitHub, не появилась ли более новая модель "
                          "Silero для русского языка, и выйти")
    ap.add_argument("--attribution-provider", type=str, default=DEFAULT_ATTRIBUTION_PROVIDER,
                     choices=list(ATTRIBUTION_PROVIDERS.keys()),
                     help="сервис для определения, какой персонаж говорит каждую реплику "
                          "диалога: yandexgpt (работает из РФ без VPN, есть бесплатный лимит), "
                          "gemini (Google, бесплатно, но часто недоступен из РФ) или anthropic "
                          f"(Claude, платно). По умолчанию {DEFAULT_ATTRIBUTION_PROVIDER}.")
    ap.add_argument("--attribution-api-key", type=str, default="",
                     help="API-ключ для выбранного --attribution-provider (для yandexgpt можно "
                          "использовать тот же ключ, что и --yandex-api-key); можно также "
                          "задать переменной окружения ANTHROPIC_API_KEY, GEMINI_API_KEY или "
                          "YANDEX_API_KEY. Без него голоса для диалогов просто чередуются по "
                          "кругу (--dialogue-*), без привязки к конкретному персонажу.")
    ap.add_argument("--attribution-model", type=str, default="",
                     help="модель для определения говорящего (по умолчанию — модель, "
                          "рекомендованная для выбранного --attribution-provider); актуальные "
                          "названия моделей см. в документации соответствующего сервиса")
    ap.add_argument("--attribution-folder-id", type=str, default="",
                     help="Folder ID для провайдера yandexgpt — можно использовать тот же, что "
                          "и --yandex-folder-id; для gemini/anthropic не нужен")
    args = ap.parse_args()

    if args.check_model_updates:
        from silero_config import check_for_model_updates
        print("Проверяю обновления моделей Silero...")
        result = check_for_model_updates()
        if not result["ok"]:
            print(f"Не удалось проверить: {result['error']}")
        elif result["new_models"]:
            print("Найдены модели, которых ещё нет в этом скрипте: " + ", ".join(result["new_models"]))
            print("Добавьте их в SILERO_MODELS в silero_config.py, чтобы использовать.")
        else:
            print(f"Новых моделей нет. Известные Silero-модели ru: {', '.join(result['checked'])}")
        return

    # GUI: --gui, запуск без аргументов, или book.fb2 вместе с --gui
    if args.gui or (args.book is None and len(sys.argv) == 1):
        from fb2_reader_gui import run_gui
        run_gui(initial_book=args.book)
        return

    if not args.book:
        ap.error("укажите путь к книге или запустите без аргументов / с --gui для графического интерфейса")

    if not args.book.exists():
        print(f"Файл не найден: {args.book}", file=sys.stderr)
        sys.exit(1)

    print(f"Читаю файл: {args.book}")
    title, chapters = parse_fb2(args.book)
    print(f"Книга: {title}")
    print(f"Найдено глав: {len(chapters)}\n")

    if args.list:
        for i, (ch_title, text) in enumerate(chapters, 1):
            print(f"  {i:3d}. {ch_title}  ({len(text)} символов)")
        return

    if args.mode in ("silero", "silero_rest"):
        allowed = speakers_for_model(args.model)
        if args.speaker not in allowed:
            print(f"Предупреждение: голос {args.speaker!r} недоступен для модели {args.model}. "
                  f"Доступны: {', '.join(allowed)}", file=sys.stderr)
            args.speaker = "xenia" if "xenia" in allowed else allowed[0]
            print(f"Использую голос: {args.speaker}")

    chapter_indices = _parse_chapters_spec(args.chapters)
    if chapter_indices:
        bad = [i for i in chapter_indices if not (1 <= i <= len(chapters))]
        if bad:
            ap.error(f"--chapters: нет таких глав: {', '.join(str(i) for i in bad)} "
                      f"(всего глав: {len(chapters)})")

    dialogue_speakers = [v.strip() for v in args.dialogue_speakers.split(",") if v.strip()] or None
    dialogue_voice_ids = [v.strip() for v in args.dialogue_voice_ids.split(",") if v.strip()] or None

    attribution_provider = args.attribution_provider or DEFAULT_ATTRIBUTION_PROVIDER
    attribution_api_key = args.attribution_api_key or os.environ.get("YANDEX_API_KEY", "") \
        or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    attribution_model = args.attribution_model or ATTRIBUTION_PROVIDERS[attribution_provider]["default_model"]
    attribution_folder_id = args.attribution_folder_id or args.yandex_folder_id
    attribution = {
        "api_key": attribution_api_key, "model": attribution_model,
        "provider": attribution_provider, "folder_id": attribution_folder_id,
    } if attribution_api_key else None

    if args.mode == "online":
        run_online(chapters, args.outdir, args.play, args.start, voice_lang="ru",
                   chapter_indices=chapter_indices, dialogue_voices=dialogue_speakers)
        print(f"\nГотово. Файлы сохранены в: {args.outdir.resolve()}")
    elif args.mode == "silero":
        run_silero(chapters, args.outdir, args.start, args.speaker, args.sample_rate, args.play,
                   model_id=args.model, sentence_break_ms=args.sentence_break_ms,
                   paragraph_break_ms=args.paragraph_break_ms, comma_break_ms=args.comma_break_ms,
                   put_accent=not args.no_accent, put_yo=not args.no_yo,
                   chapter_indices=chapter_indices, dialogue_speakers=dialogue_speakers,
                   attribution=attribution)
    elif args.mode == "silero_rest":
        run_silero_rest(chapters, args.outdir, args.start, args.speaker, args.sample_rate, args.play,
                         args.rest_url, args.sentence_break_ms, args.paragraph_break_ms,
                         args.comma_break_ms, emphasize=not args.no_emphasis, model_id=args.model,
                         chapter_indices=chapter_indices, dialogue_speakers=dialogue_speakers,
                         attribution=attribution)
    elif args.mode == "cosyvoice":
        cosyvoice_dialogue_voices = [v.strip() for v in args.cosyvoice_dialogue_voices.split(",") if v.strip()] or None
        run_cosyvoice(chapters, args.outdir, args.start, args.cosyvoice_voice, args.sample_rate, args.play,
                      args.cosyvoice_rest_url, sentence_break_ms=args.sentence_break_ms,
                      paragraph_break_ms=args.paragraph_break_ms, comma_break_ms=args.comma_break_ms,
                      max_len=args.cosyvoice_max_len,
                      chapter_indices=chapter_indices, dialogue_voices=cosyvoice_dialogue_voices,
                      attribution=attribution)
    elif args.mode == "piper":
        piper_dialogue_voices = [v.strip() for v in args.piper_dialogue_voices.split(",") if v.strip()] or None
        run_piper(chapters, args.outdir, args.start, args.piper_voice, args.play,
                  sentence_break_ms=args.sentence_break_ms, paragraph_break_ms=args.paragraph_break_ms,
                  comma_break_ms=args.comma_break_ms,
                  chapter_indices=chapter_indices, dialogue_voices=piper_dialogue_voices,
                  attribution=attribution)
    elif args.mode == "yandex":
        api_key = args.yandex_api_key or os.environ.get("YANDEX_API_KEY", "")
        folder_id = args.yandex_folder_id or os.environ.get("YANDEX_FOLDER_ID", "")
        dialogue_voices = [v.strip() for v in args.yandex_dialogue_voices.split(",") if v.strip()] or None
        run_yandex(chapters, args.outdir, args.start, args.play, api_key, folder_id,
                   voice=args.yandex_voice, speed=args.yandex_speed, emotion=args.yandex_emotion,
                   chapter_indices=chapter_indices, dialogue_voices=dialogue_voices,
                   attribution=attribution)
    else:
        run_offline(chapters, args.start, args.rate, args.voice, voice_id=args.voice,
                    chapter_indices=chapter_indices, dialogue_voice_ids=dialogue_voice_ids,
                    attribution=attribution, outdir=args.outdir)


if __name__ == "__main__":
    main()