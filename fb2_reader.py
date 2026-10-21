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

Доступные модели Silero (--model): v5_6_ru (последняя, по умолчанию),
v5_5_ru, v5_4_ru, v5_ru, v4_ru (устаревшая, поддерживает random).
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

import app_paths as _app_paths   # one place for source/PyInstaller (sys._MEIPASS) path logic

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
    "qwen_tts": "Qwen3-TTS (облако DashScope, платно, нужен API-ключ)",
    "qwen_tts_local": "Qwen3-TTS (локально, GPU, без интернета/карты)",
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


# Qwen3-TTS (Alibaba) через облачный DashScope API — модель "qwen3-tts-flash".
# В отличие от CosyVoice (свои веса, отдельная GPU-среда, свой install-скрипт)
# это облачный сервис по API-ключу, ближе по духу к Yandex SpeechKit: не
# нужен GPU и локальная модель, но нужен интернет, ключ DashScope и,
# вероятно, платный аккаунт (см. https://help.aliyun.com или
# https://www.alibabacloud.com/help/en/model-studio для актуальных цен —
# на момент добавления это не проверялось вживую).
#
# ВАЖНО (честно, чтобы не удивлять при первом запуске): этот режим написан
# по документации DashScope (модель ответа, поля audio.url и т.п.), но НЕ
# протестирован на живом аккаунте — в песочнице, где писался код, нет ни
# доступа к dashscope.aliyuncs.com, ни API-ключа. Если формат ответа на
# практике отличается — synth_qwen_tts_chunk бросит понятное исключение с
# текстом ошибки/сырым ответом, а не упадёт молча; сообщите точный текст
# ошибки, если она появится, и это будет легко поправить.
#
# Голоса — по офиц. списку голосов Qwen-TTS (см. Alibaba Cloud Model Studio,
# "Qwen-TTS voice list"), многоязычные (не привязаны к одному языку) —
# язык задаётся отдельно параметром language_type.
QWEN_TTS_VOICES = {
    "Cherry": "Cherry (жен.)",
    "Serena": "Serena (жен.)",
    "Ethan": "Ethan (муж.)",
    "Vivian": "Vivian (жен.)",
    "Ryan": "Ryan (муж.)",
}
QWEN_TTS_DEFAULT_VOICE = "Cherry"
QWEN_TTS_LANGUAGE = "Russian"
QWEN_TTS_MODEL = "qwen3-tts-flash"
# У DashScope документированный лимит — 512 токенов на запрос. Токен для
# кириллицы обычно "дороже" символа (BPE режет кириллицу на несколько
# токенов на слово чаще, чем латиницу) — берём с большим запасом, пока не
# проверено вживую, лучше мельче резать и чаще звать API, чем упереться в
# ошибку "текст слишком длинный" на середине главы.
QWEN_TTS_MAX_CHARS = 350


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


# Footnote anchors in this FB2 (and in many epub->fb2 conversions) are literal text glued to the word: "РУФnote 2 ,", "Честь.note 1 Хонор".
# They were read aloud / crashed Silero on a "note шестнадцать" + "* * *" chunk (found in the real-book run, 10 Oct 2026).
_FOOTNOTE_RE = re.compile(r"[ ]*" + r"note" + r"[ ]*" + r"[0-9]+" + r"(?![0-9A-Za-z])" + r"[ ]*")


def strip_footnote_markers(text: str) -> str:
    """Removes "noteN" footnote anchors from chapter text, keeping a single space between the neighbours and no space before
    punctuation ("РУФnote 2 , а" -> "РУФ, а"; "Честь.note 1 Хонор" -> "Честь. Хонор")."""
    if not text or "note" not in text:
        return text
    out = _FOOTNOTE_RE.sub(" ", text)
    out = re.sub(r"[ ]+([,.;:!?»)])", r"\1", out)
    out = re.sub(r"(?m)[ ]+$", "", out)
    return re.sub(r"[ ]{2,}", " ", out)


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
            text = strip_footnote_markers(iter_text(body))
            if text:
                chapters.append(("Текст", text))
            continue
        for i, sec in enumerate(top_sections, 1):
            title = section_title(sec) or f"Глава {i}"
            text = strip_footnote_markers(iter_text(sec))
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

# A leading "-" is a minus sign only when it is not glued to a word/number ("5-6" is a range);
# a hyphen between a number and letters ("20-летие") becomes a space.
_NUMBER_RE = re.compile(r"(?P<neg>(?<![\w-])-)?(?P<num>\d+(?:[.,]\d+)?)(?P<hy>-(?=[^\W\d_]))?")


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
        raw = m.group("num")
        try:
            if "," in raw or "." in raw:
                words = num2words(float(raw.replace(",", ".")), lang="ru")
            else:
                words = num2words(int(raw), lang="ru")
        except Exception:
            return m.group(0)
        if m.group("neg"):
            words = "минус " + words
        # The words are always separated from neighbouring letters/digits ("10лет" -> "десять лет"):
        # glued to a neighbour, Silero swallows the number word.
        s, a, b = m.string, m.start(), m.end()
        left = " " if a > 0 and re.match(r"\w", s[a - 1]) else ""
        right = " " if m.group("hy") or (b < len(s) and re.match(r"\w", s[b])) else ""
        return left + words + right

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


def stress_dict_digest(stress_dict, text: "str | None" = None) -> str:
    """Short content hash of the stress dictionary entries that can affect `text` (all entries when
    text is None; phrase keys are always included). Part of the chapter fingerprint, so changing
    ONE stress mark (same number of entries) re-synthesizes exactly the chapters that use that
    word - and leaves the others alone."""
    items = dict(stress_dict or {})
    if text is not None:
        words = {w.lower() for w in _STRESS_WORD_RE.findall(text)}
        words |= {w.replace("ё", "е") for w in words}
        items = {k: v for k, v in items.items()
                 if " " in k or k in words or k.replace("ё", "е") in words}
    try:
        blob = json.dumps(sorted(items.items()), ensure_ascii=False)
    except Exception:
        blob = repr(items)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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
# The app is 100% local/free: speaker attribution and smart emotions need an LLM (local or paid
# cloud), so they are OFF in the active pipeline. Dialogue voices simply alternate per quote.
# (The LLM code stays in this file but is unreachable unless AUDIOBOOK_LLM_FEATURES=1.)
LLM_FEATURES_ENABLED = os.environ.get("AUDIOBOOK_LLM_FEATURES", "0") == "1"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"
LMSTUDIO_HOST_DEFAULT = "http://localhost:1234"

ATTRIBUTION_PROVIDERS = {
    "yandexgpt": {
        "title": "Yandex SpeechKit / AI Studio (YandexGPT, тот же ключ и каталог)",
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
# Встроенная локальная LLM (llm_rest_service.py в .venv_llm, ставит install.bat): работает полностью локально, поэтому
# разрешена и при LLM_FEATURES_ENABLED=0 (см. _llm_provider_allowed). Нужна для --smart-emotion --attribution-provider builtin.
ATTRIBUTION_PROVIDERS["builtin"] = {
    "title": "Встроенная локальная LLM (Qwen2.5-7B, без ключа и облака)",
    "default_model": "builtin",
    "key_hint": "Ключ не нужен: сервис llm_rest_service.py стартует сам (install.bat ставит .venv_llm и модель ~4.7 ГБ). "
                 "Нужна свободная видеопамять (~6 ГБ).",
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
            "temperature": 0.0,
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
            "generationConfig": {"temperature": 0.0},
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
            "completionOptions": {"stream": False, "temperature": 0.0, "maxTokens": "4000"},
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


def _ollama_chat_request(prompt: str, model: str, host: str = "", log_fn=None, timeout=180,
                          max_tokens: "int | None" = None) -> str:
    """Отправляет один запрос в локальный Ollama (/api/chat, без стриминга)
    и возвращает текст ответа. Общий HTTP-транспорт для всех трёх
    LLM-фич проекта, которые умеют работать через Ollama (атрибуция
    говорящих, умная эмоция, расстановка ударений-омографов) — так же, как
    attribute_speakers_{anthropic,gemini,yandexgpt} делят ANTHROPIC_MESSAGES_URL
    и т.п. Ошибки подключения (Ollama не запущена, не тот адрес) и ошибки
    модели (не скачана) оборачиваются в понятное сообщение."""
    import requests

    if not model:
        raise ValueError(
            "Не указана модель Ollama. Укажите имя заранее скачанной модели "
            "(например 'qwen2.5:7b-instruct', 'ollama pull qwen2.5:7b-instruct')."
        )
    host = (host or OLLAMA_HOST_DEFAULT).rstrip("/")
    options = {"temperature": 0.0}
    if max_tokens:
        options["num_predict"] = int(max_tokens)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Отключаем "рассуждение" у thinking-моделей (Qwen3 и т.п.) — оно
        # на порядок замедляет ответ, а нам нужен только короткий JSON.
        "think": False,
        "options": options,
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        if resp.status_code == 400 and "think" in resp.text.lower():
            payload.pop("think")  # старая версия Ollama / модель без поддержки
            resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama по адресу {host} ({e}). Проверьте, что Ollama "
            "запущена (команда 'ollama serve' или запущенное приложение Ollama) и адрес "
            "указан верно (по умолчанию http://localhost:11434)."
        )
    if resp.status_code >= 400:
        body = resp.text[:1000]
        if log_fn:
            log_fn(f"Ollama ({host}) ответила HTTP {resp.status_code}: {body}")
        raise RuntimeError(
            f"Ollama вернула ошибку HTTP {resp.status_code}: {body[:500]}\n"
            f"  Если ошибка про модель {model!r} — скачайте её: 'ollama pull {model}'."
        )
    data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama не вернула текст ответа (ответ: {str(data)[:500]})")
    return content


def _lmstudio_chat_request(prompt: str, model: str, host: str = "", log_fn=None, timeout=300,
                            max_tokens: "int | None" = None) -> str:
    """Один запрос к локальному серверу LM Studio (OpenAI-совместимый
    /v1/chat/completions) -> текст ответа. Аналог _ollama_chat_request."""
    import requests

    host = (host or LMSTUDIO_HOST_DEFAULT).rstrip("/")

    def _post(mt, extra):
        try:
            return requests.post(
                f"{host}/v1/chat/completions",
                json={
                    "model": model or "local-model",
                    # "/no_think" — штатный выключатель рассуждения у Qwen3-семейства.
                    "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
                    "temperature": 0.0,
                    "stream": False,
                    **({"max_tokens": int(mt)} if mt else {}),
                    **extra,
                },
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Не удалось подключиться к LM Studio по адресу {host} ({e}). Запустите локальный "
                "сервер в LM Studio (вкладка Developer -> Start Server) и проверьте адрес "
                "(по умолчанию http://localhost:1234)."
            )

    def _parse(resp):
        ch = resp.json()["choices"][0]
        msg = ch.get("message") or {}
        c = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.DOTALL).strip()
        return c, (msg.get("reasoning_content") or ""), ch.get("finish_reason")

    # Просим выключить рассуждение (неизвестные поля LM Studio игнорирует; если
    # сервер всё же вернёт 400 — повторяем без них).
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    resp = _post(max_tokens, extra)
    if resp.status_code == 400:
        extra = {}
        resp = _post(max_tokens, extra)
    if resp.status_code >= 400:
        body = resp.text[:1000]
        if log_fn:
            log_fn(f"LM Studio ({host}) ответил HTTP {resp.status_code}: {body}")
        raise RuntimeError(
            f"LM Studio вернул ошибку HTTP {resp.status_code}: {body[:500]}\n"
            f"  Проверьте, что модель {model!r} загружена, а идентификатор в поле 'Модель' "
            "совпадает с показанным в LM Studio."
        )
    try:
        content, reasoning, finish = _parse(resp)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise RuntimeError(f"LM Studio не вернул ожидаемый ответ (ответ: {resp.text[:500]})")
    if not content and reasoning:
        # Модель «думала» и исчерпала max_tokens: повторяем с большим бюджетом.
        if log_fn:
            log_fn("LM Studio: модель тратит токены на рассуждение (reasoning), пустой ответ; "
                   "повтор с увеличенным лимитом. Отключите 'Thinking' у модели в LM Studio "
                   "или возьмите non-thinking Instruct-сборку — так будет намного быстрее.")
        big = max((int(max_tokens) if max_tokens else 0) * 6, 4096)
        resp = _post(big, extra)
        if resp.status_code < 400:
            try:
                content, reasoning, finish = _parse(resp)
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                pass
        if not content and reasoning:
            # Последняя попытка: JSON иногда оказывается внутри рассуждения.
            m = re.findall(r"\[[^\[\]]*\]|\{[^{}]*\}", reasoning, flags=re.DOTALL)
            if m:
                content = m[-1]
    if not content:
        raise RuntimeError(
            "LM Studio вернул пустой ответ (модель в режиме рассуждения не уложилась в лимит). "
            "Выключите Thinking у модели в LM Studio или выберите non-thinking Instruct-модель."
        )
    return content


LOCAL_LLM_AUTOSTART = os.environ.get("AUDIOBOOK_LLM_AUTOSTART", "1") != "0"
LOCAL_LLM_CONTEXT = 8192      # атрибуции говорящих нужно 8192+
LOCAL_LLM_TTL_SECONDS = 900   # LM Studio сам выгрузит модель после простоя -> VRAM свободна для TTS
_local_llm_last_check: dict = {}


def _cli_path(name: str, extra: "list[str]") -> "str | None":
    import shutil
    found = shutil.which(name)
    if found:
        return found
    for c in extra:
        c = os.path.expandvars(os.path.expanduser(c))
        if os.path.isfile(c):
            return c
    return None


def _run_cli(args: "list[str]", timeout: int = 60) -> "tuple[int, str]":
    import subprocess
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, creationflags=flags)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, str(e)


def _http_alive(url: str, timeout: float = 2.0) -> "tuple[bool, str]":
    import requests
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500, r.text
    except Exception:
        return False, ""


def _wait_alive(url: str, seconds: float) -> bool:
    import time
    end = time.time() + seconds
    while time.time() < end:
        if _http_alive(url)[0]:
            return True
        time.sleep(1.0)
    return False


def ensure_local_llm(backend: str, model: str, host: str = "", log_fn=None, force: bool = False) -> bool:
    if not LLM_FEATURES_ENABLED:   # local/free mode: never probe :1234 / :11434, never run `lms`
        return False
    """Автоматически поднимает локальную LLM: запускает сервер LM Studio
    ('lms server start') / Ollama ('ollama serve') и загружает модель с
    контекстом LOCAL_LLM_CONTEXT и авто-выгрузкой через LOCAL_LLM_TTL_SECONDS.
    Не падает: при любой неудаче пишет подсказку и возвращает False
    (дальше всё работает как раньше). Проверка не чаще раза в 2 минуты."""
    import time
    log = log_fn or print
    host = (host or (LMSTUDIO_HOST_DEFAULT if backend == "lmstudio" else OLLAMA_HOST_DEFAULT)).rstrip("/")
    key = (backend, host, model)
    if not force and time.time() - _local_llm_last_check.get(key, 0) < 120:
        return True
    _local_llm_last_check[key] = time.time()
    try:
        if backend == "lmstudio":
            probe = f"{host}/v1/models"
            lms = _cli_path("lms", ["~/.lmstudio/bin/lms.exe", "~/.lmstudio/bin/lms",
                                    "%USERPROFILE%/.lmstudio/bin/lms.exe"])
            if not _http_alive(probe)[0]:
                if not lms:
                    log("LM Studio: сервер не отвечает, а утилита 'lms' не найдена — запустите "
                        "сервер вручную (Developer -> Start Server).")
                    return False
                log("LM Studio: запускаю локальный сервер (lms server start)...")
                _run_cli([lms, "server", "start"], timeout=90)
                if not _wait_alive(probe, 45):
                    log("LM Studio: сервер не поднялся за 45 сек — запустите его вручную.")
                    return False
            if lms and model:
                rc, out = _run_cli([lms, "ps"], timeout=30)
                if model.lower() not in out.lower():
                    log(f"LM Studio: загружаю модель {model} (контекст {LOCAL_LLM_CONTEXT})...")
                    rc, out = _run_cli([lms, "load", model, "--context-length", str(LOCAL_LLM_CONTEXT),
                                        "--ttl", str(LOCAL_LLM_TTL_SECONDS), "-y"], timeout=300)
                    if rc != 0:
                        rc, out = _run_cli([lms, "load", model, "-y"], timeout=300)
                    if rc != 0:
                        log(f"LM Studio: не удалось загрузить модель автоматически ({out.strip()[:300]}).")
                        return False
            return True
        # ollama
        probe = f"{host}/api/tags"
        if not _http_alive(probe)[0]:
            exe = _cli_path("ollama", ["%LOCALAPPDATA%/Programs/Ollama/ollama.exe"])
            if not exe:
                log("Ollama: сервер не отвечает и 'ollama' не найден — запустите приложение Ollama.")
                return False
            import subprocess
            log("Ollama: запускаю сервер (ollama serve)...")
            flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)) if os.name == "nt" else 0
            subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=flags)
            if not _wait_alive(probe, 45):
                log("Ollama: сервер не поднялся за 45 сек.")
                return False
        return True
    except Exception as e:
        log(f"Автозапуск локальной LLM не удался: {e}")
        return False


def _llm_provider_allowed(provider) -> bool:
    """LLM-функции выключены глобально (LLM_FEATURES_ENABLED=0), кроме ВСТРОЕННОЙ локальной модели."""
    return LLM_FEATURES_ENABLED or provider == "builtin"


def _builtin_chat_request(prompt: str, max_tokens: "int | None" = None) -> str:
    import requests
    import stress_llm
    stress_llm.ensure_builtin_service()
    r = requests.post(stress_llm.BUILTIN_URL + "/v1/chat/completions", timeout=900,
                      json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens or 300,
                            "temperature": 0.0})
    if r.status_code >= 400:
        raise RuntimeError(f"встроенная LLM: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _local_chat_request(prompt: str, model: str, host: str = "", backend: str = "ollama",
                         log_fn=None, max_tokens: "int | None" = None, api_key: str = "") -> str:
    """Единая точка для локальных LLM-серверов: 'ollama' или 'lmstudio'.
    Перед запросом автоматически поднимает сервер/модель (ensure_local_llm);
    при отказе соединения или 'модель не найдена' — один повтор после
    принудительного перезапуска."""
    if backend == "builtin":
        return _builtin_chat_request(prompt, max_tokens=max_tokens)

    def _once():
        if backend == "lmstudio":
            return _lmstudio_chat_request(prompt, model, host=host, log_fn=log_fn, max_tokens=max_tokens)
        return _ollama_chat_request(prompt, model, host=host, log_fn=log_fn, max_tokens=max_tokens)

    if not LLM_FEATURES_ENABLED:
        raise RuntimeError("Локальные LLM (LM Studio/Ollama) отключены: приложение работает полностью "
                           "локально без LLM (AUDIOBOOK_LLM_FEATURES=1 включает обратно).")
    if not LOCAL_LLM_AUTOSTART:
        return _once()
    ensure_local_llm(backend, model, host, log_fn)
    try:
        return _once()
    except RuntimeError as e:
        msg = str(e)
        if "подключиться" in msg or "model_not_found" in msg or "HTTP 400" in msg or "HTTP 404" in msg:
            if ensure_local_llm(backend, model, host, log_fn, force=True):
                return _once()
        raise


def attribute_speakers_ollama(text: str, model: str, host: str = "", log_fn=None,
                               backend: str = "ollama", api_key: str = "") -> list:
    """То же самое через локальный Ollama (ollama.com) — работает полностью
    офлайн/бесплатно на машине пользователя, но без строгого function
    calling у большинства моделей просим вернуть JSON текстом и парсим его
    сами, как и для YandexGPT (см. _extract_json_array)."""
    if _attribution_dialogue_count(text) == 0:
        return []

    # Локальные модели имеют небольшое окно контекста (часто 4–8К токенов), а глава
    # может быть длиннее — режем по абзацам на куски ~5000 символов.
    chunks, cur, cur_len = [], [], 0
    for para in text.split("\n"):
        if cur and cur_len + len(para) > 5000:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para) + 1
    if cur:
        chunks.append("\n".join(cur))

    all_results, failed = [], 0
    for idx, chunk in enumerate(chunks, 1):
        n = _attribution_dialogue_count(chunk)
        if n == 0:
            continue
        prompt = (
            _attribution_prompt(chunk, n) +
            "\n\nВерни ОТВЕТ СТРОГО в виде JSON-массива строк, без каких-либо пояснений, "
            "markdown-разметки или текста до/после — только сам массив, например: "
            '["Иван Петров", "unknown", "Мария"]'
        )
        if log_fn:
            log_fn(f"Атрибуция говорящих через {backend} ({model}): часть {idx}/{len(chunks)}, "
                   f"{n} реплик, {len(chunk)} симв....")
        try:
            raw_text = _local_chat_request(prompt, model, host=host, backend=backend, log_fn=log_fn,
                                            max_tokens=120 + 16 * n, api_key=api_key)
            res = _extract_json_array(raw_text)
            if res is None:
                raise RuntimeError(f"не удалось извлечь список из ответа: {raw_text[:200]}")
            all_results.extend(_normalize_attribution_results(res, n, log_fn))
        except Exception as e:
            failed += 1
            if log_fn:
                log_fn(f"Часть {idx}/{len(chunks)}: ошибка ({str(e)[:300]}) — реплики помечены 'unknown'.")
            all_results.extend(["unknown"] * n)
    if failed == len([c for c in chunks if _attribution_dialogue_count(c)]):
        raise RuntimeError("Модель не смогла разобрать ни одной части главы "
                           "(проверьте ключ/Folder ID Yandex и доступ к Yandex Cloud).")
    return all_results


def attribute_speakers_llm(text: str, api_key: str, model: str = DEFAULT_ATTRIBUTION_MODEL,
                            provider: str = DEFAULT_ATTRIBUTION_PROVIDER, log_fn=None,
                            folder_id: str = "", ollama_host: str = "") -> list:
    """Диспетчер: вызывает attribute_speakers_yandexgpt (по умолчанию,
    работает из РФ без VPN), attribute_speakers_gemini (бесплатно, но
    часто недоступен из РФ), attribute_speakers_anthropic (платно) или
    attribute_speakers_ollama (локально, бесплатно, нужен запущенный
    Ollama), в зависимости от provider — см. ATTRIBUTION_PROVIDERS."""
    if provider == "yandexgpt":
        return attribute_speakers_yandexgpt(text, api_key, model, folder_id=folder_id, log_fn=log_fn)
    if provider == "anthropic":
        return attribute_speakers_anthropic(text, api_key, model, log_fn=log_fn)
    if provider in ("ollama", "lmstudio"):
        return attribute_speakers_ollama(text, model, host=ollama_host, log_fn=log_fn, backend=provider,
                                          api_key=api_key)
    return attribute_speakers_gemini(text, api_key, model, log_fn=log_fn)


# --------------------------------------------------------------------------
# «Умная» эмоциональная окраска (режим yandex, флаг --smart-emotion):
# Yandex SpeechKit умеет менять эмоцию голоса ЗАПРОСОМ (параметр emotion),
# но только у двух голосов — см. YANDEX_EMOTION_VOICES — и раньше в этом
# проекте эмоция была ОДНА на всю озвучку (--yandex-emotion, из GUI/CLI).
# Ниже — LLM (тот же провайдер/ключ, что и для атрибуции говорящих выше)
# сама решает эмоцию ОТДЕЛЬНО ДЛЯ КАЖДОГО АБЗАЦА главы, по смыслу текста —
# грустный абзац звучит иначе, чем бодрый или зловещий, без ручной разметки.
# Используется ТЕКСТОВЫЙ ответ (не tool_use/function calling, как у
# attribute_speakers_*) — для короткого списка меток из ограниченного
# набора это надёжно работает у всех трёх провайдеров одинаково, и не
# нужно поддерживать три разных JSON-схемы инструментов.
# --------------------------------------------------------------------------

YANDEX_EMOTION_VOICES = {
    "ermil": ["good", "neutral"],
    "jane": ["good", "neutral", "evil"],
}

# «Умная эмоция» для silero_rest (--smart-emotion): в отличие от Yandex
# SpeechKit, у Silero нет отдельного параметра emotion — вместо него
# используется универсальный SSML <prosody rate="..." pitch="...">, который
# сервис уже умеет применять (см. _prosody_wrap/emphasize в text_to_ssml).
# Поэтому набор меток здесь свой, более широкий (не привязан к вокабуляру
# конкретного облачного API), и каждая метка сопоставлена с конкретными
# rate/pitch — подобраны на глаз, для тонкой (не карикатурной) окраски;
# при желании можно позже сделать настраиваемыми, но для начала это
# разумные дефолты.
SILERO_EMOTION_PROSODY = {
    "neutral":  {"rate": "100%", "pitch": "+0%"},
    "joy":      {"rate": "108%", "pitch": "+10%"},
    "sadness":  {"rate": "92%",  "pitch": "-8%"},
    "fear":     {"rate": "112%", "pitch": "+6%"},
    "anger":    {"rate": "106%", "pitch": "-5%"},
    "calm":     {"rate": "95%",  "pitch": "-3%"},
}


SILERO_EMOTION_INTENSITY = 0.6   # доля от табличного сдвига rate/pitch (мягче -> меньше «карикатуры»)
SILERO_EMOTION_SMOOTHING = 0.5   # вес текущего абзаца в скользящем среднем (меньше = плавнее)


def smooth_emotion_prosody(labels: "list[str]") -> "list[dict]":
    """Метки эмоций по абзацам -> список {"rate","pitch"} со сглаживанием:
    1) одиночный «выброс» между двумя одинаковыми соседями убирается;
    2) сдвиги rate/pitch ослабляются (SILERO_EMOTION_INTENSITY);
    3) скользящее среднее по абзацам — тон меняется плавно, а не скачком
    на границе абзаца."""
    lab = list(labels)
    for i in range(1, len(lab) - 1):
        if lab[i - 1] == lab[i + 1] and lab[i] != lab[i - 1]:
            lab[i] = lab[i - 1]
    out = []
    prev_r = prev_p = 0.0
    for l in lab:
        base = SILERO_EMOTION_PROSODY.get(l) or SILERO_EMOTION_PROSODY["neutral"]
        r = (float(base["rate"].rstrip("%")) - 100.0) * SILERO_EMOTION_INTENSITY
        pt = float(base["pitch"].rstrip("%")) * SILERO_EMOTION_INTENSITY
        a = SILERO_EMOTION_SMOOTHING
        prev_r = a * r + (1 - a) * prev_r
        prev_p = a * pt + (1 - a) * prev_p
        out.append({"rate": f"{100 + round(prev_r)}%", "pitch": f"{round(prev_p):+d}%"})
    return out


_EMOTION_LABEL_HINTS = {
    "good": "тёплая/бодрая/доброжелательная интонация",
    "neutral": "обычное повествование без выраженной окраски",
    "evil": "злая/угрожающая/зловещая интонация",
    "joy": "радость/приподнятое настроение",
    "sadness": "грусть/уныние",
    "fear": "страх/тревога/напряжение",
    "anger": "злость/раздражение",
    "calm": "спокойствие/умиротворение",
}


def _emotion_prompt(paragraphs: "list[str]", allowed_emotions: "list[str]", keyed: bool = False) -> str:
    labels = "/".join(allowed_emotions)
    hints = "; ".join(f"'{e}' — {_EMOTION_LABEL_HINTS[e]}" for e in allowed_emotions if e in _EMOTION_LABEL_HINTS)
    numbered = "\n".join(f"{i + 1}. {p[:400]}" for i, p in enumerate(paragraphs))
    return (
        "Ниже — пронумерованные абзацы одной главы книги на русском языке. Для КАЖДОГО "
        f"абзаца определи наиболее подходящую эмоциональную окраску речи из набора: {labels}.\n"
        f"Значения: {hints}. Опирайся на смысл и контекст абзаца (что происходит, "
        "какие эмоции у персонажей/рассказчика), а не на отдельные слова. Озвучка — "
        "аудиокнига: интонация НЕ должна скакать. Если окраска не выражена явно — ставь "
        "'neutral'; соседние абзацы одной сцены обычно имеют одну и ту же метку, меняй её "
        "только при явной смене настроения сцены.\n\n"
        + (
            f"Верни ОТВЕТ СТРОГО в виде JSON-объекта: ключ — номер абзаца (строкой), значение — "
            f"метка, ровно {len(paragraphs)} ключей, например "
            + str({str(i + 1): "neutral" for i in range(min(3, len(paragraphs)))}).replace("'", '"')
            + ", без пояснений, markdown-разметки или текста до/после объекта.\n\n"
            if keyed else
            f"Верни ОТВЕТ СТРОГО в виде JSON-массива строк длиной ровно {len(paragraphs)} — "
            "по одной метке на каждый абзац, в том же порядке, без каких-либо пояснений, "
            f"markdown-разметки или текста до/после массива, например: "
            + str(["neutral"] * min(3, len(paragraphs))) + "\n\n"
        )
        + "--- АБЗАЦЫ ---\n" + numbered
    )


def _normalize_emotion_results(results, count: int, allowed_emotions: "list[str]", log_fn=None) -> "list[str]":
    default = "neutral" if "neutral" in allowed_emotions else allowed_emotions[0]
    out = []
    for r in (results or []):
        r = str(r).strip().lower()
        out.append(r if r in allowed_emotions else default)
    if len(out) != count:
        if log_fn:
            log_fn(f"Внимание: LLM вернула {len(out)} эмоций вместо {count} абзацев — "
                   "выравниваю список (лишнее обрезаю/недостающее заполняю дефолтом).")
        if len(out) < count:
            out = out + [default] * (count - len(out))
        else:
            out = out[:count]
    return out


def classify_paragraph_emotions_anthropic(paragraphs, api_key, model, allowed_emotions, log_fn=None) -> "list[str]":
    import requests
    if not api_key:
        raise ValueError("Не указан API-ключ для определения эмоций через Anthropic.")
    prompt = _emotion_prompt(paragraphs, allowed_emotions)
    resp = requests.post(
        ANTHROPIC_MESSAGES_URL,
        headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_API_VERSION,
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 2048, "temperature": 0.0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic API вернул ошибку HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    raw_text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    results = _extract_json_array(raw_text)
    if results is None:
        raise RuntimeError(f"Anthropic не вернул распознаваемый JSON-массив эмоций (ответ: {raw_text[:500]})")
    return _normalize_emotion_results(results, len(paragraphs), allowed_emotions, log_fn)


def classify_paragraph_emotions_gemini(paragraphs, api_key, model, allowed_emotions, log_fn=None) -> "list[str]":
    import requests
    if not api_key:
        raise ValueError("Не указан API-ключ для определения эмоций через Gemini.")
    prompt = _emotion_prompt(paragraphs, allowed_emotions)
    resp = requests.post(
        GEMINI_GENERATE_URL_TMPL.format(model=model),
        params={"key": api_key}, headers={"content-type": "application/json"},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.0}},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini API вернул ошибку HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    raw_text = ""
    for cand in data.get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            raw_text += part.get("text", "")
    results = _extract_json_array(raw_text)
    if results is None:
        raise RuntimeError(f"Gemini не вернул распознаваемый JSON-массив эмоций (ответ: {raw_text[:500]})")
    return _normalize_emotion_results(results, len(paragraphs), allowed_emotions, log_fn)


def classify_paragraph_emotions_yandexgpt(paragraphs, api_key, model, allowed_emotions,
                                           folder_id: str = "", log_fn=None) -> "list[str]":
    import requests
    if not api_key:
        raise ValueError("Не указан API-ключ YandexGPT для определения эмоций.")
    if not folder_id:
        raise ValueError("Не указан Folder ID YandexGPT для определения эмоций.")
    prompt = _emotion_prompt(paragraphs, allowed_emotions)
    model_uri = f"gpt://{folder_id}/{model}"
    resp = requests.post(
        YANDEXGPT_COMPLETION_URL,
        headers={"Authorization": f"Api-Key {api_key}", "content-type": "application/json"},
        json={"modelUri": model_uri,
              "completionOptions": {"stream": False, "temperature": 0.0, "maxTokens": "2000"},
              "messages": [{"role": "user", "text": prompt}]},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"YandexGPT API вернул ошибку HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        raw_text = data["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"YandexGPT не вернул ожидаемый текст ответа (ответ: {str(data)[:500]})")
    results = _extract_json_array(raw_text)
    if results is None:
        raise RuntimeError(f"YandexGPT не вернул распознаваемый JSON-массив эмоций (ответ: {raw_text[:500]})")
    return _normalize_emotion_results(results, len(paragraphs), allowed_emotions, log_fn)


def classify_paragraph_emotions_ollama(paragraphs, model, allowed_emotions, host: str = "",
                                        log_fn=None, backend: str = "ollama", api_key: str = "") -> "list[str]":
    """То же самое через локальную LLM. Метки приходят ПО НОМЕРАМ абзацев
    ({"1": "joy", ...}), поэтому пропуск одного пункта не сдвигает все
    остальные метки на соседние абзацы (раньше: '14 вместо 15' -> эмоции
    уезжали и интонация «скакала»). Не пришедшие пункты перезапрашиваются
    (до 2 раз), остаток — 'neutral'."""
    default = "neutral" if "neutral" in allowed_emotions else allowed_emotions[0]
    labels = [None] * len(paragraphs)
    step = 15  # маленькие модели путают счёт на длинных списках — режем на пачки
    for start in range(0, len(paragraphs), step):
        idxs = list(range(start, min(start + step, len(paragraphs))))
        for attempt in range(3):
            if not idxs:
                break
            part = [paragraphs[i] for i in idxs]
            try:
                raw_text = _local_chat_request(_emotion_prompt(part, allowed_emotions, keyed=True),
                                                model, host=host, backend=backend, log_fn=log_fn,
                                                max_tokens=80 + 12 * len(part), api_key=api_key)
            except Exception as e:
                if log_fn:
                    log_fn(f"Эмоции, пачка {start // step + 1}: ошибка ({str(e)[:200]}) — по умолчанию.")
                break
            parsed = _parse_stress_reply(raw_text, len(part))
            for j, i in enumerate(idxs):
                v = (parsed.get(j) or "").strip().lower()
                if v in allowed_emotions:
                    labels[i] = v
            left = [i for i in idxs if labels[i] is None]
            if len(left) == len(idxs):
                break  # прогресса нет
            idxs = left
    missing = sum(1 for x in labels if x is None)
    if missing and log_fn:
        log_fn(f"Эмоции: для {missing} абзац(ев) метка не получена — 'neutral'.")
    return [x or default for x in labels]


def classify_paragraph_emotions_llm(paragraphs, allowed_emotions, api_key, model=DEFAULT_ATTRIBUTION_MODEL,
                                     provider=DEFAULT_ATTRIBUTION_PROVIDER, folder_id: str = "",
                                     ollama_host: str = "", log_fn=None) -> "list[str]":
    """Диспетчер, аналогичный attribute_speakers_llm — определяет эмоцию
    КАЖДОГО абзаца из paragraphs (список строк) из набора allowed_emotions
    (см. YANDEX_EMOTION_VOICES), тем же провайдером/ключом, что и атрибуция
    говорящих. Пустой список абзацев -> пустой результат без обращения к API."""
    if not paragraphs:
        return []
    if provider == "yandexgpt":
        return classify_paragraph_emotions_yandexgpt(paragraphs, api_key, model, allowed_emotions,
                                                       folder_id=folder_id, log_fn=log_fn)
    if provider == "anthropic":
        return classify_paragraph_emotions_anthropic(paragraphs, api_key, model, allowed_emotions, log_fn=log_fn)
    if provider in ("ollama", "lmstudio", "builtin"):
        return classify_paragraph_emotions_ollama(paragraphs, model, allowed_emotions,
                                                    host=ollama_host, log_fn=log_fn, backend=provider,
                                                    api_key=api_key)
    return classify_paragraph_emotions_gemini(paragraphs, api_key, model, allowed_emotions, log_fn=log_fn)


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


def _voice_gender(voice) -> "str | None":
    """'m' / 'f' / None (неизвестно) для ключа голоса — по пометке "(муж.)"/
    "(жен.)"/"(мужской)"/"(женский)" в подписях голосов Silero, Yandex,
    Qwen (облако/локально) и Piper. Для остальных (CosyVoice-профили,
    системные голоса offline) пол не известен -> None."""
    if not voice:
        return None
    for table in (SILERO_SPEAKERS, YANDEX_VOICES, QWEN_TTS_VOICES, QWEN_TTS_LOCAL_VOICES, PIPER_VOICES):
        label = table.get(voice)
        if not label:
            continue
        low = label.lower()
        if "жен" in low:
            return "f"
        if "муж" in low:
            return "m"
    return None


def _guess_name_gender(name: str) -> "str | None":
    """'m' / 'f' / None по имени персонажа — через pymorphy3 (необязательная
    зависимость, см. _get_pymorphy_analyzer): берётся первое слово имени,
    которое разбор признаёт именем/отчеством/фамилией с определённым
    родом. Без pymorphy3 или для непонятных имён ("он", "старик", "unknown")
    возвращает None — тогда голос выбирается как раньше, без учёта пола."""
    analyzer = _get_pymorphy_analyzer()
    if analyzer is None or not name:
        return None
    for token in re.findall(r"[А-Яа-яЁё]+", name):
        try:
            parses = analyzer.parse(token.lower())
        except Exception:
            continue
        for p in parses[:3]:
            tag = p.tag
            if not any(g in str(tag) for g in ("Name", "Patr", "Surn")):
                continue
            if "femn" in str(tag):
                return "f"
            if "masc" in str(tag):
                return "m"
    return None


def _llm_text_request(prompt: str, creds: dict, log_fn=None) -> str:
    """Один текстовый запрос к LLM-провайдеру из creds (тот же словарь, что
    attribution: api_key/model/provider/folder_id/ollama_host) -> сырой
    текст ответа. Общий транспорт для разовых вспомогательных вопросов
    (например, пол персонажа); те же HTTP-вызовы, что в classify_*."""
    import requests
    provider = creds.get("provider", DEFAULT_ATTRIBUTION_PROVIDER)
    model = creds.get("model", DEFAULT_ATTRIBUTION_MODEL)
    api_key = creds.get("api_key", "")
    if provider in ("ollama", "lmstudio"):
        return _local_chat_request(prompt, model, host=creds.get("ollama_host", ""),
                                    backend=provider, log_fn=log_fn, max_tokens=600, api_key=api_key)
    if provider == "anthropic":
        resp = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_API_VERSION,
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 1024, "temperature": 0.0, "messages": [{"role": "user", "content": prompt}]},
            timeout=120)
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
    if provider == "yandexgpt":
        resp = requests.post(
            YANDEXGPT_COMPLETION_URL,
            headers={"Authorization": f"Api-Key {api_key}", "content-type": "application/json"},
            json={"modelUri": f"gpt://{creds.get('folder_id', '')}/{model}",
                  "completionOptions": {"stream": False, "temperature": 0.0, "maxTokens": "1000"},
                  "messages": [{"role": "user", "text": prompt}]},
            timeout=120)
        resp.raise_for_status()
        return resp.json()["result"]["alternatives"][0]["message"]["text"]
    resp = requests.post(
        GEMINI_GENERATE_URL_TMPL.format(model=model), params={"key": api_key},
        headers={"content-type": "application/json"},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.0}}, timeout=120)
    resp.raise_for_status()
    out = ""
    for cand in resp.json().get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            out += part.get("text", "")
    return out


def detect_character_genders(names: "list[str]", text: str, creds: dict, log_fn=None) -> dict:
    """{имя_в_нижнем_регистре: 'm'|'f'} — LLM читает фрагменты главы, где
    упоминается персонаж, и определяет его пол (учитывает прозвища, роли
    вроде "старуха", "капитан", окончания глаголов прошедшего времени —
    то, что одним разбором имени через pymorphy не поймать). Только для
    имён, где LLM уверена; остальные не попадут в результат (тогда
    сработает запасной вариант — _guess_name_gender). При любой ошибке
    возвращает {}."""
    if not names or not creds:
        return {}
    blocks = []
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", text)
    for n in names:
        token = n.split()[0].lower() if n.split() else n.lower()
        snippets = [s.strip()[:220] for s in sentences if token[:5] in s.lower()][:3]
        blocks.append(f"- {n}: " + (" | ".join(snippets) if snippets else "(фрагментов нет)"))
    prompt = (
        "Ниже — персонажи книги на русском и фрагменты, где они упоминаются. Для КАЖДОГО "
        "определи пол: 'm' (мужской) или 'f' (женский), по имени, роли и согласованию "
        "слов в тексте; если нельзя понять — 'unknown'.\n\n"
        f"Верни ОТВЕТ СТРОГО в виде JSON-массива строк длиной ровно {len(names)}, в том же "
        "порядке, без пояснений и markdown, например: [\"m\", \"f\", \"unknown\"]\n\n"
        + "\n".join(blocks)
    )
    try:
        results = _extract_json_array(_llm_text_request(prompt, creds, log_fn=log_fn))
    except Exception as e:
        if log_fn:
            log_fn(f"Определение пола персонажей через LLM не удалось ({e}) — использую разбор имён.")
        return {}
    if not results or len(results) != len(names):
        return {}
    out = {}
    for n, g in zip(names, results):
        g = str(g).strip().lower()
        if g in ("m", "f"):
            out[n.strip().lower()] = g
    return out


def _pick_voice_for_new_character(name_key: str, dialogue_voices, character_voice_map: dict,
                                   name_genders: "dict | None" = None):
    """Выбор голоса для нового персонажа: сначала ещё НЕ занятые голоса
    подходящего пола (пол — от LLM из name_genders, иначе по разбору имени),
    затем любой голос этого пола, затем (если пол неизвестен или таких
    голосов нет) старое поведение — следующий по кругу голос из списка."""
    gender = (name_genders or {}).get(name_key) or _guess_name_gender(name_key)
    if gender:
        matching = [v for v in dialogue_voices if _voice_gender(v) == gender]
        if matching:
            used = set(character_voice_map.values())
            free = [v for v in matching if v not in used]
            if free:
                return free[0]
            n_same = sum(1 for v in character_voice_map.values() if v in matching)
            return matching[n_same % len(matching)]
    return dialogue_voices[len(character_voice_map) % len(dialogue_voices)]


def _group_paragraphs_by_speaker_names(text: str, main_voice, dialogue_voices,
                                        speaker_names: list, character_voice_map: dict,
                                        name_genders: "dict | None" = None):
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
    hist = []  # голоса последних реплик
    for para, is_dialogue in paras:
        if is_dialogue:
            name = speaker_names[di] if di < len(speaker_names) else "unknown"
            di += 1
            name_key = name.strip().lower()
            if not name_key or name_key == "unknown":
                if len(hist) >= 2:
                    # диалог обычно идёт «пинг-понгом»: неопознанная реплика —
                    # скорее всего от собеседника предыдущего говорящего
                    voice = hist[-2]
                else:
                    voice = dialogue_voices[unknown_i % len(dialogue_voices)]
                    unknown_i += 1
            elif name_key in character_voice_map:
                voice = character_voice_map[name_key]
            else:
                voice = _pick_voice_for_new_character(name_key, dialogue_voices, character_voice_map, name_genders)
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
    if not attribution or not LLM_FEATURES_ENABLED:
        return _group_paragraphs_by_voice(text, main_voice, dialogue_voices)

    try:
        speaker_names = attribute_speakers_llm(
            text, attribution.get("api_key", ""), attribution.get("model", DEFAULT_ATTRIBUTION_MODEL),
            provider=attribution.get("provider", DEFAULT_ATTRIBUTION_PROVIDER),
            folder_id=attribution.get("folder_id", ""),
            ollama_host=attribution.get("ollama_host", ""),
            log_fn=log_fn,
        )
    except Exception as e:
        if log_fn:
            log_fn(f"Атрибуция говорящих не удалась ({e}) — использую обычное чередование голосов.")
        return _group_paragraphs_by_voice(text, main_voice, dialogue_voices)

    character_voice_map = _load_character_voice_map(outdir)
    # Пол НОВЫХ (ещё без закреплённого голоса) персонажей определяет та же
    # LLM, что делала атрибуцию — один дополнительный запрос на главу.
    # Пол определяем для ВСЕХ персонажей главы (а не только новых): закреплённые
    # раньше голоса могли быть выбраны без учёта пола (мужчина -> женский голос).
    all_names = []
    for n in speaker_names:
        k = n.strip().lower()
        if k and k != "unknown" and n.strip() not in all_names:
            all_names.append(n.strip())
    name_genders = detect_character_genders(all_names, text, attribution, log_fn) if all_names else {}
    for k in list(character_voice_map.keys()):
        g = name_genders.get(k) or _guess_name_gender(k)
        vg = _voice_gender(character_voice_map[k])
        if g and vg and g != vg and any(_voice_gender(v) == g for v in dialogue_voices):
            if log_fn:
                log_fn(f"Персонаж «{k}»: голос {character_voice_map[k]} не подходит по полу "
                       f"({'жен.' if g == 'f' else 'муж.'}) — подбираю другой.")
            del character_voice_map[k]
    groups = _group_paragraphs_by_speaker_names(
        text, main_voice, dialogue_voices, speaker_names, character_voice_map, name_genders
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
               dialogue_voices=None, attribution=None, should_stop=None, play_fn=None,
               smart_emotion: bool = False, emotion_attribution=None):
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
    Остальной текст (авторская речь) по-прежнему звучит голосом voice.

    smart_emotion — вместо ОДНОЙ эмоции (emotion) на всю озвучку, LLM сама
    определяет эмоцию для КАЖДОГО абзаца главы по смыслу текста (см.
    classify_paragraph_emotions_llm). Работает только для голосов из
    YANDEX_EMOTION_VOICES (ermil, jane) — для остальных голосов Yandex
    SpeechKit эмоции не поддерживает в принципе, флаг тихо игнорируется
    (с предупреждением в лог). При любой ошибке LLM откатывается на
    единую emotion, как раньше, не срывая всю озвучку главы.

    emotion_attribution — {"api_key":..., "model":..., "provider":...,
    "folder_id":...} для LLM-вызова smart_emotion; по умолчанию (None)
    используется тот же attribution, что и для атрибуции говорящих. Задан
    ОТДЕЛЬНО от attribution намеренно: smart_emotion не должен незаметно
    включать LLM-атрибуцию говорящих (resolve_voice_groups) там, где
    пользователь её не просил — GUI подставляет сюда ключ независимо от
    галочки «Определять, какой персонаж говорит»."""
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
            emotion=emotion, format=audio_format, smart_emotion=smart_emotion,
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

        # smart_emotion работает только для голосов с поддержкой эмоций и
        # только если хотя бы одна голосовая группа этой главы говорит
        # именно таким голосом — иначе (обычный voice без dialogue_voices,
        # либо диалоговые голоса без ermil/jane) ведём себя как раньше.
        emotion_voices_in_use = {g_voice for g_voice, _ in voice_groups} & set(YANDEX_EMOTION_VOICES)
        para_emotion_map: "dict[str, str] | None" = None
        if smart_emotion and LLM_FEATURES_ENABLED and emotion_voices_in_use:
            all_paras = [p for g_text in (t for _v, t in voice_groups) for p in g_text.split("\n") if p.strip()]
            # Один общий набор допустимых эмоций на главу — пересечение
            # поддерживаемых наборов используемых голосов (чтобы не
            # присвоить 'evil' абзацу, который в итоге озвучит ermil, не
            # умеющий в evil); если голосов с эмоциями несколько и наборы
            # не пересекаются — берём объединение и подрезаем при подстановке.
            allowed = sorted(set().union(*(YANDEX_EMOTION_VOICES[v] for v in emotion_voices_in_use)))
            _emo_creds = emotion_attribution if emotion_attribution is not None else attribution
            try:
                para_emotion_list = classify_paragraph_emotions_llm(
                    all_paras, allowed_emotions=allowed,
                    api_key=(_emo_creds or {}).get("api_key", ""),
                    model=(_emo_creds or {}).get("model", DEFAULT_ATTRIBUTION_MODEL),
                    provider=(_emo_creds or {}).get("provider", DEFAULT_ATTRIBUTION_PROVIDER),
                    folder_id=(_emo_creds or {}).get("folder_id", ""),
                    ollama_host=(_emo_creds or {}).get("ollama_host", ""),
                    log_fn=log,
                )
                para_emotion_map = dict(zip(all_paras, para_emotion_list))
                log(f"smart_emotion: определены эмоции для {len(all_paras)} абзац(ев) главы.")
            except Exception as e:
                log(f"smart_emotion не удался ({e}) — использую единую emotion={emotion or 'neutral'} "
                    "для всей главы, как без этого флага.")
                para_emotion_map = None

        if para_emotion_map is not None:
            # Отдельный путь синтеза: КАЖДЫЙ абзац — отдельный запрос к
            # Yandex SpeechKit со своей emotion (обычный путь ниже пакует
            # несколько абзацев в один запрос до YANDEX_MAX_CHARS — так
            # быстрее, но тогда общая emotion была бы одна на весь пакет).
            segments = []
            for g_voice, g_text in voice_groups:
                for para in g_text.split("\n"):
                    if not para.strip():
                        continue
                    seg_emotion = (
                        para_emotion_map.get(para, emotion) if g_voice in YANDEX_EMOTION_VOICES else ""
                    )
                    if seg_emotion and seg_emotion not in YANDEX_EMOTION_VOICES.get(g_voice, []):
                        seg_emotion = "neutral" if "neutral" in YANDEX_EMOTION_VOICES.get(g_voice, []) else ""
                    for sub in (split_text(para, YANDEX_MAX_CHARS) if len(para) > YANDEX_MAX_CHARS else [para]):
                        segments.append((sub, g_voice, seg_emotion))
            chunks_total = len(segments) or 1
            if on_progress:
                on_progress(pos, total, 0, chunks_total)
            chunk_bytes = []
            for chunk_no, (chunk, seg_voice, seg_emotion) in enumerate(
                tqdm(segments, desc=f"Гл.{idx}", unit="абзац"), 1
            ):
                if chunk.strip() and _text_has_speakable_content(chunk):
                    seg_lang = yandex_lang_for_voice(seg_voice)
                    data = synth_yandex_chunk(
                        chunk, api_key, folder_id, seg_voice, seg_lang, speed, seg_emotion, audio_format,
                        log_fn=log,
                    )
                    chunk_bytes.append(data)
                if on_progress:
                    on_progress(pos, total, chunk_no, chunks_total)
        else:
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


def _detect_audio_format(data: bytes) -> str:
    """Определяет формат аудио по первым байтам — используется для
    Qwen3-TTS, чей формат ответа не был проверен вживую (см. QWEN_TTS_*
    выше), чтобы не хардкодить расширение файла наугад."""
    if data[:4] == b"RIFF":
        return "wav"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    return "wav"


def synth_qwen_tts_chunk(text: str, api_key: str, voice: str, language: str = QWEN_TTS_LANGUAGE,
                          log_fn=None) -> bytes:
    """Один запрос к Qwen3-TTS через DashScope (модель qwen3-tts-flash).
    Возвращает содержимое аудио. См. предупреждение у QWEN_TTS_VOICES выше —
    это писалось по документации, без возможности проверить вживую в
    песочнице; если формат ответа окажется иным — исключение ниже покажет
    сырой ответ целиком, что нужно поправить.

    log_fn(message), если передан — как у synth_yandex_chunk, для диагностики."""
    if not api_key:
        raise ValueError(
            "Не указан API-ключ DashScope (переменная окружения DASHSCOPE_API_KEY "
            "или поле в GUI). Получить ключ можно в консоли Alibaba Cloud Model "
            "Studio (https://www.alibabacloud.com/help/en/model-studio/)."
        )
    try:
        import dashscope
    except ImportError as e:
        raise RuntimeError(
            "Пакет dashscope не установлен — нужен для режима Qwen3-TTS "
            "(pip install dashscope)."
        ) from e

    if log_fn:
        preview = f"[{len(text)} симв.] {text[:80]!r}…" if len(text) > 80 else text
        log_fn(f"Запрос к Qwen3-TTS (DashScope): model={QWEN_TTS_MODEL}, voice={voice}, "
               f"language={language}, text={preview}")

    try:
        response = dashscope.MultiModalConversation.call(
            model=QWEN_TTS_MODEL,
            api_key=api_key,
            text=text,
            voice=voice,
            language_type=language,
            stream=False,
        )
    except Exception as e:
        if log_fn:
            log_fn(f"Ошибка обращения к DashScope: {e}")
        raise RuntimeError(f"Ошибка при обращении к Qwen3-TTS (DashScope): {e}") from e

    status_code = getattr(response, "status_code", None)
    if status_code is not None and status_code != 200:
        code = getattr(response, "code", "")
        message = getattr(response, "message", "")
        if log_fn:
            log_fn(f"Qwen3-TTS (DashScope) вернул ошибку: HTTP {status_code} {code} {message}")
        raise RuntimeError(
            f"Qwen3-TTS (DashScope) вернул ошибку: HTTP {status_code} {code}: {message}\n"
            "  Частые причины: неверный/просроченный API-ключ, не подключён "
            "платёжный аккаунт, текст длиннее лимита в 512 токенов на запрос. "
            "Проверить ключ и баланс: https://www.alibabacloud.com/help/en/model-studio/"
        )

    audio_url = None
    try:
        audio_url = response.output.audio["url"]
    except Exception:
        try:
            audio_url = response.output.audio.url
        except Exception:
            pass
    if not audio_url:
        if log_fn:
            log_fn(f"Не удалось найти ссылку на аудио в ответе Qwen3-TTS. Полный ответ: {response}")
        raise RuntimeError(
            "Qwen3-TTS (DashScope) ответил, но не удалось найти в ответе ссылку на "
            f"аудио — формат ответа отличается от ожидаемого. Полный ответ (см. лог): {response}"
        )

    import urllib.request
    with urllib.request.urlopen(audio_url, timeout=60) as resp:
        return resp.read()


def run_qwen_tts(chapters, outdir: Path, start: int, play: bool, api_key: str,
                  voice: str = QWEN_TTS_DEFAULT_VOICE, language: str = QWEN_TTS_LANGUAGE,
                  on_progress=None, chapter_indices=None, char_ranges=None,
                  dialogue_voices=None, attribution=None, should_stop=None, play_fn=None):
    """Озвучка через облачный Qwen3-TTS (DashScope, модель qwen3-tts-flash).
    См. предупреждение у QWEN_TTS_VOICES/synth_qwen_tts_chunk — интеграция
    написана по документации и не проверена на живом аккаунте.

    Текст режется на куски по QWEN_TTS_MAX_CHARS (осторожная оценка лимита
    в 512 токенов на запрос — см. константу) и склеивается через pydub,
    если он установлен, иначе сохраняется частями, как у Yandex.

    dialogue_voices — необязательный список голосов (ключи QWEN_TTS_VOICES)
    для чередования на репликах прямой речи, как в run_yandex."""
    import time as _time

    outdir.mkdir(parents=True, exist_ok=True)
    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    log_path = outdir / "qwen_tts_client.log"

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

        fingerprint = _params_fingerprint(
            text, mode="qwen_tts", voice=voice, language=language,
            dialogue_voices=",".join(dialogue_voices) if dialogue_voices else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        # Расширение файла заранее не известно (см. _detect_audio_format) —
        # проверяем "уже готово" по .wav и .mp3 сразу, на случай если формат
        # ответа отличался между запусками.
        existing_path = None
        for ext in ("wav", "mp3"):
            candidate = outdir / f"{idx:03d}_{sanitize_filename(title)}.{ext}"
            if _is_already_done(candidate, fingerprint):
                existing_path = candidate
                break
        if existing_path is not None:
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено): {title} -> {existing_path.name}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(existing_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (Qwen3-TTS): {title}")

        voice_groups = resolve_voice_groups(text, voice, dialogue_voices, outdir,
                                             attribution=attribution, log_fn=log)
        segments = _chunk_voice_groups(voice_groups, QWEN_TTS_MAX_CHARS)
        chunks_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, chunks_total)

        chunk_bytes = []
        for chunk_no, (chunk, seg_voice) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if chunk.strip() and _text_has_speakable_content(chunk):
                data = synth_qwen_tts_chunk(chunk, api_key, seg_voice, language, log_fn=log)
                chunk_bytes.append(data)
            if on_progress:
                on_progress(pos, total, chunk_no, chunks_total)

        if not chunk_bytes:
            continue

        audio_format = _detect_audio_format(chunk_bytes[0])
        out_path = outdir / f"{idx:03d}_{sanitize_filename(title)}.{audio_format}"

        if len(chunk_bytes) == 1:
            out_path.write_bytes(chunk_bytes[0])
        else:
            try:
                from pydub import AudioSegment
                import io as _io
                combined = AudioSegment.empty()
                for data in chunk_bytes:
                    combined += AudioSegment.from_file(_io.BytesIO(data), format=audio_format)
                combined.export(str(out_path), format=audio_format)
            except ImportError:
                for i, data in enumerate(chunk_bytes):
                    part_path = out_path.with_name(f"{out_path.stem}_part{i + 1}.{audio_format}")
                    part_path.write_bytes(data)
                print("  (pydub не установлен — глава сохранена частями *_partN)")

        _save_fingerprint(out_path, fingerprint)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


QWEN_TTS_LOCAL_DEFAULT_URL = "http://127.0.0.1:7860"
# Сигнатура подтверждена вживую по странице <сервер>/?view=api конкретной
# инсталляции Qwen3-TTS-Pinokio (SUP3RMASSIVE/Qwen3-TTS-Pinokio) — вкладка
# "Custom Voice" её демо использует функцию /generate_custom_voice с
# именованными параметрами text/language/speaker/instruct/model_size/seed
# и возвращает (аудиофайл, статус-строка). Список голосов и языков у этого
# сервера СВОЙ (не совпадает со списком голосов облачного DashScope) — см.
# QWEN_TTS_LOCAL_VOICES ниже. Если у кого-то стоит другая сборка/версия
# Gradio-демо с другой сигнатурой — сверьте на <URL сервера>/?view=api и
# поправьте константы ниже.
QWEN_TTS_LOCAL_API_NAME = "/generate_custom_voice"
QWEN_TTS_LOCAL_VOICES = {
    "Aiden": "Aiden (муж.)", "Dylan": "Dylan (муж.)", "Eric": "Eric (муж.)",
    "Ono_anna": "Ono_anna (жен.)", "Ryan": "Ryan (муж.)", "Serena": "Serena (жен.)",
    "Sohee": "Sohee (жен.)", "Uncle_fu": "Uncle_fu (муж.)", "Vivian": "Vivian (жен.)",
}
QWEN_TTS_LOCAL_DEFAULT_VOICE = "Serena"
QWEN_TTS_LOCAL_MODEL_SIZE = "1.7B"
# seed=-1 (случайный) на каждый кусок текста — вот откуда "скачущие"
# интонации и как будто немного другой голос от куска к куску: модель
# каждый раз генерирует заново со случайным сидом. Фиксированный сид даёт
# стабильный тембр/интонацию у одного голоса между кусками.
QWEN_TTS_LOCAL_SEED = 42
# Куски покрупнее (относительно облачных 350 симв.) уменьшают число "швов",
# но модель Qwen3-TTS сама умеет уверенно озвучить лишь ограниченный кусок
# за один вызов — при 1800 симв. часть текста в куске обрывалась (отсюда
# "озвучил не всю главу"). 500 — компромисс, при проблемах уменьшайте ещё.
QWEN_TTS_LOCAL_MAX_CHARS = 500
# Пустая инструкция "instruct" оставляет модели полную свободу в выборе
# интонации — отсюда "слишком эмоционально". Нейтральная инструкция по
# умолчанию слегка сдерживает экспрессию; при желании отредактируйте.
QWEN_TTS_LOCAL_INSTRUCT = "Спокойное, ровное повествование, без лишней экспрессии и театральности."


def synth_qwen_tts_local_chunk(text: str, server_url: str, voice: str,
                                language: str = QWEN_TTS_LANGUAGE,
                                api_name: str = QWEN_TTS_LOCAL_API_NAME,
                                model_size: str = QWEN_TTS_LOCAL_MODEL_SIZE,
                                seed: int = QWEN_TTS_LOCAL_SEED,
                                instruct: str = QWEN_TTS_LOCAL_INSTRUCT,
                                log_fn=None) -> bytes:
    """Один запрос к локально запущенному Gradio-серверу Qwen3-TTS (без
    DashScope/облака/карты — модель считается на видеокарте пользователя).
    См. предупреждение у QWEN_TTS_LOCAL_API_NAME выше — сигнатура снята с
    живого сервера (Qwen3-TTS-Pinokio), но у другой сборки может отличаться."""
    try:
        from gradio_client import Client
    except ImportError as e:
        raise RuntimeError(
            "Пакет gradio_client не установлен — нужен для режима "
            "«Qwen3-TTS (локально)» (pip install gradio_client)."
        ) from e

    if log_fn:
        preview = f"[{len(text)} симв.] {text[:80]!r}…" if len(text) > 80 else text
        log_fn(f"Запрос к локальному Qwen3-TTS ({server_url}): speaker={voice}, "
               f"language={language}, text={preview}")

    try:
        client = Client(server_url)
        result = client.predict(
            text=text, language=language, speaker=voice, instruct=instruct,
            model_size=model_size, seed=seed, api_name=api_name,
        )
    except Exception as e:
        if log_fn:
            log_fn(f"Ошибка обращения к локальному серверу Qwen3-TTS: {e}")
        raise RuntimeError(
            f"Ошибка при обращении к локальному серверу Qwen3-TTS ({server_url}): {e}\n"
            "  Проверьте, что сервер запущен и отвечает по этому адресу, и что "
            "имя функции/порядок параметров совпадает с тем, что показано на "
            f"{server_url.rstrip('/')}/?view=api — если нет, пришлите мне эту "
            "страницу, поправлю вызов."
        ) from e

    # /generate_custom_voice возвращает (аудио, статус) — аудио у
    # gradio_client приходит либо готовым путём к скачанному файлу (str),
    # либо словарём FileData с ключом "path" — на всякий случай понимаем оба.
    audio_part = result[0] if isinstance(result, (list, tuple)) else result
    audio_path = audio_part.get("path") if isinstance(audio_part, dict) else audio_part
    if not audio_path or not os.path.exists(str(audio_path)):
        if log_fn:
            log_fn(f"Неожиданный ответ локального сервера Qwen3-TTS: {result!r}")
        raise RuntimeError(
            f"Локальный сервер Qwen3-TTS вернул неожиданный ответ (не путь к "
            f"файлу): {result!r}. Проверьте {server_url.rstrip('/')}/?view=api "
            "и пришлите мне точную сигнатуру, если она отличается."
        )
    with open(audio_path, "rb") as f:
        return f.read()


def run_qwen_tts_local(chapters, outdir: Path, start: int, play: bool, server_url: str,
                        voice: str = QWEN_TTS_LOCAL_DEFAULT_VOICE, language: str = QWEN_TTS_LANGUAGE,
                        on_progress=None, chapter_indices=None, char_ranges=None,
                        dialogue_voices=None, attribution=None, should_stop=None, play_fn=None):
    """Озвучка через локальный Gradio-сервер Qwen3-TTS (без облака/карты —
    структура идентична run_qwen_tts, только вызов синтеза другой, см.
    synth_qwen_tts_local_chunk)."""
    import time as _time

    outdir.mkdir(parents=True, exist_ok=True)
    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    log_path = outdir / "qwen_tts_local_client.log"

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

        fingerprint = _params_fingerprint(
            text, mode="qwen_tts_local", voice=voice, language=language,
            dialogue_voices=",".join(dialogue_voices) if dialogue_voices else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
        )
        existing_path = None
        for ext in ("wav", "mp3"):
            candidate = outdir / f"{idx:03d}_{sanitize_filename(title)}.{ext}"
            if _is_already_done(candidate, fingerprint):
                existing_path = candidate
                break
        if existing_path is not None:
            print(f"[{idx}/{len(chapters)}] Пропускаю (уже озвучено): {title} -> {existing_path.name}")
            if play:
                print("  Проигрывание...")
                (play_fn or play_file)(existing_path)
            if on_progress:
                on_progress(pos, total, 1, 1)
            continue

        print(f"[{idx}/{len(chapters)}] Озвучиваю (Qwen3-TTS, локально): {title}")

        voice_groups = resolve_voice_groups(text, voice, dialogue_voices, outdir,
                                             attribution=attribution, log_fn=log)
        segments = _chunk_voice_groups(voice_groups, QWEN_TTS_LOCAL_MAX_CHARS)
        chunks_total = len(segments) or 1
        if on_progress:
            on_progress(pos, total, 0, chunks_total)

        chunk_bytes = []
        for chunk_no, (chunk, seg_voice) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if chunk.strip() and _text_has_speakable_content(chunk):
                data = synth_qwen_tts_local_chunk(chunk, server_url, seg_voice, language, log_fn=log)
                chunk_bytes.append(data)
            if on_progress:
                on_progress(pos, total, chunk_no, chunks_total)

        if not chunk_bytes:
            continue

        audio_format = _detect_audio_format(chunk_bytes[0])
        out_path = outdir / f"{idx:03d}_{sanitize_filename(title)}.{audio_format}"

        if len(chunk_bytes) == 1:
            out_path.write_bytes(chunk_bytes[0])
        else:
            try:
                from pydub import AudioSegment
                import io as _io
                combined = AudioSegment.empty()
                for data in chunk_bytes:
                    combined += AudioSegment.from_file(_io.BytesIO(data), format=audio_format)
                combined.export(str(out_path), format=audio_format)
            except ImportError:
                for i, data in enumerate(chunk_bytes):
                    part_path = out_path.with_name(f"{out_path.stem}_part{i + 1}.{audio_format}")
                    part_path.write_bytes(data)
                print("  (pydub не установлен — глава сохранена частями *_partN)")

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


# --------------------------------------------------------------------------
# Ударения для локального режима silero: словарь книжных имён/терминов +
# silero-stress как общая подстраховка. silero_rest_service.py (отдельный REST-
# сервис) уже делает то же самое через silero-stress на сервере — здесь то же
# самое добавляется для ЛОКАЛЬНОГО режима (--mode silero, апелляция к
# model.apply_tts напрямую), у которого раньше не было ничего, кроме
# встроенного в саму модель Silero угадывания ударений (put_accent=True),
# которое на редких/составных словах и омографах иногда ошибается.
# --------------------------------------------------------------------------

DEFAULT_STRESS_DICT_PATH = _app_paths.user_file("stress_dictionary.json")

# Захватывает и дефисные составные слова целиком ("во-первых", "кто-то") —
# иначе они виделись бы словарю как два отдельных слова ("во" + "первых"),
# и нельзя было бы задать ударение именно для сочетания целиком.
_STRESS_WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


# --------------------------------------------------------------------------
# Слова с ударением, зависящим от контекста (падежа/числа/части речи или
# просто разных слов с одинаковым написанием — омографы вроде "за́мок"/
# "замо́к", а также словоформы вроде "ка́зни"/"казни́", "ка́тера"/"катера́"):
# для НИХ словарь в принципе не может дать один правильный ответ на все
# случаи — единственный, кто в состоянии выбрать вариант по контексту
# предложения, это silero-stress. Список — почти 1200 слов из открытого списка
# омографов проекта Silero (https://github.com/snakers4/silero-stress,
# wiki Homograph-List) — хранится рядом со скриптом в
# ambiguous_stress_words_ru.txt, по одному слову в нижнем регистре на
# строку. Используется и на входе (load_stress_dictionary отфильтровывает
# такие слова, если они вдруг оказались в stress_dictionary.json — вручную
# добавлены или остались от старого автопополнения), и при автопоиске
# (enrich_stress_dictionary_online такие слова даже не пытается искать).
# --------------------------------------------------------------------------

DEFAULT_AMBIGUOUS_STRESS_WORDS_PATH = _app_paths.resolve_asset("ambiguous_stress_words_ru.txt")

_ambiguous_stress_words: "set | None" = None
_ambiguous_stress_words_load_failed = False


def _load_ambiguous_stress_words(path: "Path | str | None" = None) -> set:
    """Лениво загружает и кэширует список контекстно-зависимых слов (см.
    выше). Файл необязателен: если его нет — просто ничего не
    фильтруется, работает как раньше. Ошибка чтения тоже не прерывает
    синтез — только предупреждение один раз."""
    global _ambiguous_stress_words, _ambiguous_stress_words_load_failed
    if _ambiguous_stress_words is not None:
        return _ambiguous_stress_words
    if _ambiguous_stress_words_load_failed:
        return set()
    p = Path(path) if path else DEFAULT_AMBIGUOUS_STRESS_WORDS_PATH
    if not p.exists():
        _ambiguous_stress_words_load_failed = True
        return set()
    try:
        raw_words = {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        # Слово с одной гласной буквой (или вовсе без гласных) физически не
        # может иметь неоднозначное ударение — ударить там попросту больше
        # некуда, кроме этой единственной гласной. Список омографов взят из
        # чужого источника (см. комментарий выше) и в нём попадаются такие
        # ложные срабатывания (однослоговые слова) — здесь их отфильтровываем,
        # чтобы они не блокировали спокойно правильные слова в
        # stress_dictionary.json и не засоряли список "сомнительных" в
        # диалоге проверки ударений GUI.
        _RU_VOWELS = set("аеёиоуыэюя")

        def _vowel_count(w: str) -> int:
            return sum(1 for ch in w if ch in _RU_VOWELS)

        words = {w for w in raw_words if _vowel_count(w) >= 2}
        dropped = len(raw_words) - len(words)
        if dropped:
            print(f"Список контекстно-зависимых слов {p}: пропущено {dropped} "
                  f"слов(о) с одной гласной — там ударение однозначно, путать нечего.")

        # Дополнительно вычитаем слова, которые пользователь (или я по его
        # просьбе) явно проверил и признал НЕ омографом на самом деле — см.
        # not_ambiguous_stress_words_ru.txt / mark_word_not_ambiguous. Список
        # взят из внешнего источника (вики-страница Silero) и, помимо
        # однослоговых слов (см. выше), в нём попадаются слова, у которых
        # теоретически есть другое написание/разбор, но на практике спутать
        # почти невозможно, — в отличие от фильтра по гласным, автоматически
        # доказать это для конкретного слова нельзя, поэтому такие слова
        # исключаются только по одному, по факту проверки, а не общим
        # правилом.
        not_ambiguous = _load_not_ambiguous_stress_words()
        if not_ambiguous:
            before = len(words)
            words -= not_ambiguous
            if len(words) != before:
                print(f"Список контекстно-зависимых слов {p}: дополнительно исключено "
                      f"{before - len(words)} слов(о), отмеченных вручную как "
                      f"НЕ омографы (not_ambiguous_stress_words_ru.txt).")

        _ambiguous_stress_words = words
        return words
    except Exception as e:
        _ambiguous_stress_words_load_failed = True
        print(f"Не удалось загрузить список контекстно-зависимых слов {p}: {e} — "
              "продолжаю без этого фильтра.")
        return set()


DEFAULT_NOT_AMBIGUOUS_STRESS_WORDS_PATH = _app_paths.resolve_asset("not_ambiguous_stress_words_ru.txt")

_not_ambiguous_stress_words: "set | None" = None


def _load_not_ambiguous_stress_words(path: "Path | str | None" = None) -> set:
    """Слова, которые пользователь явно отметил как НЕ омографы (кнопка «Это
    не омограф» в диалоге проверки ударений GUI) — файл может отсутствовать,
    тогда просто ничего не вычитается. Не кэшируется так же агрессивно, как
    _load_ambiguous_stress_words (её саму читаем заново при каждом вызове
    _load_ambiguous_stress_words благодаря отсутствию собственного кэша
    здесь), потому что список может пополняться прямо во время работы GUI."""
    p = Path(path) if path else DEFAULT_NOT_AMBIGUOUS_STRESS_WORDS_PATH
    if not p.exists():
        return set()
    try:
        return {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    except Exception:
        return set()


def mark_word_not_ambiguous(word: str, path: "Path | str | None" = None) -> None:
    """Дописывает word в not_ambiguous_stress_words_ru.txt (создаёт файл при
    необходимости, не дублирует уже существующие записи) и сбрасывает кэш
    _load_ambiguous_stress_words, чтобы изменение подхватилось без
    перезапуска программы."""
    global _ambiguous_stress_words
    p = Path(path) if path else DEFAULT_NOT_AMBIGUOUS_STRESS_WORDS_PATH
    existing = _load_not_ambiguous_stress_words(p)
    word = word.strip().lower()
    if word in existing:
        return
    existing.add(word)
    p.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
    _ambiguous_stress_words = None  # сбрасываем кэш — перечитается со следующим вызовом


def load_stress_dictionary(path: "Path | str | None" = None,
                            book_overrides_path: "Path | str | None" = None) -> dict:
    """The USER's manual stress dictionaries only: forced_stress_words.json (top priority),
    manual_stress_overrides.json and the per-book overrides. These words bypass the neural
    network completely. The old auto-built stress_dictionary.json is no longer used (it was
    built from a corpus with systemic errors, e.g. "с+ебе"); `path` is kept for compatibility."""
    return load_manual_stress_entries(book_overrides_path)


DEFAULT_FORCED_STRESS_PATH = _app_paths.user_file("forced_stress_words.json")


def _load_forced_stress(path: "Path | str | None" = None) -> dict:
    """forced_stress_words.json — слова, которые ВСЕГДА читаются так, как
    записано (выше LLM, словаря и silero-stress), например {"душу": "душ+у"}.
    Для случаев, где пользователь точно знает нужное ударение."""
    p = Path(path) if path else DEFAULT_FORCED_STRESS_PATH
    try:
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {str(k).lower(): str(v) for k, v in raw.items() if not str(k).startswith("_")}
    except Exception as e:
        print(f"Не удалось прочитать {p}: {e}")
    return {}


def load_manual_stress_entries(book_overrides_path: "Path | str | None" = None) -> dict:
    """Everything the USER wrote by hand, merged with the same precedence as
    load_stress_dictionary (global manual overrides < book overrides < forced_stress_words.json).
    These words have ABSOLUTE authority: no context resolver, ё-pass, silero-stress or
    silero-stress may change them."""
    d = dict(_load_manual_stress_overrides())
    if book_overrides_path:
        d.update(_load_manual_stress_overrides(book_overrides_path))
    d.update(_load_forced_stress())
    return d


DEFAULT_MANUAL_STRESS_OVERRIDES_PATH = _app_paths.user_file("manual_stress_overrides.json")


def ensure_forced_stress_file(path: "Path | str | None" = None) -> Path:
    """Creates forced_stress_words.json ({}) when missing; returns its path (GUI 'open dictionary')."""
    p = Path(path) if path else DEFAULT_FORCED_STRESS_PATH
    if not p.exists():
        p.write_text("{\n}\n", encoding="utf-8")
    return p


def book_manual_stress_overrides_path(book_path: "Path | str") -> Path:
    """Путь к файлу с вручную заданными ударениями ДЛЯ КОНКРЕТНОЙ КНИГИ —
    рядом с самой книгой, как и check_ambiguous_stress.py кладёт свой
    текстовый отчёт (книга.fb2 -> книга.manual_stress_overrides.json).
    Отдельно от общего manual_stress_overrides.json: слово может по-разному
    читаться в разных книгах, поэтому решение, принятое для одной книги, не
    должно молча становиться глобальным правилом для всех остальных."""
    return Path(str(Path(book_path).with_suffix("")) + ".manual_stress_overrides.json")


def _load_manual_stress_overrides(path: "Path | str | None" = None) -> dict:
    """Читает manual_stress_overrides.json — отдельный файл для ударений,
    которые пользователь явно указал вручную (через диалог «Проверить
    ударения» в GUI) для конкретных слов из конкретной книги. Специально
    отдельно от stress_dictionary.json и НЕ фильтруется по списку
    омографов — см. load_stress_dictionary."""
    p = Path(path) if path else DEFAULT_MANUAL_STRESS_OVERRIDES_PATH
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k).lower(): str(v) for k, v in raw.items()}
    except Exception as e:
        print(f"Не удалось прочитать {p}: {e} — продолжаю без ручных ударений.")
        return {}


def save_manual_stress_overrides(entries: dict, path: "Path | str | None" = None) -> None:
    """Дописывает entries ({слово: 'сл+ово'}) в manual_stress_overrides.json,
    сохраняя уже имеющиеся там записи (читает-объединяет-пишет, как
    enrich_stress_dictionary_online). Бросает исключение наружу при ошибке
    записи — вызывающий (GUI) сам решает, как её показать пользователю."""
    p = Path(path) if path else DEFAULT_MANUAL_STRESS_OVERRIDES_PATH
    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    existing.update({str(k).lower(): str(v) for k, v in entries.items()})
    p.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def add_preferred_stress(word: str, replacement: str) -> Path:
    """Добавляет слово с личным предпочтением по ударению (не привязано к
    конкретной книге и не найдено сканированием текста — просто «мне так
    привычнее», см. GUI-кнопку «Словарь ударений») в подходящий файл:

    - если слово НЕ входит в список омографов (_load_ambiguous_stress_words)
      — прямо в общий stress_dictionary.json, как обычная словарная запись;
    - если слово омограф (не всегда однозначно, silero-stress сам решает по
      контексту) — в manual_stress_overrides.json, потому что
      load_stress_dictionary отфильтровывает омографы из stress_dictionary.json
      и запись туда молча пропадала бы при следующей загрузке.

    В обоих случаях действует ГЛОБАЛЬНО, для всех книг (в отличие от
    book_manual_stress_overrides_path — тех ударений, что назначены для
    конкретной книги через «Проверить ударения» по конкретному тексту).
    Возвращает путь к файлу, куда фактически записано — для сообщения
    пользователю."""
    word_l = word.strip().lower()
    ambiguous = _load_ambiguous_stress_words()
    if word_l in ambiguous:
        save_manual_stress_overrides({word_l: replacement})
        return DEFAULT_MANUAL_STRESS_OVERRIDES_PATH
    else:
        save_manual_stress_overrides({word_l: replacement}, DEFAULT_STRESS_DICT_PATH)
        return DEFAULT_STRESS_DICT_PATH


def _read_json_dict(path: "Path | str") -> dict:
    """Общая маленькая утилита: читает JSON-файл вида {слово: значение},
    возвращает {} на любую ошибку/отсутствие файла — не бросает исключений."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def list_stress_entries(book_path: "Path | str | None" = None) -> list:
    """Собирает ВСЕ записи ударений из всех источников для показа/управления
    в GUI (кнопка «Словарь ударений»): основной stress_dictionary.json,
    общий manual_stress_overrides.json и, если передан book_path — ещё и
    книжный <книга>.manual_stress_overrides.json. Возвращает список
    {"word": ..., "stress": ..., "source": <путь к файлу, откуда запись>},
    отсортированный по слову. Одно и то же слово может встретиться из
    нескольких источников сразу отдельными строками (например, есть и в
    общем словаре, и переопределено для конкретной книги) — это нормально,
    видно, что применится (книжная запись имеет приоритет при синтезе)."""
    sources = [DEFAULT_STRESS_DICT_PATH, DEFAULT_MANUAL_STRESS_OVERRIDES_PATH]
    if book_path:
        sources.append(book_manual_stress_overrides_path(book_path))
    entries = []
    for p in sources:
        for word, stress in _read_json_dict(p).items():
            entries.append({"word": str(word).lower(), "stress": str(stress), "source": p})
    entries.sort(key=lambda e: (e["word"], str(e["source"])))
    return entries


def delete_stress_entry(word: str, source: "Path | str") -> bool:
    """Удаляет word из конкретного файла source (один из путей, которые
    возвращает list_stress_entries). Возвращает True, если запись была и
    удалена, False, если её там не было (ничего не делает). Бросает
    исключение наружу только при ошибке ЗАПИСИ — GUI сам решает, как её
    показать."""
    p = Path(source)
    data = _read_json_dict(p)
    word_l = word.strip().lower()
    if word_l not in data:
        return False
    del data[word_l]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return True


_KEEP_HYPHENS = False


def set_keep_hyphens(enabled: bool) -> None:
    """True when Silero's own put_accent is OFF for the run (the client chain marks every word itself):
    the hyphen of a dictionary replacement ("сан-март+ин", "вс+ё-таки") is then kept as in the book text.
    False (put_accent may run inside Silero, which would stress the first half of a hyphenated word on its
    own): the hyphen is dropped from the replacement."""
    global _KEEP_HYPHENS
    _KEEP_HYPHENS = bool(enabled)


def _prepare_stress_replacement(word: str, replacement: str) -> str:
    """Готовит значение из словаря ударений к вставке в текст, отдельно от
    самого поиска в словаре — общая часть для apply_stress_dictionary и
    apply_stress_dictionary_protected.

    Два преобразования:
    1) сохраняем регистр первой буквы оригинала (простая эвристика — не
       пытаемся повторить регистр каждой буквы, этого достаточно для
       "Имя"/"имя"/"ИМЯ" в начале предложения);
    2) если исходное слово-ключ составное, через дефис ("во-первых",
       "по-русски", "в-третьих" и т.п.) — дефис из ЗАМЕНЫ убираем (только
       из неё, не из остального текста книги). Причина: Silero сам
       разбивает текст на слова по дефису для своего ВСТРОЕННОГО put_accent
       (это отдельный механизм, до которого apply_stress_dictionary_protected
       не дотягивается — тот защищает только от silero-stress) — увидев "во"
       отдельным "словом" без явного "+", он присваивает ему собственное
       ударение, и получается заметное, лишнее ударение на "во" вдобавок к
       правильному на "первых". Дефис в исходном тексте книги (обычные
       тире/составные слова, которых нет в словаре) при этом никак не
       затрагивается — это только для конкретных слов из словаря."""
    def _cap(part: str) -> str:
        if part.startswith("+") and len(part) > 1:               # "+он" -> "+Он" (was lowercased)
            return "+" + part[1].upper() + part[2:]
        return part[:1].upper() + part[1:]

    w_parts, r_parts = word.split("-"), replacement.split("-")
    if "-" in word and _KEEP_HYPHENS and len(w_parts) == len(r_parts):
        # "Сан-Мартин" -> "Сан-Март+ин": the capitalisation of EVERY part is copied
        replacement = "-".join(_cap(r) if w[:1].isupper() and r else r for w, r in zip(w_parts, r_parts))
    elif word[0].isupper() and replacement:
        replacement = _cap(replacement)
    if "-" in word and not _KEEP_HYPHENS:
        replacement = replacement.replace("-", "")
    return replacement


# --- Контекстное разрешение ударения по падежу для отдельных омографов ----
#
# Часть слов вроде "звезды" не получится корректно занести в обычный
# словарь ударений (stress_dictionary.json), потому что у них РАЗНОЕ
# правильное ударение в зависимости от падежа/числа: "звезды́" (род.п. ед.ч.,
# "командным пунктом звезды́") vs "зв+ёзды" (им./вин.п. мн.ч., "на небе
# зажглись звёзды"). Такие слова были исключены из словаря целиком (см.
# ambiguous_stress_words_ru.txt), и по умолчанию решение отдаётся на откуп
# silero-stress — но silero-stress тоже иногда ошибается на таких примерах. Ниже —
# небольшой, вручную проверенный список самых частых слов с чередованием
# ударения по типу "род.п. ед.ч. (на окончании) / им.п. мн.ч. (на основе)"
# (это регулярный для русского языка тип склонения — т.н. "тип b" по
# Зализняку) и простая эвристика на pymorphy3 + локальный контекст, чтобы
# выбрать нужную форму. Если pymorphy3 не установлен или ничего не удаётся
# определить — слово просто не трогаем (оставляем silero-stress как раньше),
# никогда не гадаем вслепую.
_CASE_AMBIGUOUS_NOUNS = {
    "звезды":  {"gen_sg": "звезд+ы",  "nom_pl": "зв+ёзды"},
    "сестры":  {"gen_sg": "сестр+ы",  "nom_pl": "с+ёстры"},
    "стены":   {"gen_sg": "стен+ы",   "nom_pl": "ст+ены"},
    "страны":  {"gen_sg": "стран+ы",  "nom_pl": "стр+аны"},
    "горы":    {"gen_sg": "гор+ы",    "nom_pl": "г+оры"},
    "слезы":   {"gen_sg": "слез+ы",   "nom_pl": "сл+ёзы"},
    "руки":    {"gen_sg": "рук+и",    "nom_pl": "р+уки"},
    "ноги":    {"gen_sg": "ног+и",    "nom_pl": "н+оги"},
    "головы":  {"gen_sg": "голов+ы",  "nom_pl": "г+оловы"},
    "души":    {"gen_sg": "душ+и",    "nom_pl": "д+уши"},
    # Расширение списка (по просьбе пользователя — "мало 10 слов") тем же
    # регулярным типом склонения ("тип b" по Зализняку): ударение на
    # окончании в род.п. ед.ч., на основе — в им.п. мн.ч. Каждое слово
    # проверено вручную по таблицам склонения, не сгенерировано автоматически.
    "волны":   {"gen_sg": "волн+ы",   "nom_pl": "в+олны"},
    "весны":   {"gen_sg": "весн+ы",   "nom_pl": "в+ёсны"},
    "вёсны":   {"gen_sg": "весн+ы",   "nom_pl": "в+ёсны"},
    "иглы":    {"gen_sg": "игл+ы",    "nom_pl": "+иглы"},
    "козы":    {"gen_sg": "коз+ы",    "nom_pl": "к+озы"},
    "овцы":    {"gen_sg": "овц+ы",    "nom_pl": "+овцы"},
    "свечи":   {"gen_sg": "свеч+и",   "nom_pl": "св+ечи"},
    "скалы":   {"gen_sg": "скал+ы",   "nom_pl": "ск+алы"},
    "земли":   {"gen_sg": "земл+и",   "nom_pl": "з+емли"},
    "семьи":   {"gen_sg": "семь+и",   "nom_pl": "с+емьи"},
    "толпы":   {"gen_sg": "толп+ы",   "nom_pl": "т+олпы"},
    "доски":   {"gen_sg": "доск+и",   "nom_pl": "д+оски"},
    "полосы":  {"gen_sg": "полос+ы",  "nom_pl": "п+олосы"},
    "травы":   {"gen_sg": "трав+ы",   "nom_pl": "тр+авы"},
    "игры":    {"gen_sg": "игр+ы",    "nom_pl": "+игры"},
    "стрелы":  {"gen_sg": "стрел+ы",  "nom_pl": "стр+елы"},
    "пилы":    {"gen_sg": "пил+ы",    "nom_pl": "п+илы"},
    "дыры":    {"gen_sg": "дыр+ы",    "nom_pl": "д+ыры"},
    "совы":    {"gen_sg": "сов+ы",    "nom_pl": "с+овы"},
    "беды":    {"gen_sg": "бед+ы",    "nom_pl": "б+еды"},
    "плиты":   {"gen_sg": "плит+ы",   "nom_pl": "пл+иты"},
    "воды":    {"gen_sg": "вод+ы",    "nom_pl": "в+оды"},
    "косы":    {"gen_sg": "кос+ы",    "nom_pl": "к+осы"},
}

# Предлоги/слова непосредственно перед словом, однозначно требующие
# родительного падежа — сильный локальный сигнал, не требующий pymorphy3.
_GENITIVE_CONTEXT_RE = re.compile(
    r"(?:без|для|около|вокруг|среди|кроме|после|вместо|ради|вдоль|возле|"
    r"напротив|сверх|накануне|из-за|из-под|от|из|до|у|с|со|против)\s*$",
    re.IGNORECASE,
)
# Числительные-счётчики 2-4 (и "оба"/"обе") тоже требуют род.п. ед.ч.
_COUNTING_CONTEXT_RE = re.compile(
    r"(?:два|две|три|четыре|оба|обе)\s*$", re.IGNORECASE,
)

# Симметричные признаки им./вин.п. МНОЖЕСТВЕННОГО числа — раньше их не было
# вовсе, решение при отсутствии род.падежного предлога/числительного
# целиком отдавалось на откуп статистике pymorphy3 (которая по чистой
# частотности форм часто выигрывает у им.п. мн.ч. даже там, где по факту
# в тексте множественное число). Указательные/притяжательные местоимения и
# количественные слова во мн.ч. перед существительным — сильный сигнал
# "это точно множественное число", не менее надёжный, чем предлог для
# родительного падежа.
_PLURAL_CONTEXT_RE = re.compile(
    r"(?:эти|те|какие|некоторые|многие|немногие|все|всех|наши|ваши|мои|твои|"
    r"свои|их|other|несколько|сколько)\s*$",
    re.IGNORECASE,
)
# Прилагательное/причастие непосредственно перед словом, согласованное во
# мн.ч. (типичные окончания им./вин.п. мн.ч. "-ые"/"-ие") — например
# "далёкие звезды", "родные сестры". Не идеально (у род.п. ед.ч.
# прилагательных окончание "-ой"/"-ей", так что коллизий с этим паттерном
# практически не бывает), но не исключает и других формально возможных
# грамматических категорий — поэтому это тоже только один из сигналов, а
# не абсолютное правило.
_PLURAL_ADJECTIVE_CONTEXT_RE = re.compile(
    r"[а-яё]*(?:ые|ие)\s*$", re.IGNORECASE,
)

_pymorphy_analyzer = None
_pymorphy_load_failed = False


def _get_pymorphy_analyzer():
    """Лениво создаёт pymorphy3.MorphAnalyzer(). Необязательная зависимость:
    если pymorphy3 не установлен, просто возвращаем None и больше не
    пытаемся (без пятен на консоль при каждом слове)."""
    global _pymorphy_analyzer, _pymorphy_load_failed
    if _pymorphy_analyzer is not None or _pymorphy_load_failed:
        return _pymorphy_analyzer
    try:
        import pymorphy3
        _pymorphy_analyzer = pymorphy3.MorphAnalyzer()
    except Exception:
        _pymorphy_load_failed = True
        _pymorphy_analyzer = None
    return _pymorphy_analyzer


def _resolve_case_ambiguous_noun(word_lower: str):
    """Возвращает "gen_sg" или "nom_pl" (или None, если не удалось решить)
    для слова word_lower на основе суммарной уверенности разборов pymorphy3
    (без учёта контекста предложения — контекст проверяется отдельно,
    до вызова этой функции, как более сильный сигнал)."""
    analyzer = _get_pymorphy_analyzer()
    if analyzer is None:
        return None
    try:
        parses = analyzer.parse(word_lower)
    except Exception:
        return None
    gen_sg_score = 0.0
    nom_pl_score = 0.0
    for p in parses:
        tag = str(p.tag)
        score = p.score or 0.0
        if "sing" in tag and "gent" in tag:
            gen_sg_score += score
        elif "plur" in tag and ("nomn" in tag or "accs" in tag):
            nom_pl_score += score
    if gen_sg_score == 0.0 and nom_pl_score == 0.0:
        return None
    return "gen_sg" if gen_sg_score >= nom_pl_score else "nom_pl"


def resolve_case_ambiguous_nouns(text: str) -> dict:
    """Сканирует text на предмет слов из _CASE_AMBIGUOUS_NOUNS и для каждого
    вхождения пытается определить нужную форму (род.п. ед.ч. или им.п.
    мн.ч.) по локальному контексту — сначала проверяются ЯВНЫЕ грамматические
    признаки в обе стороны (предлог/числительное => род.п. ед.ч.;
    указательное/притяжательное местоимение или согласованное прилагательное
    во мн.ч. => им.п. мн.ч.), и только если ни один явный признак не
    сработал — решение отдаётся pymorphy3 (по чистой частотности форм слова
    вне контекста, без учёта соседних слов). Возвращает {слово_lower:
    "сл+ово"} — временный, НЕ сохраняемый в JSON словарь для подмешивания в
    stress_dict перед apply_stress_dictionary(_protected) для конкретного
    фрагмента текста. Никогда не бросает исключений — в худшем случае
    вернёт {}."""
    result = {}
    try:
        for m in _STRESS_WORD_RE.finditer(text):
            word = m.group(0)
            word_lower = word.lower()
            forms = _CASE_AMBIGUOUS_NOUNS.get(word_lower)
            if forms is None or word_lower in result:
                continue
            prefix = text[:m.start()]
            has_genitive_marker = bool(
                _GENITIVE_CONTEXT_RE.search(prefix) or _COUNTING_CONTEXT_RE.search(prefix)
            )
            has_plural_marker = bool(
                _PLURAL_CONTEXT_RE.search(prefix) or _PLURAL_ADJECTIVE_CONTEXT_RE.search(prefix)
            )
            if has_genitive_marker and not has_plural_marker:
                choice = "gen_sg"
            elif has_plural_marker and not has_genitive_marker:
                choice = "nom_pl"
            elif has_genitive_marker and has_plural_marker:
                # Оба признака одновременно — противоречие (например, ошибка
                # в самом тексте или наложение из соседнего предложения),
                # надёжнее довериться pymorphy3, чем выбирать наугад.
                choice = _resolve_case_ambiguous_noun(word_lower)
            else:
                choice = _resolve_case_ambiguous_noun(word_lower)
            if choice is not None:
                result[word_lower] = forms[choice]
    except Exception:
        return result
    return result


# --- Омографы, у которых совпадает написание СОВЕРШЕННО РАЗНЫХ слов ------
#
# Отдельный случай, не сводящийся к падежу одного и того же слова: "графа"
# может быть либо род.п. ед.ч. слова "граф" (дворянский титул — "не было
# графа" -> "гр+афа"), либо им.п. ед.ч. отдельного слова "графа" (колонка
# таблицы — "заполните графу" -> "им.п. графа́" -> "граф+а"). Это разные
# леммы с разными исходными словами, а не словоформы одного слова — поэтому
# решаем не по падежу/числу, а по тому, какую НОРМАЛЬНУЮ ФОРМУ (лемму)
# pymorphy3 считает более вероятной для этого написания.
_LEMMA_HOMOGRAPHS = {
    "графа": {"граф": "гр+афа", "графа": "граф+а"},
}


def _resolve_lemma_homograph(word_lower: str):
    """Возвращает готовую строку с ударением (или None) для word_lower из
    _LEMMA_HOMOGRAPHS, выбирая между вариантами по суммарной уверенности
    разборов pymorphy3, сгруппированных по нормальной форме (лемме)."""
    variants = _LEMMA_HOMOGRAPHS.get(word_lower)
    if not variants:
        return None
    analyzer = _get_pymorphy_analyzer()
    if analyzer is None:
        return None
    try:
        parses = analyzer.parse(word_lower)
    except Exception:
        return None
    scores = {}
    for p in parses:
        if p.normal_form in variants:
            scores[p.normal_form] = scores.get(p.normal_form, 0.0) + (p.score or 0.0)
    if not scores:
        return None
    best_lemma = max(scores, key=scores.get)
    return variants[best_lemma]


def resolve_lemma_homographs(text: str) -> dict:
    """Аналог resolve_case_ambiguous_nouns, но для _LEMMA_HOMOGRAPHS (разные
    слова с одинаковым написанием, не падежные формы одного слова). Тоже
    временный результат для подмешивания в stress_dict, ничего не пишет в
    JSON, никогда не бросает исключений."""
    result = {}
    try:
        for m in _STRESS_WORD_RE.finditer(text):
            word_lower = m.group(0).lower()
            if word_lower in result or word_lower not in _LEMMA_HOMOGRAPHS:
                continue
            replacement = _resolve_lemma_homograph(word_lower)
            if replacement is not None:
                result[word_lower] = replacement
    except Exception:
        return result
    return result


def apply_stress_dictionary(text: str, stress_dict: dict) -> str:
    """Заменяет слова из пользовательского словаря на их вариант с "+" перед
    ударной гласной — раньше, чем текст попадёт в silero-stress/put_accent, чтобы
    словарь имел приоритет над автоматическим угадыванием (для имён
    персонажей и специфических терминов книги, которые модель не знает)."""
    if not stress_dict and "ё" not in text.lower():
        return text

    def repl(m: "re.Match") -> str:
        word = m.group(0)
        replacement = (stress_dict or {}).get(word.lower())
        if replacement is None:
            # "ё" is always stressed: mark it ("подошёл" -> "подош+ёл") so the
            # engines/_sane_yo do not roll it back to "е".
            marked = _mark_yo_stress(text, m.start(), word)
            return marked if marked is not None else word
        return _prepare_stress_replacement(word, replacement)

    return _STRESS_WORD_RE.sub(repl, text)


# Маркер-обёртка для apply_stress_dictionary_protected/restore_stress_placeholders
# ниже — заведомо не встречается в обычном тексте книги (не кириллица, не
# цифры, так что numbers_to_words_ru его не тронет) и не похожа на слово,
# которое silero-stress попытался бы акцентировать.
_STRESS_PLACEHOLDER_RE = re.compile(r"STRESSDICT(\d+)")


# --- Multi-word entries in the user dictionaries ("мы все": "мы вс+е") --------------------------------
# A key with spaces is a PHRASE. It must have the same number of words as its value and the same letters
# (marks aside, е/ё equal). Every word of a matched phrase gets its own placeholder, so the neural net
# still sees the whole sentence around them. Phrase marks win over single-word entries.
_PHRASE_CACHE: dict = {}
_phrase_warned: set = set()


def _stress_plain(w: str) -> str:
    return w.replace("+", "").replace("ё", "е").replace("Ё", "Е").lower()


def _phrase_word_pattern(word: str) -> str:
    out = []
    for ch in _stress_plain(word):
        out.append("[её]" if ch == "е" else re.escape(ch))
    return r"\+?".join(out) + r"\+?"


def _compiled_phrases(stress_dict: dict) -> list:
    items = tuple(sorted((k, v) for k, v in stress_dict.items() if re.search(r"\s", str(k).strip())))
    if not items:
        return []
    hit = _PHRASE_CACHE.get(items)
    if hit is not None:
        return hit
    pats = []
    for k, v in items:
        kw, vw = str(k).split(), str(v).split()
        if len(kw) != len(vw) or any(_stress_plain(a) != _stress_plain(b) for a, b in zip(kw, vw)):
            if k not in _phrase_warned:
                _phrase_warned.add(k)
                print(f"Stress dictionary: phrase {k!r} -> {v!r} skipped (the value must be the same words "
                      "with a '+' before the stressed vowel).")
            continue
        rx = (r"(?<![А-Яа-яЁё+-])"
              + r"\s+".join("(" + _phrase_word_pattern(w) + ")" for w in kw)
              + r"(?![А-Яа-яЁё+]|-[А-Яа-яЁё])")
        pats.append((len(kw), re.compile(rx, re.I), vw))
    pats.sort(key=lambda t: -t[0])          # longest phrase first
    if len(_PHRASE_CACHE) > 8:
        _PHRASE_CACHE.clear()
    _PHRASE_CACHE[items] = pats
    return pats


def _phrase_forms(text: str, stress_dict: dict) -> dict:
    """{start offset of a word: replacement} for every word of every phrase found in `text`."""
    forms: dict = {}
    for _n, rx, vw in _compiled_phrases(stress_dict or {}):
        for m in rx.finditer(text):
            starts = [m.start(i + 1) for i in range(len(vw))]
            if any(st in forms for st in starts):
                continue                      # overlaps a longer phrase already applied
            for st, w in zip(starts, vw):
                forms[st] = w
    return forms


def apply_stress_dictionary_protected(text: str, stress_dict: dict,
                                       occurrence_forms: "dict | None" = None) -> "tuple[str, dict]":
    """То же самое, что apply_stress_dictionary, но вместо того чтобы сразу
    подставить в текст готовое "сл+ово" из словаря, подставляет временный
    непечатаемый маркер-плейсхолдер и возвращает (текст_с_плейсхолдерами,
    {плейсхолдер: готовая_замена}). Используется, когда после словаря текст
    ещё пройдёт через silero-stress — silero-stress разбирает текст
    на токены сам и на составных словах через дефис ("во-первых",
    "по-русски" и т.п.) может не распознать уже стоящее "+"-ударение и
    добавить своё собственное ЕЩЁ РАЗ, поверх — из-за двух ударений в одном
    слове получается характерное "странное растягивание" при синтезе.
    Плейсхолдер (не кириллица, не похож на слово) silero-stress не трогает
    вообще, а после него исходная словарная форма возвращается на место
    через restore_stress_placeholders — так словарь остаётся полностью
    защищён от повторной обработки, а не просто "обычно не переписывается"."""
    if not stress_dict and not occurrence_forms and "ё" not in text.lower():
        return text, {}

    phrase_forms = _phrase_forms(text, stress_dict)
    if phrase_forms:
        occurrence_forms = {**phrase_forms, **(occurrence_forms or {})}

    placeholders: dict = {}
    counter = [0]

    def repl(m: "re.Match") -> str:
        word = m.group(0)
        # occurrence_forms — решение по КОНКРЕТНОМУ месту текста (омограф,
        # разобранный LLM по контексту), приоритетнее словаря.
        replacement = (occurrence_forms or {}).get(m.start())
        if replacement is None:
            replacement = (stress_dict or {}).get(word.lower())
        if replacement is None:
            # restored/original "ё" -> "+ё", protected from silero-stress/_sane_yo
            replacement = _mark_yo_stress(text, m.start(), word)
        if replacement is None:
            return word
        replacement = _prepare_stress_replacement(word, replacement)
        idx = counter[0]
        counter[0] += 1
        placeholder = f"STRESSDICT{idx}"
        placeholders[placeholder] = replacement
        return placeholder

    protected = _STRESS_WORD_RE.sub(repl, text)
    return protected, placeholders


# Односложные слова, которым метка нужна: без неё Silero читает «лет» как «лёт», «пять» как «пить», «сто» как «ста».
_MONO_KEEP_STRESS = frozenset({"лет", "пять", "сто"})
# Связка «да то,» ("Да то, Стью, ...") без обеих меток читается невнятно — оставляем «д+а т+о» как есть.
_MONO_KEEP_PHRASE = r"(?<![А-Яа-яЁё+])[Дд]\+а т\+о(?![А-Яа-яЁё+])"
_MONO_STRESS_RE = re.compile(_MONO_KEEP_PHRASE + r"|[А-Яа-яЁё+]*\+[А-Яа-яЁё+]*")


def strip_monosyllable_stress(text: str) -> str:
    """Убирает "+" в словах с одной гласной ("+и", "+он", "н+е"): ударение там
    однозначно, а лишняя метка в SSML/Silero даёт неестественное растягивание."""
    if not _STRIP_MONO_ENABLED:
        return text

    def repl(m: "re.Match") -> str:
        w = m.group(0)
        if " " in w:
            return w          # phrase kept whole (_MONO_KEEP_PHRASE)
        plain = w.replace("+", "")
        if "ё" in plain.lower():
            return w          # "ё" is always stressed — keep "вс+ё", "ч+ём", "+ёж"
        if plain.lower() in _MONO_KEEP_STRESS:
            return w
        # метка в дефисном слове с единственной «+» («Сен-Ж+юст»): «ж+юст» не односложное слово; «в+о-втор+ых», «п+о-насто+ящему» метку теряют, как раньше
        s, e = m.start(), m.end()
        full = m.string
        while s > 1 and full[s - 1] == "-" and (full[s - 2].isalpha() or full[s - 2] == "+"):
            s -= 1
            while s > 0 and (full[s - 1].isalpha() or full[s - 1] == "+"):
                s -= 1
        while e < len(full) - 1 and full[e] == "-" and (full[e + 1].isalpha() or full[e + 1] == "+"):
            e += 1
            while e < len(full) and (full[e].isalpha() or full[e] == "+"):
                e += 1
        compound_raw = full[s:e]
        compound = compound_raw.replace("+", "")
        if compound_raw.count("+") == 1 and len(re.findall(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", compound)) > 1:
            return w
        if len(re.findall(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", plain)) <= 1:
            return plain
        return w
    return _MONO_STRESS_RE.sub(repl, text)



# ---------------------------------------------------------------------------
# Pronunciation substitutions: TTS-ONLY rephrasing ("БГБ" -> "б+э г+э б+э"). The book text is never changed; the file
# pronunciation_substitutions.json ({"БГБ": "б+э г+э б+э"}) lives next to the scripts/.exe. Keys are whole words, case-SENSITIVE,
# each inflected form is its own key; the value is spoken as written (put "+" before stressed vowels yourself).
# ---------------------------------------------------------------------------
DEFAULT_PRONUNCIATION_PATH = _app_paths.user_file("pronunciation_substitutions.json")
_PRON_CH = "А-Яа-яЁёA-Za-z0-9"
_PRON_PH_RE = re.compile(r"PRONSUB([A-Z]+)")
_pron_regex_cache: dict = {}


def load_pronunciation_substitutions(path=None, book_path=None) -> dict:
    out = {}
    for p in (path or DEFAULT_PRONUNCIATION_PATH, book_path):
        if not p:
            continue
        try:
            pp = Path(p)
            if pp.exists():
                raw = json.loads(pp.read_text(encoding="utf-8"))
                out.update({str(k): str(v) for k, v in raw.items() if not str(k).startswith("_") and str(v).strip()})
        except Exception as e:
            print(f"Не удалось прочитать {p}: {e}")
    return out


def _pron_regex(subs: dict):
    keys = tuple(sorted(subs, key=len, reverse=True))
    rx = _pron_regex_cache.get(keys)
    if rx is None:
        rx = re.compile(r"(?<![" + _PRON_CH + r"])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![" + _PRON_CH + r"])")
        _pron_regex_cache[keys] = rx
    return rx


def _pron_ph_name(i: int) -> str:
    i += 1
    s = ""
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return "PRONSUB" + s


def protect_pronunciation_substitutions(text: str, subs: dict):
    """Hides every substituted word behind a letters-only placeholder (survives numbers_to_words_ru, silero-stress, the gate)."""
    if not subs or not text:
        return text, {}
    held: dict = {}

    def _sub(m):
        name = _pron_ph_name(len(held))
        held[name] = subs[m.group(0)]
        return name
    return _pron_regex(subs).sub(_sub, text), held


def restore_pronunciation_substitutions(text: str, held: dict) -> str:
    if not held:
        return text
    return _PRON_PH_RE.sub(lambda m: held.get("PRONSUB" + m.group(1), ""), text)


_LATIN_DIGRAPHS = (("sch", "ш"), ("tch", "ч"), ("th", "з"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"), ("ck", "к"), ("oo", "у"),
                   ("ee", "и"), ("kh", "х"), ("zh", "ж"), ("qu", "кв"))
_LATIN_LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz",
                          ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к", "р",
                           "с", "т", "а", "в", "в", "кс", "и", "з"]))


def latin_to_cyrillic_ru(text: str) -> str:
    """Safety net: Silero's alphabet is Cyrillic - ONE Latin letter in an SSML request makes the whole request fail
    ('NoneType' object has no attribute 'keys') and the chunk falls back to a flat plain-text synthesis. Whatever Latin
    is left after the pronunciation substitutions is transliterated roughly, so the chunk keeps pauses and prosody."""
    if not text or not re.search(r"[A-Za-z]", text):
        return text

    def conv(m: "re.Match") -> str:
        w, low, out, i = m.group(0), m.group(0).lower(), [], 0
        while i < len(low):
            for dg, ru in _LATIN_DIGRAPHS:
                if low.startswith(dg, i):
                    out.append(ru)
                    i += len(dg)
                    break
            else:
                out.append(_LATIN_LETTERS.get(low[i], ""))
                i += 1
        res = "".join(out)
        return res[:1].upper() + res[1:] if w[:1].isupper() else res
    return re.sub(r"[A-Za-z]+", conv, text)


def pronunciation_digest(subs: dict, text: str) -> str:
    """Hash of the substitutions that occur in `text` ("" when none) - part of the chapter fingerprint, so editing ONE
    substitution re-synthesizes exactly the chapters that use it."""
    if not subs or not text:
        return ""
    used = sorted({m.group(0) for m in _pron_regex(subs).finditer(text)})
    if not used:
        return ""
    return hashlib.sha256(json.dumps([(k, subs[k]) for k in used], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def restore_stress_placeholders(text: str, placeholders: dict) -> str:
    """Возвращает на место словарные ударения, спрятанные
    apply_stress_dictionary_protected под плейсхолдерами — после того, как
    silero-stress (или что угодно ещё) обработал остальной текст. Заменяет все
    плейсхолдеры ЗА ОДИН проход регулярным выражением, а не по очереди
    через text.replace() для каждого — иначе, например, "STRESSDICT1"
    случайно совпал бы с началом "STRESSDICT10" (это реальная подстрока)
    и испортил бы его; \\d+ в _STRESS_PLACEHOLDER_RE жадный, поэтому при
    поиске по тексту сразу захватывает число целиком и такой коллизии не
    возникает. Если какой-то плейсхолдер вдруг не найден в placeholders
    (не должно происходить, но на всякий случай) — на этом этапе он просто
    вырезается, а не остаётся в тексте нечитаемым мусором."""
    if not placeholders:
        return strip_monosyllable_stress(text)

    def repl(m: "re.Match") -> str:
        return placeholders.get(m.group(0), "")

    text = _STRESS_PLACEHOLDER_RE.sub(repl, text)
    return strip_monosyllable_stress(text)


# --------------------------------------------------------------------------
# Онлайн-подстраховка для словаря ударений: если слова нет в локальном
# stress_dictionary.json, пробуем найти его ударение в нескольких бесплатных
# (без ключа API) онлайн-словарях по очереди — см. _STRESS_ONLINE_PROVIDERS
# ниже. Первый источник, который нашёл слово, и используется; остальные не
# опрашиваются. Найденное дописывается в тот же JSON, чтобы при следующих
# запусках сеть больше не требовалась. Любая ошибка — нет интернета, сайт
# недоступен, слова там нет, файл словаря нельзя записать — НЕ прерывает
# синтез, а только пишется в консоль: в худшем случае слово просто
# останется без словарного ударения и ударение расставит silero-stress/
# put_accent, как и раньше.
# --------------------------------------------------------------------------

# Слова, для которых онлайн-поиск уже пробовали (по всем источникам) и не
# нашли (или была ошибка сети) — чтобы не долбить сайты повторным запросом
# на каждое вхождение одного и того же слова в книге.
_online_stress_failed_cache: set = set()


class _StressSourceRateLimited(Exception):
    """Сигнал "источник ответил 429 (слишком много запросов)" — отдельно
    от прочих ошибок, чтобы вызывающий код мог отключить именно этот
    источник до конца запуска, а не просто залогировать единичную ошибку
    (см. _provider_disabled в fetch_stress_online)."""
    pass


# --------------------------------------------------------------------------
# Источник 0 (проверяется первым, БЕЗ обращения к сети): офлайн-корпус
# ударений русского языка — TSV-датасет из открытого репозитория
# https://github.com/Koziev/NLP_Datasets (Stress/all_accents.zip, собран
# по Википедии/Викисловарю + грамматическому словарю словоформ, ~1.68 млн
# слов и словоформ). Файл лежит рядом со скриптом в сжатом виде
# (stress_corpus_ru.tsv.gz, формат каждой строки: "слово<TAB>слово_с_^_
# перед_ударной_буквой"). Основной смысл: у большинства слов ударение
# находится мгновенно и локально, без единого HTTP-запроса — Викисловарь
# (см. ниже) при этом используется гораздо реже и почти не упирается в
# лимит запросов (429 Too Many Requests), с которым раньше сталкивались
# при прогоне книги целиком через один онлайн-источник.
# --------------------------------------------------------------------------

DEFAULT_OFFLINE_STRESS_CORPUS_PATH = _app_paths.resolve_asset("stress_corpus_ru.tsv.gz")

# None = ещё не пробовали загрузить; {} = пробовали, но не получилось
# (файла нет / битый / кончилась память и т.п.) — в обоих случаях больше
# не пытаемся загружать повторно в рамках одного процесса.
_offline_stress_corpus: "dict | None" = None
_offline_stress_corpus_load_failed = False


def _load_offline_stress_corpus(path: "Path | str | None" = None) -> dict:
    """Лениво загружает и кэширует офлайн-корпус ударений (см. описание
    выше) в память как {слово_в_нижнем_регистре: "сл+ово"}. Загружается
    один раз за весь запуск (несколько секунд на ~1.7 млн строк). Файл
    необязателен: если его нет рядом со скриптом (например, кто-то не
    скачал/не скопировал stress_corpus_ru.tsv.gz) — молча используется
    пустой словарь, синтез продолжается через silero-stress/онлайн-источники,
    как и раньше. Любая ошибка чтения (битый gzip, нехватка памяти и
    т.п.) тоже не прерывает синтез — только один раз печатается
    предупреждение."""
    global _offline_stress_corpus, _offline_stress_corpus_load_failed
    if _offline_stress_corpus is not None:
        return _offline_stress_corpus
    if _offline_stress_corpus_load_failed:
        return {}
    p = Path(path) if path else DEFAULT_OFFLINE_STRESS_CORPUS_PATH
    if not p.exists():
        _offline_stress_corpus_load_failed = True
        return {}
    try:
        import gzip
        corpus = {}
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                word, accented = parts
                # Формат исходного датасета — "^" перед ударной гласной,
                # переводим в формат Silero ("+").
                corpus[word.lower()] = accented.replace("^", "+").lower()
        _offline_stress_corpus = corpus
        print(f"Загружен офлайн-корпус ударений: {len(corpus)} слов(о/форм) из {p}")
        return corpus
    except Exception as e:
        _offline_stress_corpus_load_failed = True
        print(f"Не удалось загрузить офлайн-корпус ударений {p}: {e} — "
              "продолжаю без него (останутся silero-stress/онлайн-источники).")
        return {}


def fetch_stress_from_offline_corpus(word: str, timeout: float = None) -> "str | None":
    """Источник 0: локальный офлайн-корпус (см. выше). timeout принимается
    для единообразия сигнатуры с остальными источниками, но не
    используется — обращения к сети тут нет."""
    corpus = _load_offline_stress_corpus()
    if not corpus:
        return None
    return corpus.get(word.lower())


_WIKTIONARY_ACCENT_RE = re.compile(
    r"[А-Яа-яЁё]*[АаЕеЁёИиОоУуЫыЭэЮюЯя]́[А-Яа-яЁё]*"
)


def _wiktionary_accent_to_silero(cand: str) -> str:
    """Переводит найденную форму слова с объединяющим акутом (U+0301) в
    формат Silero: "+" перед ударной гласной, например "молоко́" -> "молок+о"."""
    chars = []
    i = 0
    while i < len(cand):
        if i + 1 < len(cand) and cand[i + 1] == "́":
            chars.append("+")
            chars.append(cand[i])
            i += 2
        else:
            chars.append(cand[i])
            i += 1
    return "".join(chars)


def _find_stress_in_wiktionary_content(content: str, word: str,
                                        section: "str | None" = None) -> "str | None":
    """Ищет в тексте страницы Викисловаря (вики-разметка) форму word,
    отмеченную ударением (объединяющий акут). Если задан section (например
    "Russian" — так называется раздел на en.wiktionary.org для русских
    слов), ищет только внутри него, отрезая текст по следующему заголовку
    того же уровня ("==...==") — иначе на многоязычных страницах
    en.wiktionary.org можно случайно найти совпадающее по написанию слово
    из другого языка с другим ударением."""
    if section:
        m = re.search(rf"==\s*{re.escape(section)}\s*==", content)
        if not m:
            return None
        rest = content[m.end():]
        m_next = re.search(r"\n==[^=]", rest)
        content = rest[:m_next.start()] if m_next else rest

    target = word.lower()
    for cand in _WIKTIONARY_ACCENT_RE.findall(content):
        if cand.replace("́", "").lower() == target:
            return _wiktionary_accent_to_silero(cand)
    return None


def _fetch_wiktionary_page(api_url: str, word: str, timeout: float) -> "str | None":
    """Общая часть запроса к API MediaWiki (одинакова для ru.wiktionary.org
    и en.wiktionary.org — оба на движке MediaWiki с тем же action=query).
    Возвращает сырой вики-текст страницы или None, если страницы нет/
    ответ пустой. Исключения (нет сети, таймаут и т.п.) не ловятся здесь —
    их ловит вызывающий код каждого источника отдельно (fetch_stress_from_*).
    Исключение: HTTP 429 (слишком много запросов) поднимается отдельным
    типом _StressSourceRateLimited, чтобы fetch_stress_online мог отличить
    "сайт временно ограничил частоту запросов" (нет смысла пробовать снова
    в этом запуске) от прочих ошибок."""
    import requests
    resp = requests.get(
        api_url,
        params={
            "action": "query",
            "titles": word,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        },
        timeout=timeout,
        headers={"User-Agent": "audiobook-fb2-reader/1.0 (stress lookup)"},
    )
    if resp.status_code == 429:
        raise _StressSourceRateLimited(f"HTTP 429 от {api_url}")
    resp.raise_for_status()
    data = resp.json()
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing"):
        return None
    revisions = pages[0].get("revisions") or []
    if not revisions:
        return None
    return ((revisions[0].get("slots") or {}).get("main") or {}).get("content") or None


def fetch_stress_from_ru_wiktionary(word: str, timeout: float = 4.0) -> "str | None":
    """Источник 1: русский Викисловарь (https://ru.wiktionary.org) — вся
    страница на русском, ищем ударение по всему тексту статьи (включая
    формы словоизменения в таблицах)."""
    content = _fetch_wiktionary_page("https://ru.wiktionary.org/w/api.php", word, timeout)
    if not content:
        return None
    return _find_stress_in_wiktionary_content(content, word)


def fetch_stress_from_en_wiktionary(word: str, timeout: float = 4.0) -> "str | None":
    """Источник 2 (резервный): англоязычный Викисловарь
    (https://en.wiktionary.org) тоже содержит статьи о русских словах (в
    разделе "==Russian=="), нередко с более полными таблицами
    словоизменения, чем на ru.wiktionary.org — полезно, когда слова не
    нашлось (или ещё не заведено) на русском Викисловаре."""
    content = _fetch_wiktionary_page("https://en.wiktionary.org/w/api.php", word, timeout)
    if not content:
        return None
    return _find_stress_in_wiktionary_content(content, word, section="Russian")


# Порядок важен: источники опрашиваются по очереди, пока один из них не
# найдёт слово. Офлайн-корпус проверяется первым (мгновенно, без сети) —
# у большинства слов ударение находится уже на этом шаге, и до
# Викисловарей дело просто не доходит. Каждый источник — независимая
# функция (word, timeout) -> "+"-форма или None; добавить ещё один
# словарь — значит дописать сюда ещё одну такую функцию.
_STRESS_ONLINE_PROVIDERS = [
    ("офлайн-корпус (Koziev/NLP_Datasets)", fetch_stress_from_offline_corpus),
    ("ru.wiktionary.org", fetch_stress_from_ru_wiktionary),
    ("en.wiktionary.org", fetch_stress_from_en_wiktionary),
]

# Источники (по имени из _STRESS_ONLINE_PROVIDERS), которые в этом запуске
# уже ответили 429 (слишком много запросов) — дальше не опрашиваются
# вовсе, чтобы не продолжать долбить сайт, который и так попросил
# притормозить. Офлайн-корпус сюда никогда не попадает — 429 у него не
# бывает физически.
_stress_provider_rate_limited: set = set()


def fetch_stress_online(word: str, timeout: float = 4.0) -> "str | None":
    """Пробует по очереди все источники из _STRESS_ONLINE_PROVIDERS
    (сначала офлайн-корпус, затем Викисловари) и возвращает первый
    найденный результат (слово в формате Silero: "сл+ово" — "+" перед
    ударной гласной). Возвращает None, если слово не нашлось ни в одном
    источнике. Ошибка одного источника (нет сети, сайт лежит, странный
    ответ) не мешает попробовать следующий — падает функция только если
    откажут вообще все, и то не исключением, а возвратом None с
    предупреждением в консоль. HTTP 429 у онлайн-источника отключает
    именно его до конца текущего запуска (см. _stress_provider_rate_limited),
    вместо того чтобы продолжать получать 429 на каждом следующем слове."""
    errors = []
    for name, provider in _STRESS_ONLINE_PROVIDERS:
        if name in _stress_provider_rate_limited:
            continue
        try:
            found = provider(word, timeout=timeout)
        except _StressSourceRateLimited as e:
            _stress_provider_rate_limited.add(name)
            print(f"Источник «{name}» ответил 429 (слишком много запросов, {e}) — "
                  "отключаю его до конца текущего запуска, использую другие источники.")
            continue
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if found:
            return found
    if errors:
        print(f"Не удалось найти ударение для «{word}» в интернете "
              f"({'; '.join(errors)}) — пропускаю, продолжаю без него.")
    return None


def enrich_stress_dictionary_online(text: str, stress_dict: dict,
                                     dict_path: "Path | str | None" = None,
                                     enabled: bool = True) -> None:
    """Для слов из text, которых ещё нет в stress_dict (и для которых
    онлайн-поиск ещё не проваливался в этом запуске), пробует найти
    ударение в интернете (fetch_stress_online). Найденное сразу
    добавляется в stress_dict (в памяти — используется тем же вызовом
    apply_stress_dictionary дальше) и дописывается в JSON-файл словаря
    dict_path (по умолчанию DEFAULT_STRESS_DICT_PATH), чтобы при
    следующих запусках слово уже не требовало обращения к сети.

    Ничего не бросает: нет пакета requests, нет сети, Викисловарь лежит,
    файл словаря нельзя прочитать/записать — во всех случаях просто
    печатается предупреждение и функция возвращается, ничего не сломав."""
    if not enabled:
        return
    try:
        ambiguous = _load_ambiguous_stress_words()
        words_to_try = []
        seen = set()
        for m in _STRESS_WORD_RE.finditer(text):
            word = m.group(0)
            key = word.lower()
            if (key in stress_dict or key in _online_stress_failed_cache
                    or key in seen or key in ambiguous):
                continue
            seen.add(key)
            words_to_try.append(word)
        if not words_to_try:
            return

        new_entries = {}
        for word in words_to_try:
            found = fetch_stress_online(word)
            if found:
                new_entries[word.lower()] = found.lower()
            else:
                _online_stress_failed_cache.add(word.lower())

        if not new_entries:
            return
        stress_dict.update(new_entries)

        p = Path(dict_path) if dict_path else DEFAULT_STRESS_DICT_PATH
        try:
            existing = {}
            if p.exists():
                existing = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            existing.update(new_entries)
            p.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Словарь ударений пополнен из интернета: {len(new_entries)} "
                  f"слов(о) добавлено в {p}")
        except Exception as e:
            print(f"Нашёл ударения в интернете для {len(new_entries)} слов(а), но не "
                  f"смог сохранить их в {p}: {e} — они будут использоваться только в "
                  "этом запуске.")
    except Exception as e:
        # Подстраховка на случай непредвиденной ошибки в самой функции —
        # онлайн-словарь ударений вспомогательный, он не должен ронять
        # синтез книги.
        print(f"Онлайн-поиск ударений не удался ({type(e).__name__}: {e}) — "
              "продолжаю без него.")


def _sane_yo(text: str) -> str:
    """"ё" в русском языке всегда ударная. Если в ответе акцентора стоит "ё", но САМ он
    её "+" не пометил (внутреннее противоречие), такую неподтверждённую "ё" возвращаем
    обратно в "е": "е" вместо "ё" на письме — обычное дело, а лишняя услышанная "ё"
    очень заметна на слух, так что при сомнении лучше промолчать про "ё"."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch not in ("ё", "Ё"):
            continue
        prev = chars[i - 1] if i > 0 else ""
        if prev != "+":
            chars[i] = "е" if ch == "ё" else "Е"
    return "".join(chars)


_silero_stress_instance = None
_silero_stress_load_failed = False


def _load_silero_stress_phrases(accentor) -> None:
    """Контекстные правила омографов silero-stress 1.5 (silero_stress_phrases.json: слово -> {фраза: форма слова}).
    Приоритет выше нейросети; ошибка в файле не мешает работе (ключи с "_" — комментарии)."""
    try:
        import tempfile
        path = _app_paths.resolve_asset("silero_stress_phrases.json")
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        clean = {w: v for w, v in data.items() if not w.startswith("_") and isinstance(v, dict) and v}
        if not clean:
            return
        tmp = Path(tempfile.gettempdir()) / "silero_stress_phrases_clean.json"
        tmp.write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
        accentor.homosolver.load_phrases_dict(str(tmp))
    except Exception as e:
        print(f"silero_stress_phrases.json пропущен ({type(e).__name__}: {e})")


def _load_silero_stress():
    """Лениво загружает и кэширует silero-stress — единственный акцентор конвейера
    (https://github.com/snakers4/silero-stress; разрешает омографы по контексту).
    Вывод — "+" перед ударной гласной, поэтому словарь, _sane_yo и грамматический
    шлюз работают с ним напрямую. При ошибке (пакет не установлен) возвращает None,
    один раз печатает предупреждение и больше не пытается."""
    global _silero_stress_instance, _silero_stress_load_failed
    if _silero_stress_instance is not None or _silero_stress_load_failed:
        return _silero_stress_instance
    try:
        from silero_stress import load_accentor
        _silero_stress_instance = load_accentor()
        _load_silero_stress_phrases(_silero_stress_instance)
        print("silero-stress загружен.")
    except Exception as e:
        _silero_stress_load_failed = True
        print(f"silero-stress недоступен ({e}) — автоматических ударений не будет, "
              "остаются только словарь и встроенный put_accent Silero. "
              "Чтобы включить: pip install silero-stress")
        _silero_stress_instance = None
    return _silero_stress_instance


def reload_stress_models() -> None:
    """Drops every cached stress state (context tail, OOV tally, fallback counters) so the next
    text is processed from scratch with the CURRENT dictionaries. Models stay loaded."""
    reset_stress_context()
    _oov_words.clear()
    for k in list(_fallback_stats):
        _fallback_stats[k] = 0
    try:
        import grammar_gate
        if hasattr(grammar_gate, "clear_caches"):
            grammar_gate.clear_caches()
    except Exception:
        pass


def set_stress_review(enabled: bool) -> None:
    """GUI/run_* switch for the review list (review_list.json)."""
    global STRESS_REVIEW_ENABLED
    STRESS_REVIEW_ENABLED = bool(enabled)


def _mark_yo_stress(text: str, start: int, word: str) -> "str | None":
    """"ё" is always stressed: returns the word with "+" in front of it
    ("подошёл" -> "подош+ёл"), or None when nothing should be changed. The mark
    also makes _sane_yo leave the word alone (an "ё" without "+" in front of it is
    rolled back to "е" by _sane_yo)."""
    low = word.lower()
    if low.count("ё") != 1 or (start > 0 and text[start - 1] == "+"):
        return None
    i = low.index("ё")
    return word[:i] + "+" + word[i:]


# --------------------------------------------------------------------------
# Review list -> review_list.json
# --------------------------------------------------------------------------
# silero-stress accentuates the whole chapter once more; every word that carries no mark
# afterwards ("unmarked": names, neologisms) or that no dictionary knows ("oov") is counted
# over the whole book and dumped, most frequent first, into review_list.json. Words that the
# user's dictionaries already decide are not reported. Fix a reported word once in the
# "Словарь ударений" tab (or by adding {"слово": "сл+ово"} to forced_stress_words.json).
# Known context-dependent homographs are listed separately ("homographs").
# Disable: --no-stress-review or AUDIOBOOK_STRESS_REVIEW=0;
# other location: AUDIOBOOK_REVIEW_LIST=<path>.

STRESS_REVIEW_ENABLED = os.environ.get("AUDIOBOOK_STRESS_REVIEW", "1") != "0"
STRESS_REVIEW_PATH = Path(os.environ.get("AUDIOBOOK_REVIEW_LIST", "")
                          or _app_paths.user_file("review_list.json"))
_OUT_WORD_RE = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")
_stress_review_state: "dict | None" = None


def reset_stress_review(label: str = "") -> None:
    """Starts a fresh tally (call once at the start of a book run)."""
    global _stress_review_state
    _stress_review_state = {"label": label, "chapters": [], "words": {}, "totals": {}}


def _align_plain(src_plain: "list[str]", out_tokens: "list[str]") -> dict:
    """{source token index: output token index} for tokens that line up.
    Robust to silero-stress dropping/merging a token (difflib on the plain words)."""
    import difflib
    out_plain = [t.replace("+", "").lower().replace("ё", "е") for t in out_tokens]
    sm = difflib.SequenceMatcher(a=src_plain, b=out_plain, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping


def collect_stress_review(text: str, resolved_words=None, chapter_label: str = "", log_fn=None) -> int:
    """Adds one chapter to the running tally. Returns the number of flagged occurrences.
    Never raises."""
    log = log_fn or (lambda m: None)
    if not STRESS_REVIEW_ENABLED or not text or not text.strip():
        return 0
    try:
        if _stress_review_state is None:
            reset_stress_review()
        st = _stress_review_state
        if chapter_label and chapter_label in st["chapters"]:
            return 0
        ss = _load_silero_stress()
        if ss is None:
            return 0
        import time
        t0 = time.time()
        resolved = resolved_words or ()
        known_sources = []
        corpus = _load_offline_stress_corpus()
        if corpus:
            known_sources.append(corpus)

        chunks, cur, n = [], [], 0
        for p in text.split("\n"):
            cur.append(p)
            n += len(p) + 1
            if n >= 3000:
                chunks.append("\n".join(cur))
                cur, n = [], 0
        if cur:
            chunks.append("\n".join(cur))

        flagged = 0
        vowels_re = re.compile(r"[аеёиоуыэюя]")
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                ss_toks = _OUT_WORD_RE.findall(ss(chunk))
            except Exception:
                ss_toks = None
            src = [(m.group(0), m.start()) for m in _STRESS_WORD_RE.finditer(chunk)]
            src_plain = [t.lower().replace("ё", "е") for t, _ in src]
            ss_map = _align_plain(src_plain, ss_toks) if ss_toks else {}
            for i, (tok, pos) in enumerate(src):
                low = tok.lower()
                if len(vowels_re.findall(low)) <= 1 or "ё" in low:
                    continue
                st["totals"][low] = st["totals"].get(low, 0) + 1
                if low in resolved:
                    continue
                ss_form = ss_toks[ss_map[i]].lower() if i in ss_map else None
                reasons = []
                if ss_form is not None and "+" not in ss_form:
                    reasons.append("unmarked")
                if known_sources and "-" not in low and not any(low in d for d in known_sources):
                    reasons.append("oov")
                if not reasons:
                    continue
                flagged += 1
                e = st["words"].setdefault(low, {"flagged": 0, "reasons": set(),
                                                  "silero_stress": {}, "examples": []})
                e["flagged"] += 1
                e["reasons"].update(reasons)
                if ss_form is not None:
                    e["silero_stress"][ss_form] = e["silero_stress"].get(ss_form, 0) + 1
                if len(e["examples"]) < 3:
                    ctx = chunk[max(0, pos - 40):pos + len(tok) + 40].replace("\n", " ").strip()
                    e["examples"].append({"chapter": chapter_label, "context": ctx})
        if chapter_label:
            st["chapters"].append(chapter_label)
        log(f"stress review: {flagged} flagged occurrence(s), {len(st['words'])} distinct word(s) so far "
            f"[{time.time() - t0:.0f}s]")
        return flagged
    except Exception as e:
        log(f"stress review failed ({type(e).__name__}: {e}) — skipped, synthesis is not affected.")
        return 0


def dump_stress_review(path: "Path | str | None" = None) -> "Path | None":
    """Writes the running tally to review_list.json (atomically, most frequent
    words first). Never raises; returns the path or None."""
    st = _stress_review_state
    if st is None:
        return None
    p = Path(path) if path else STRESS_REVIEW_PATH
    try:
        homographs = _load_ambiguous_stress_words()
        entries = []
        for low, e in st["words"].items():
            entries.append({
                "word": low,
                "occurrences": st["totals"].get(low, e["flagged"]),
                "flagged_occurrences": e["flagged"],
                "reasons": sorted(e["reasons"]),
                "silero_stress": dict(sorted(e["silero_stress"].items(), key=lambda kv: -kv[1])),
                "known_homograph": low in homographs,
                "examples": e["examples"],
            })
        entries.sort(key=lambda x: (-x["occurrences"], x["word"]))
        words = [x for x in entries if not x["known_homograph"]]
        homos = [x for x in entries if x["known_homograph"]]

        paste_ready = {}
        for x in words:
            sug = next((f for f in x.get("silero_stress", {}) if f.count("+") == 1), None)
            if sug:
                x["suggested"] = sug
                paste_ready[x["word"]] = sug
        payload = {
            "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "book": st.get("label", ""),
            "chapters_processed": st["chapters"],
            "summary": {"distinct_words": len(words), "distinct_homographs": len(homos),
                        "flagged_occurrences": sum(x["flagged_occurrences"] for x in entries),
                        "silero_applied": dict(_fallback_stats)},
            "how_to_use": ("1) Open review_list_paste.txt (same folder): one ready line per word, most "
                           "frequent first. 2) Delete the lines whose stress is wrong, fix them by hand "
                           "(the '+' goes BEFORE the stressed vowel). 3) Paste the lines right after the "
                           "opening { of forced_stress_words.json - or fix the words in the 'Словарь "
                           "ударений' tab of the app. Words in that file are never touched by silero-stress. "
                           "Reasons: 'unmarked' = silero-stress put no mark; 'oov' = in no dictionary. "
                           "'homographs' depend on context -> informational."),
            "paste_ready": paste_ready,
            "unresolved_oov": dict(sorted(_oov_words.items(), key=lambda kv: -kv[1])[:500]),
            "words": words,
            "homographs": homos,
        }
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        try:   # copy-paste friendly twin: every line ends with a comma -> paste right after "{"
            txt = "\n".join(f'  {json.dumps(w, ensure_ascii=False)}: {json.dumps(f, ensure_ascii=False)},'
                            for w, f in paste_ready.items())
            tp = p.with_name(p.stem + "_paste.txt")
            tp.write_text(txt + ("\n" if txt else ""), encoding="utf-8")
        except Exception as e:
            print(f"Could not write the paste list: {e}")
        return p
    except Exception as e:
        print(f"Could not write {p}: {e}")
        return None


def _stall_guard_start(label: str, seconds: float = 180.0):
    """Hang diagnostics: if the code between start and stop takes longer than `seconds`, prints where every thread
    is stuck (repeating every `seconds`, at most 5 times) so a "frozen" run leaves a stack trace in the log
    instead of silence."""
    import threading
    import traceback
    stop = threading.Event()

    def watch():
        for n in range(1, 6):
            if stop.wait(seconds):
                return
            lines = [f"!! STALL: '{label}' has been running for {n * seconds:.0f}s - thread stacks follow"]
            for tid, frame in sys._current_frames().items():
                lines.append(f"--- thread {tid} ---\n" + "".join(traceback.format_stack(frame)))
            print("\n".join(lines))

    threading.Thread(target=watch, name="stall-guard", daemon=True).start()
    return stop


def _stall_guard_stop(guard) -> None:
    if guard is not None:
        guard.set()


def run_stress_review_for_chapter(text: str, stress_dict: dict, title: str, log_fn=print) -> None:
    """Chapter hook used by run_silero / run_silero_rest: tally, then dump."""
    if not STRESS_REVIEW_ENABLED:
        return
    collect_stress_review(text, resolved_words=set(stress_dict or ()), chapter_label=title, log_fn=log_fn)
    out = dump_stress_review()
    if out is not None:
        log_fn(f"stress review list updated: {out}")


# ==========================================================================
# Accentuation chain (single engine)
#   1. forced_stress_words.json / manual overrides - the user's words, absolute priority,
#                 hidden behind placeholders so nothing downstream can change them
#   2. numbers -> words
#   3. silero-stress over the chunk (with the previous chunk's tail as left context)
#   4. grammar gate (slovnet morphology) for case/number homographs
#   5. global_prose_sanitizer (context-free forced forms, particles) -> mono-syllable strip
# ==========================================================================

_STRIP_MONO_ENABLED = True


def set_strip_monosyllables(enabled: bool) -> None:
    """Switches strip_monosyllable_stress() on/off for this run (GUI option)."""
    global _STRIP_MONO_ENABLED
    _STRIP_MONO_ENABLED = bool(enabled)


_CYR_VOWEL_RE = re.compile(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]")
_PLACEHOLDER_SPLIT_RE = re.compile(r"STRESSDICT\d+")

# --- context priming (October 2026) ---------------------------------------------------------------------
# silero-stress resolves homographs from a ~300-character window, but the pipeline hands it one chunk at a
# time, so the first words of a chunk have no left context ("Они все побежали" alone -> "всё"; after a
# sentence about people running for the exits -> "все"). The tail of the PREVIOUS chunk is therefore
# prepended to what silero-stress reads, and only the current chunk's words are kept from its answer.
# Off: AUDIOBOOK_STRESS_CONTEXT=0. Reset per chapter by reset_stress_context().
_STRESS_CTX_ON = os.environ.get("AUDIOBOOK_STRESS_CONTEXT", "1") != "0"
_STRESS_CTX_CHARS = 220
_stress_ctx_tail = ""


def reset_stress_context() -> None:
    global _stress_ctx_tail
    _stress_ctx_tail = ""


def _stress_ctx_update(src_text: str) -> None:
    """Remember the plain tail (no marks, no dictionary placeholders) of the chunk just processed."""
    global _stress_ctx_tail
    if not _STRESS_CTX_ON or not src_text:
        return
    plain = re.sub(r"\s+", " ", _PLACEHOLDER_SPLIT_RE.sub(" ", src_text.replace("+", ""))).strip()
    if not re.search(r"[А-Яа-яЁё]", plain):
        return
    tail = plain[-_STRESS_CTX_CHARS:]
    if len(plain) > _STRESS_CTX_CHARS:
        cut = tail.find(" ")
        tail = tail[cut + 1:] if cut >= 0 else tail
    _stress_ctx_tail = tail


# silero-stress 1.5 shifts the "+" LEFT by one position per punctuation char glued to the front of a word
# («Пора -> «По+ра, «—Пора -> «—П+ора; a space after the mark removes the shift; "-" and "(" are not affected).
# Found in the real-book audio test (10 Oct 2026): "с+мирно", "+Пола", "По+ра" and marks lost on vowel-initial words.
_GLUED_PUNCT_RE = re.compile(r"([^\w\s+\-(])(?=[А-Яа-яЁё])")


def _unglue_leading_punct(text: str) -> str:
    return _GLUED_PUNCT_RE.sub(r"\1 ", text)


def _silero_stress_primed(accentor, src_text: str) -> str:
    """accentor(src_text), but with the previous chunk's tail as left context. The returned text starts at
    the first word of src_text. Any mismatch -> the plain unprimed call."""
    ctx = _stress_ctx_tail if _STRESS_CTX_ON else ""
    if not ctx:
        return accentor(_unglue_leading_punct(src_text))
    try:
        out = accentor(_unglue_leading_punct(ctx + " " + src_text))
        ctx_words = [w.lower().replace("ё", "е") for w in _STRESS_WORD_RE.findall(ctx)]
        ms = list(_OUT_WORD_RE.finditer(out))
        k = len(ctx_words)
        if len(ms) > k and [m.group(0).replace("+", "").lower().replace("ё", "е") for m in ms[:k]] == ctx_words:
            _fallback_stats["ctx_primed"] = _fallback_stats.get("ctx_primed", 0) + 1
            return out[ms[k].start():]
    except Exception:
        pass
    return accentor(_unglue_leading_punct(src_text))


_fallback_stats = {"blank": 0, "applied": 0, "yo_swapped": 0}
_oov_words: dict = {}   # >=2-syllable words left clean by every layer (logged to review_list.json)

_VOWELS_LOW_RE = re.compile(r"[аеёиоуыэюя]")


def _copy_case(src: str, marked: str) -> str:
    """The marked form with the capitalisation of the source word ("+" is skipped when looking for the
    first letter). silero-stress already keeps the case; this makes the layer independent of that."""
    if marked.replace("+", "") == src.replace("ё", "е").replace("Ё", "Е") or marked.replace("+", "") == src:
        return marked
    if len(src) > 1 and src.isupper():
        return marked.upper()
    if src[:1].isupper():
        chars = list(marked)
        for i, ch in enumerate(chars):
            if ch != "+":
                chars[i] = ch.upper()
                break
        return "".join(chars)
    return marked


# silero-stress ошибочно отдаёт «ё» в основах, где её нет ("королевой" -> "корол+ёвой"): метка остаётся на «е».
_YO_FALSE_STEM_RE = re.compile(r"корол\+?ёв", re.I)


def apply_silero_layer(src_text: str, silero_accentor) -> str:
    """The accentor layer: silero-stress marks every word of src_text it knows. Only the plain letters
    of src_text are ever kept (a word is replaced by silero's form only when both spell the same word
    modulo "+" and е/ё), so a misaligned or rewritten answer cannot damage the text. "ё" returned by
    silero-stress ("все" -> "вс+ё") is taken; the numeric "лет"->"лёт" case is refused. Words that are
    still unmarked afterwards (>= 2 vowels) are counted as OOV for review_list.json."""
    if silero_accentor is None or not src_text or not re.search(r"[А-Яа-яЁё]", src_text):
        return src_text
    src = list(_STRESS_WORD_RE.finditer(src_text))
    if not src:
        return src_text
    try:
        ss_out = _silero_stress_primed(silero_accentor, src_text)
    except Exception as e:
        print(f"silero-stress failed on a fragment ({e}) — fragment left unaccented.")
        return src_text
    ssm = list(_OUT_WORD_RE.finditer(ss_out))
    src_plain = [m.group(0).lower().replace("ё", "е") for m in src]
    ss_map = _align_plain(src_plain, [m.group(0) for m in ssm]) if ssm else {}
    pieces, last = [], 0
    for i, m in enumerate(src):
        word = m.group(0)
        low = word.lower()
        if "ё" in low or (m.start() > 0 and src_text[m.start() - 1] == "+"):
            continue
        sf = _sane_yo(ssm[ss_map[i]].group(0)) if i in ss_map else None
        if sf and _YO_FALSE_STEM_RE.match(sf):
            sf = sf.replace("ё", "е").replace("Ё", "Е")
        if (sf and "+" in sf
                and sf.replace("+", "").replace("ё", "е").lower() == low
                and not (low == "лет" and "ё" in sf)):
            sf = _copy_case(word, sf)
            pieces.append(src_text[last:m.start()])
            pieces.append(sf)
            last = m.end()
            _fallback_stats["applied"] += 1
            if "ё" in sf.lower() and "е" in low:
                _fallback_stats["yo_swapped"] = _fallback_stats.get("yo_swapped", 0) + 1
        elif len(_VOWELS_LOW_RE.findall(low)) >= 2:
            _oov_words[low] = _oov_words.get(low, 0) + 1
    pieces.append(src_text[last:])
    return "".join(pieces)


_warned_legacy_engines: set = set()


def normalize_stress_engine(name) -> str:
    """The stress flow has two modes: "silero" (silero-stress + grammar gate, the default) and
    "none". Old settings / CLI values ('ruaccent', 'silero_stress', 'hybrid', 'ollama',
    'lmstudio') map to "silero"."""
    n = (name or "silero").strip().lower()
    if n in ("silero", "none"):
        return n
    if n not in ("ruaccent", "silero_stress") and n not in _warned_legacy_engines:
        _warned_legacy_engines.add(n)
        print(f"Stress engine {n!r} is no longer supported — using silero-stress + grammar gate.")
    return "silero"


def stress_engine_needs_silero(stress_engine: str) -> bool:
    return stress_engine != "none"


def stress_layers_are_client_side(stress_engine: str, silero_accentor) -> bool:
    """True when the accent marks are produced in the client before the text is sent
    (REST path): the service then only synthesizes the marked text."""
    return stress_engine != "none" and silero_accentor is not None


# ---------------------------------------------------------------------------
# global_prose_sanitizer: the last stress step before the text goes to Silero. Fixed forms for
# words that the accentor keeps getting wrong in fiction (found in live chapter tests), plus the
# hyphenated-particle rule. A plain table, no LLM. The user's own dictionary
# (forced_stress_words.json) still wins: those words are hidden behind placeholders while this runs.
# ---------------------------------------------------------------------------
_PROSE_WORDS = {
    "могущий": "мог+ущий", "могущая": "мог+ущая", "могущие": "мог+ущие", "могущее": "мог+ущее",
    "губы": "г+убы",
    "глубоко": "глубок+о",
    "самому": "самом+у",
    "поняла": "понял+а",
    "такая": "так+ая",
    "также": "т+акже",
    # words with ONE reading where silero-stress is wrong. Every entry must agree with the independent
    # offline corpus (stress_corpus_ru.tsv.gz): test_accentuation.py enforces it, except _PROSE_CORPUS_EXCEPTIONS.
    # (Removed Oct 10 2026: импульсный -> и́мпульсный, жизнеобеспечение -> жизнеобесп+ечение, уровнях -> +уровнях:
    # the old rules contradicted the corpus AND the raw model, i.e. they CREATED errors.)
    "молча": "молч+а",
}
# context-dependent readings pinned by the user after listening; the corpus lists only one reading, so they differ on purpose
_PROSE_CORPUS_EXCEPTIONS = {"могущий", "могущая", "могущие", "могущее", "самому"}
_PROSE_CH = "А-Яа-яЁё"


def _prose_pattern(phrase: str):
    words = [r"\+?".join(re.escape(ch) for ch in w) + r"\+?" for w in phrase.split()]
    return re.compile(r"(?<![" + _PROSE_CH + r"])" + r"\+?" + r"\s+".join(words) + r"(?![" + _PROSE_CH + r"])", re.I)


_PROSE_RULES = [(_prose_pattern(k), v) for k, v in _PROSE_WORDS.items()]

# Hyphenated particles: "когда-нибудь", "кто-то", "всё-таки", "дай-ка" carry ONE stress (in the
# host word). A second "+" inside the particle is always wrong.
_HYPHEN_PARTICLES = ("нибудь", "либо", "то", "таки", "ка")
_PARTICLE_TOKEN_RE = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)+")


def strip_particle_stress(text: str) -> str:
    """Removes a "+" that landed inside a trailing hyphenated particle."""
    if not text or "-" not in text or "+" not in text:
        return text

    def _fix(m):
        parts = m.group(0).split("-")
        if len(parts) >= 2 and parts[-1].replace("+", "").lower() in _HYPHEN_PARTICLES and "+" in parts[-1]:
            parts[-1] = parts[-1].replace("+", "")
        return "-".join(parts)

    return _PARTICLE_TOKEN_RE.sub(_fix, text)


_NUM_LET_RE = re.compile(r"(\d\s*)л\+?ёт(?![" + _PROSE_CH + r"])")


def global_prose_sanitizer(text: str) -> str:
    """Forces the known-good stress for words the accentor misplaces ("самому" -> "самом+у",
    "поняла" -> "понял+а", "губы" -> "г+убы"...), strips stray marks from hyphenated particles and
    undoes a bad "ё" after a number ("10 лёт" -> "10 лет")."""
    if not text:
        return text
    for pat, form in _PROSE_RULES:
        def _sub(m, form=form):
            return (form[0].upper() + form[1:]) if m.group(0).lstrip("+")[:1].isupper() else form
        text = pat.sub(_sub, text)
    text = strip_particle_stress(text)
    return _NUM_LET_RE.sub(r"\1лет", text)


def _grammar_gate_pass(text: str) -> str:
    """Grammar gate (grammar_gate.py): the slovnet morphology tagger decides case/number stress
    homographs (земли, сети, трети, голоса, корпуса...). Off when AUDIOBOOK_GRAMMAR_GATE=0 or when
    slovnet/navec/razdel or the model files are missing (it then does nothing and never raises)."""
    try:
        import grammar_gate
        return grammar_gate.grammar_gate(text)
    except Exception as e:
        if not getattr(_grammar_gate_pass, "_warned", False):
            _grammar_gate_pass._warned = True
            print(f"grammar gate skipped ({type(e).__name__}: {e})")
        return text

_GUARD_FOLD_RE = re.compile(r"[^0-9a-zа-я]")


def _guard_fold(t: str) -> str:
    """Letters and digits only, lower case, ё == е, no '+': what must survive every stress layer."""
    return _GUARD_FOLD_RE.sub("", t.lower().replace("ё", "е"))


def _text_guard(before: str, after: str, layer: str) -> str:
    """Runtime form of the book test «output = input modulo '+' and е/ё»: if a stress layer changed,
    dropped or added a letter, keep that layer's INPUT (the chunk loses only its marks, never words;
    a lost word is a silent hole in the audio). Counted in _fallback_stats["text_guard"]."""
    if after is before or os.environ.get("AUDIOBOOK_TEXT_GUARD") == "0" or _guard_fold(before) == _guard_fold(after):
        return after
    _fallback_stats["text_guard"] = _fallback_stats.get("text_guard", 0) + 1
    if _fallback_stats["text_guard"] <= 20:
        print(f"stress text guard: layer {layer} changed the letters, layer skipped for this chunk: {before[:70]!r}")
    return before


def apply_stress_layers(text: str, stress_engine: str, silero_accentor) -> str:
    """silero-stress (with the previous chunk as left context) -> grammar gate -> sanitizer.
    Dictionary words are already hidden behind STRESSDICT placeholders."""
    if stress_engine == "none":
        return text
    out = _text_guard(text, apply_silero_layer(text, silero_accentor), "silero") if silero_accentor is not None else text
    _stress_ctx_update(text)
    out = _text_guard(out, _grammar_gate_pass(out), "gate")
    return _text_guard(out, global_prose_sanitizer(out), "sanitizer")


def stress_dry_run(text: str, strip_monosyllables: bool = True, book_overrides_path=None,
                   progress_cb=None) -> "tuple[str, dict]":
    """GUI «Проверить ударения»: runs the SAME chain the audio gets, without synthesis:
    digits -> words, user dictionary (placeholders) -> silero-stress -> grammar gate ->
    global_prose_sanitizer -> mono-syllable strip. Returns (text with "+" marks, info)."""
    reset_stress_context()
    import time as _time
    t0 = _time.time()
    silero = _load_silero_stress()
    set_keep_hyphens(silero is not None)
    stress_dict = load_stress_dictionary(book_overrides_path=book_overrides_path)
    set_strip_monosyllables(strip_monosyllables)
    swaps0 = _fallback_stats.get("yo_swapped", 0)
    blanks0 = _fallback_stats.get("applied", 0)
    paragraphs = numbers_to_words_ru(text).split("\n")
    todo = sum(1 for p in paragraphs if p.strip())
    out, done = [], 0
    for para in paragraphs:
        if not para.strip():
            out.append(para)
            continue
        protected, ph = apply_stress_dictionary_protected(para, stress_dict)
        marked = apply_stress_layers(protected, "silero", silero)
        out.append(strip_monosyllable_stress(restore_stress_placeholders(marked, ph)))
        done += 1
        if progress_cb and done % 10 == 0:
            progress_cb(f"  …{done}/{todo} абзацев")
    result = "\n".join(out)
    info = {
        "silero_stress": silero is not None,
        "dictionary_words": len(stress_dict),
        "yo_swapped": _fallback_stats.get("yo_swapped", 0) - swaps0,
        "silero_filled": _fallback_stats.get("applied", 0) - blanks0,
        "marks": result.count("+"),
        "seconds": _time.time() - t0,
    }
    return result, info




def find_word_occurrences(text: str, query: str, whole_word: bool = True) -> "list[tuple[int, int]]":
    """[(start, end)] of every occurrence of `query` in `text` - case-insensitive, "ё" == "е", any "+"
    in the query ignored. whole_word=True matches only complete words (a hyphenated word is one
    word). Indices refer to the ORIGINAL text (the folding keeps the length)."""
    q = (query or "").replace("+", "").strip().lower().replace("ё", "е")
    if not q or not text:
        return []
    hay = text.lower().replace("ё", "е")
    out, i = [], 0
    while True:
        i = hay.find(q, i)
        if i < 0:
            break
        j = i + len(q)
        if whole_word:
            before = hay[i - 1] if i > 0 else ""
            after = hay[j] if j < len(hay) else ""
            glued_l = before.isalpha() or (before == "-" and i > 1 and hay[i - 2].isalpha())
            glued_r = after.isalpha() or (after == "-" and j + 1 < len(hay) and hay[j + 1].isalpha())
            if glued_l or glued_r:
                i += 1
                continue
        out.append((i, j))
        i = j
    return out


def stress_word_variants(text: str, word: str, silero_accentor=None, limit: int = 200,
                         left: int = 320, right: int = 120) -> "list[dict]":
    """What the pipeline (silero-stress with its left context -> grammar gate -> sanitizer, WITHOUT the
    user's dictionary) makes of EVERY occurrence of `word` in `text`. Used by the dictionary editor's
    homograph alert: several different forms for one spelling means the word is context-dependent
    in this chapter, and a forced dictionary entry would flatten all of them to one form.
    Returns [{"pos": offset in text, "form": "вс+ё", "context": "..."}], at most `limit` entries;
    an occurrence the models could not align is skipped."""
    acc = silero_accentor if silero_accentor is not None else _load_silero_stress()
    if acc is None:
        return []
    res = []
    for s_, e_ in find_word_occurrences(text, word)[:limit]:
        a = max(0, s_ - left)
        if a > 0:                       # start the window at a word boundary
            sp = text.find(" ", a, s_)
            a = sp + 1 if sp != -1 else a
        b = min(len(text), e_ + right)
        window = text[a:b]
        k = len(_STRESS_WORD_RE.findall(text[a:s_]))
        try:
            marked = global_prose_sanitizer(_grammar_gate_pass(apply_silero_layer(window, acc)))
        except Exception:
            continue
        toks = _OUT_WORD_RE.findall(marked)
        src = _STRESS_WORD_RE.findall(window)
        if len(toks) != len(src) or k >= len(toks):
            continue
        form = toks[k]
        if form.replace("+", "").lower().replace("ё", "е") != text[s_:e_].lower().replace("ё", "е"):
            continue
        ctx = text[max(0, s_ - 60):e_ + 60].replace("\n", " ")
        res.append({"pos": s_, "form": form.lower(), "context": ctx})
    return res


def _parse_stress_reply(raw_text: str, n: int) -> dict:
    """Ответ модели -> {номер_с_0: строка}. Понимает JSON-объект {"1": ..},
    JSON-массив (по порядку) и мусор вокруг; частичный ответ — не ошибка."""
    import json as _json
    res = {}
    m = re.search(r"\{.*\}", raw_text or "", flags=re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        i = int(re.sub(r"\D", "", str(k))) - 1
                    except ValueError:
                        continue
                    if 0 <= i < n and isinstance(v, (str, int)):
                        res[i] = str(v)
                return res
        except ValueError:
            pass
    arr = _extract_json_array(raw_text or "")
    if arr and len(arr) == n:  # массив принимаем только если длина сошлась
        res = {i: str(v) for i, v in enumerate(arr)}
    return res



def _emphasis_kind(part_text: str) -> "str | None":
    """По завершающему знаку куска текста определяет, нужно ли усилить
    интонацию — "question" для "?", "exclaim" для "!". Возвращает None,
    если куск не заканчивается одним из этих знаков (например, это
    середина предложения, отрезанная по запятой/тире)."""
    stripped = part_text.rstrip()
    if stripped.endswith("?"):
        return "question"
    if stripped.endswith("!"):
        return "exclaim"
    return None


def apply_emphasis(audio: "np.ndarray", sample_rate: int, kind: "str | None") -> "np.ndarray":
    """Лёгкое усиление интонации на уровне уже готового звука — локальный
    режим silero (в отличие от silero_rest) не принимает SSML/<prosody>, но
    вопросительные и восклицательные фразы всё равно должны звучать не
    совсем ровно. Здесь используется простое (без дополнительных
    зависимостей вроде librosa) изменение скорости проигрывания через
    ресемплинг: небольшое ускорение "поднимает" тон вверх (классический
    эффект "бурундука", в малой дозе не режет слух) — так же в сторону
    "выше и быстрее" двигает интонацию <prosody pitch="high"[ rate="fast"]>
    в text_to_ssml() для silero_rest. Восклицание ускоряется заметнее
    вопроса, как там же.

    Это приближение, а не настоящий питч-шифтер (меняется и высота, и
    темп вместе) — но для коротких вопросительных/восклицательных фраз
    разница на слух небольшая, а плюс к выразительности заметный."""
    import numpy as np
    if kind is None or audio.size == 0:
        return audio
    speed = 1.05 if kind == "question" else 1.10
    new_len = max(1, int(round(audio.shape[0] / speed)))
    old_idx = np.linspace(0, audio.shape[0] - 1, num=audio.shape[0])
    new_idx = np.linspace(0, audio.shape[0] - 1, num=new_len)
    return np.interp(new_idx, old_idx, audio).astype(audio.dtype)


def apply_fade(audio: "np.ndarray", sample_rate: int, fade_ms: float = 6.0) -> "np.ndarray":
    """Плавный fade-in/fade-out (полу-косинус) на самом краю фрагмента —
    убирает щелчки/потрескивание на стыках при склейке кусков через
    np.concatenate. Модель почти никогда не начинает и не заканчивает
    ровно с нулевой амплитуды, поэтому прямая склейка "конец одного куска
    впритык к началу следующего" даёт резкий скачок амплитуды — на слух
    это как раз и звучит как "шум"/потрескивание на границах фраз, а не
    как гладкая речь. Фрагменты со вставленной тишиной (retry-fallback)
    фейдить не нужно — там и так тишина."""
    import numpy as np
    n = audio.shape[0]
    fade_len = min(n // 2, int(sample_rate * fade_ms / 1000))
    if fade_len <= 1:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_len))
    audio = audio.copy()
    audio[:fade_len] *= ramp
    audio[-fade_len:] *= ramp[::-1]
    return audio


# --------------------------------------------------------------------------
# silero-vad (https://github.com/snakers4/silero-vad) — тот же автор, что и
# у Silero TTS, ~2МБ модель, без новых тяжёлых зависимостей (тот же torch).
# Точно находит, где в куске реально начинается/заканчивается речь —
# используется, чтобы обрезать реальную тишину по краям КАЖДОГО куска
# перед склейкой (точнее, чем фиксированный fade_ms в apply_fade, который
# не знает, есть ли внутри этих 6мс уже начавшаяся речь или ещё тишина),
# и чтобы точнее ловить "пустые" фрагменты — сейчас пустота определяется
# грубо, по максимальной громкости (см. cosyvoice-часть), а VAD реально
# ищет речь, а не просто "не тишина ли это".
# --------------------------------------------------------------------------

_vad_model = None
_vad_get_speech_timestamps = None
_vad_load_failed = False


def _load_silero_vad():
    """Лениво загружает и кэширует silero-vad. При любой ошибке (пакет не
    установлен, модель не грузится) возвращает (None, None) один раз,
    печатает предупреждение и больше не пытается — остальная озвучка
    продолжается без VAD-обрезки, на одном fade по фиксированному времени
    (см. apply_fade)."""
    global _vad_model, _vad_get_speech_timestamps, _vad_load_failed
    if _vad_model is not None or _vad_load_failed:
        return _vad_model, _vad_get_speech_timestamps
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        _vad_model = load_silero_vad()
        _vad_get_speech_timestamps = get_speech_timestamps
        print("silero-vad загружен — границы речи в кусках будут обрезаться точно, "
              "а не по фиксированному времени.")
    except Exception as e:
        _vad_load_failed = True
        print(f"silero-vad недоступен ({e}) — обрезка тишины по границам речи отключена, "
              "используется только fade по фиксированному времени. "
              "Чтобы включить: pip install silero-vad")
        _vad_model = None
    return _vad_model, _vad_get_speech_timestamps


def trim_silence(audio: "np.ndarray", sample_rate: int, vad_model, get_speech_timestamps,
                  margin_ms: float = 25.0):
    """Обрезает реальную тишину по краям audio, используя silero-vad —
    оставляет небольшой запас (margin_ms) с каждой стороны, чтобы не
    подрезать начало/конец самого звука (взрывные согласные и т.п.).

    Возвращает (обрезанный_audio, has_speech) — has_speech=False означает,
    что VAD вообще не нашёл речи в куске (похоже на "пустой"/неудавшийся
    синтез — вызывающий код может залогировать это как подозрение на
    пропуск, отдельно от обычных ошибок HTTP/исключений).

    Если VAD недоступен или в куске всего пара сэмплов — возвращает audio
    как есть с has_speech=True (не мешаем обычной работе)."""
    import numpy as np
    if vad_model is None or audio.size < 100:
        return audio, True
    try:
        import torch
        target_sr = 16000
        if sample_rate != target_sr:
            n_target = max(1, int(round(audio.shape[0] * target_sr / sample_rate)))
            old_idx = np.linspace(0, audio.shape[0] - 1, num=audio.shape[0])
            new_idx = np.linspace(0, audio.shape[0] - 1, num=n_target)
            audio_16k = np.interp(new_idx, old_idx, audio).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)

        timestamps = get_speech_timestamps(
            torch.from_numpy(audio_16k), vad_model,
            sampling_rate=target_sr, return_seconds=False,
        )
        if not timestamps:
            return audio, False

        ratio = sample_rate / target_sr
        margin = int(sample_rate * margin_ms / 1000)
        start = max(0, int(timestamps[0]["start"] * ratio) - margin)
        end = min(audio.shape[0], int(timestamps[-1]["end"] * ratio) + margin)
        if end <= start:
            return audio, True
        return audio[start:end], True
    except Exception:
        # VAD - необязательное улучшение, любая его ошибка не должна
        # ронять сам синтез
        return audio, True


def run_silero(chapters, outdir: Path, start: int, speaker: str, sample_rate: int, play: bool,
               model_id: str = DEFAULT_SILERO_MODEL, sentence_break_ms: int = 320,
               paragraph_break_ms: int = 550, comma_break_ms: int = 180,
               put_accent: bool = True, put_yo: bool = True, on_progress=None,
               chapter_indices=None, char_ranges=None, dialogue_speakers=None, attribution=None,
               should_stop=None, play_fn=None,
               stress_engine: str = "silero", stress_dict_path: "Path | str | None" = None,
               emphasize: bool = True, use_vad_trim: bool = True,
               use_stress_online: bool = False,
               book_stress_overrides_path: "Path | str | None" = None,
               strip_monosyllables: bool = True,
               stress_review: "bool | None" = None):
    """Озвучка через Silero TTS — нейросетевой русский голос, локально.
    По умолчанию v5_6_ru (последняя модель: эмфазы, восклицания, вопросы; прежняя — v5_5_ru).

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

    stress_engine — "silero" (по умолчанию): словарь -> silero-stress ->
    грамматический шлюз slovnet -> санитайзер; "none" — не расставлять ударения
    самим (остаётся только put_accent Silero). stress_dict_path — свой словарь
    "слово": "сл+ово" (JSON) для имён/терминов книги; он применяется раньше
    silero-stress и имеет над ним приоритет (см. load_stress_dictionary).

    emphasize — небольшое усиление интонации для кусков текста,
    заканчивающихся "?"/"!" (см. apply_emphasis) — приближение к тому,
    что для silero_rest даёт SSML <prosody>, но на уровне уже готового
    звука, так как локальный Silero не принимает SSML.
    """
    import numpy as np
    import wave

    stress_dict = load_stress_dictionary(stress_dict_path, book_stress_overrides_path)
    stress_engine = normalize_stress_engine(stress_engine)
    reset_stress_review(label=Path(outdir).name)
    set_strip_monosyllables(strip_monosyllables)
    if stress_review is not None:
        set_stress_review(stress_review)
    silero_stress_accentor = _load_silero_stress() if stress_engine_needs_silero(stress_engine) else None
    # The client chain owns the marks only when silero-stress really loaded; otherwise Silero's own
    # put_accent/put_yo stay available as the fallback.
    _chain_active = stress_layers_are_client_side(stress_engine, silero_stress_accentor)
    set_keep_hyphens(_chain_active)
    if stress_dict:
        print(f"Ручной словарь ударений: {len(stress_dict)} слов(а) "
              "(forced_stress_words.json + manual overrides) — для них нейросеть не вызывается.")
    vad_model, vad_get_timestamps = _load_silero_vad() if use_vad_trim else (None, None)

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
        # Порядок важен: сначала свой словарь (высший приоритет — конкретные
        # имена/термины книги), потом silero-stress (общая подстраховка для
        # всего остального). Слова из словаря на время silero-stress прячутся за
        # плейсхолдерами (apply_stress_dictionary_protected) и возвращаются
        # обратно уже после него (restore_stress_placeholders) — иначе на
        # составных словах через дефис ("во-первых", "по-русски" и т.п.)
        # акцентор может не узнать уже стоящее "+"-ударение и добавить своё
        # ещё раз, поверх; два ударения в одном слове звучат как заметное
        # "растягивание"/искажение при синтезе. put_accent=True ниже
        # остаётся включённым как ещё один, третий по счёту, уровень
        # подстраховки — он расставляет ударения только там, где их ещё не
        # оказалось (в защищённые плейсхолдерами слова он тоже не попадает).
        enrich_stress_dictionary_online(part_text, stress_dict, stress_dict_path,
                                         enabled=use_stress_online)
        # Слова вроде "звезды" (см. resolve_case_ambiguous_nouns) требуют
        # разного ударения в зависимости от падежа/числа — их нет в общем
        # stress_dict (исключены как омографы), поэтому решение по
        # конкретному фрагменту подмешиваем во ВРЕМЕННУЮ копию словаря,
        # ничего не сохраняя в JSON.
        part_text, _stress_placeholders = apply_stress_dictionary_protected(part_text, stress_dict)
        part_text = apply_stress_layers(part_text, stress_engine, silero_stress_accentor)
        part_text = restore_stress_placeholders(part_text, _stress_placeholders)
        # put_yo — тоже встроенный механизм САМОГО Silero (не silero-stress), и он
        # применяется к уже восстановленному тексту целиком, без разбора, что
        # именно пришло из словаря — в отличие от put_accent (который, судя
        # по всему, не трогает буквы там, где уже стоит "+"), put_yo может
        # заменить "е" на "ё" даже в словарном слове, где это не нужно
        # (например "самое" -> ошибочно "самоё"). Полагаться тут не на что —
        # поэтому для фрагментов, где сработал словарь, put_yo на этот
        # фрагмент просто отключается: словам ИЗ словаря "ё" при
        # необходимости нужно проставлять прямо в самом stress_dictionary.json
        # (там, где она действительно нужна), а не надеяться на угадывание.
        # На остальные (более частые) фрагменты без словарных слов это не
        # влияет — там put_yo работает как раньше.
        # Pass-through guarantee: when the client chain (dictionary -> silero-stress -> gate)
        # is active, Silero's own put_accent/put_yo are OFF so it can never alter a "+" word.
        effective_put_yo = put_yo and not _stress_placeholders and not _chain_active
        _eff_put_accent = put_accent and not _chain_active
        return model.apply_tts(
            text=part_text,
            speaker=part_speaker,
            sample_rate=sample_rate,
            put_accent=_eff_put_accent,
            put_yo=effective_put_yo,
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
        reset_stress_context()
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="silero", model=model_id, speaker=speaker,
            sample_rate=sample_rate, max_chars=max_chars,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms, put_accent=put_accent, put_yo=put_yo,
            dialogue_speakers=",".join(dialogue_speakers) if dialogue_speakers else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
            stress_engine=stress_engine, emphasize=emphasize,
            stress_dict_digest=stress_dict_digest(stress_dict, text), use_vad_trim=use_vad_trim,
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

        run_stress_review_for_chapter(text, stress_dict, title)

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

        prev_seg_speaker = None
        for seg_i, (part_text, pause_ms, seg_speaker) in enumerate(
            tqdm(segments, desc=f"Гл.{idx}", unit="фрагм."), 1
        ):
            if not part_text.strip():
                if on_progress:
                    on_progress(pos, total, seg_i, segs_total)
                continue
            # Небольшая дополнительная пауза перед репликой другого голоса
            # (смена seg_speaker внутри диалога) — короткая "пауза для
            # вдоха" перед сменой говорящего звучит естественнее, чем
            # реплики впритык друг к другу без какой-либо границы.
            if prev_seg_speaker is not None and seg_speaker != prev_seg_speaker and audio_parts:
                audio_parts.append(np.zeros(int(sample_rate * 0.09), dtype=np.float32))
            prev_seg_speaker = seg_speaker

            audio = synth_with_fallback(part_text, idx, title, seg_speaker)
            if vad_model is not None:
                audio, had_speech = trim_silence(audio, sample_rate, vad_model, vad_get_timestamps)
                if not had_speech and _text_has_speakable_content(part_text):
                    print(f"  [Гл.{idx} «{title}»] ПРЕДУПРЕЖДЕНИЕ: silero-vad не нашёл речи "
                          f"во фрагменте {part_text[:80]!r} — похоже на пропуск, проверьте "
                          f"результат на слух.")
            audio = apply_fade(audio, sample_rate)
            if emphasize:
                audio = apply_emphasis(audio, sample_rate, _emphasis_kind(part_text))
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
# «Да уж» без паузы между словами Silero сливает в «Длож»/«До уж» (замер ASR 16 окт 2026); 30 мс уже достаточно.
_DA_UZH_RE = re.compile(r"(?<![А-Яа-яЁё+])([Дд]\+?а)\s+(?=\+?(?:уж|и|еще|ещё|это|у|или|о)(?![А-Яа-яЁё]))")

# Буква-индекс модификации после дефиса («Шрайки-А», «Шрайки-Б»): читается названием буквы. Silero глотает «бэ» после слова
# (замер ASR 16 окт 2026), поэтому в SSML буква получает паузу 100 мс и rate 80%. Ориентир — дефис, а не регистр слова.
_LETTER_NAMES_RU = {"А": "а", "Б": "бэ", "В": "вэ", "Г": "гэ", "Д": "дэ", "Е": "е", "Ж": "жэ", "З": "зэ", "К": "ка",
                    "Л": "эль", "М": "эм", "Н": "эн", "П": "пэ", "Р": "эр", "С": "эс", "Т": "тэ", "Ф": "эф", "Х": "ха"}
_DESIGNATOR_LETTER_RE = re.compile(r"(?<=[а-яё])-([А-ЯЁ])(?![А-Яа-яЁё])")
_LETTER_MARK_RE = re.compile(r"\s*⟦([^⟧]+)⟧")


def protect_designator_letters(text: str, held: dict) -> str:
    """«Шрайки-Б» -> «Шрайки PRONSUBx»; в held буква лежит как маркер ⟦бэ⟧ (его разворачивает text_to_ssml; для обычного
    текста — strip_letter_marks). Имена плейсхолдеров продолжают нумерацию held от protect_pronunciation_substitutions."""
    def _sub(m):
        name_ru = _LETTER_NAMES_RU.get(m.group(1))
        if not name_ru:
            return m.group(0)
        ph = _pron_ph_name(len(held))
        held[ph] = "⟦" + name_ru + "⟧"
        return " " + ph
    return _DESIGNATOR_LETTER_RE.sub(_sub, text)


def strip_letter_marks(text: str) -> str:
    return _LETTER_MARK_RE.sub(lambda m: " " + m.group(1), text)


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


# Усиление вопросов/восклицаний. Раньше было pitch="high"/rate="fast" — это
# крупные пресеты Silero, из-за них интонация «скакала» на каждой реплике.
EMPHASIS_QUESTION_PITCH = "+15%"
EMPHASIS_EXCLAIM_PITCH = "+10%"
EMPHASIS_EXCLAIM_RATE = "107%"


def _prosody_wrap(escaped_sentence: str, original_sentence: str, emphasize: bool) -> str:
    """Слегка усиливает интонацию вопросительных/восклицательных предложений
    через <prosody pitch=.../rate=...> поверх того, что модель и так делает
    по самому знаку "?"/"!" — особенно заметно на репликах в кавычках и
    коротких восклицаниях."""
    if not emphasize:
        return f"<s>{escaped_sentence}</s>"
    stripped = original_sentence.rstrip()
    if stripped.endswith("?"):
        return f'<s><prosody pitch="{EMPHASIS_QUESTION_PITCH}">{escaped_sentence}</prosody></s>'
    if stripped.endswith("!"):
        return f'<s><prosody pitch="{EMPHASIS_EXCLAIM_PITCH}" rate="{EMPHASIS_EXCLAIM_RATE}">{escaped_sentence}</prosody></s>'
    return f"<s>{escaped_sentence}</s>"


def text_to_ssml(text: str, sentence_break_ms: int = 320,
                  paragraph_break_ms: int = 550, comma_break_ms: int = 180,
                  emphasize: bool = True, paragraph_prosody: "list | None" = None) -> str:
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

    paragraph_prosody — необязательный список [{"rate":..,"pitch":..} | None, ...]
    ДЛИНОЙ РОВНО КАК paragraphs (см. ниже) — по ПОЗИЦИИ абзаца, не по тексту
    (текст абзаца тут уже может отличаться от исходного из-за расставленных
    ударений "+", поэтому сопоставление по индексу, а не по строке) — см.
    SILERO_EMOTION_PROSODY, --smart-emotion режима silero_rest. Если для
    абзаца задана запись, весь <p>...</p> дополнительно оборачивается в
    <prosody rate=".." pitch="..">, поверх локальных <prosody> усиления
    вопросов/восклицаний (emphasize) — Silero поддерживает вложенные
    <prosody>, внутренний переопределяет только то, что задаёт сам.

    Ударения (silero-stress) в готовый SSML не добавляются здесь — это делает
    сервер (silero_rest_service.py) при получении запроса, в том числе для
    уже готового SSML, который присылает этот клиент.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()] or [text.strip()]

    p_chunks = []
    for para_idx, para in enumerate(paragraphs):
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        s_chunks = []
        for sent in sentences:
            escaped = _xml_escape(sent)
            escaped = _SOFT_PAUSE_RE.sub(
                lambda m: f'{_xml_escape(m.group(1))}<break time="{comma_break_ms}ms"/> ',
                escaped,
            )
            escaped = _DA_UZH_RE.sub(r'\1<break time="60ms"/> ', escaped)
            escaped = _LETTER_MARK_RE.sub(lambda m: f'<break time="100ms"/> <prosody rate="80%">{m.group(1)}</prosody>', escaped)
            s_chunks.append(_prosody_wrap(escaped, sent, emphasize))
        if s_chunks:
            inner = f'<break time="{sentence_break_ms}ms"/>'.join(s_chunks)
            para_prosody = paragraph_prosody[para_idx] if paragraph_prosody and para_idx < len(paragraph_prosody) else None
            if para_prosody:
                inner = (f'<prosody rate="{para_prosody["rate"]}" pitch="{para_prosody["pitch"]}">'
                         f'{inner}</prosody>')
            p_chunks.append("<p>" + inner + "</p>")

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
    # Every chunk is a separate synthesis call (a possible timbre jump). A tiny tail chunk is
    # appended to the previous one instead of being voiced on its own.
    if len(chunks) > 1 and len(chunks[-1]) < 150 and len(chunks[-2]) + len(chunks[-1]) + 1 <= max_len + 250:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()
    return chunks or [text]


def _prosody_for_line(line: str, prosody_by_para: dict):
    """Эмоция для строки куска. Абзац длиннее max_len режется по предложениям
    (_split_paragraphs_for_ssml), и строка куска — лишь часть абзаца, ключа
    prosody_by_para (полный текст абзаца) в таком виде нет: ищем абзац,
    содержащий строку (с точностью до пробелов)."""
    if not prosody_by_para:
        return None
    hit = prosody_by_para.get(line)
    if hit is not None or line in prosody_by_para:
        return hit
    norm = " ".join(line.split())
    if not norm:
        return None
    for para, pr in prosody_by_para.items():
        if len(para) > len(line) and norm in " ".join(para.split()):
            return pr
    return None


def run_silero_rest(chapters, outdir: Path, start: int, speaker: str, sample_rate: int,
                     play: bool, rest_url: str, sentence_break_ms: int, paragraph_break_ms: int,
                     comma_break_ms: int, emphasize: bool = True, max_len: int = 700,
                     model_id: str = DEFAULT_SILERO_MODEL, on_progress=None,
                     chapter_indices=None, char_ranges=None, dialogue_speakers=None, attribution=None,
                     should_stop=None, play_fn=None,
                     stress_dict_path: "Path | str | None" = None,
                     use_vad_trim: bool = True, use_stress_online: bool = False,
                     book_stress_overrides_path: "Path | str | None" = None,
                     stress_engine: str = "silero",
                     smart_emotion: bool = False, emotion_attribution=None,
                     strip_monosyllables: bool = True,
                     stress_review: "bool | None" = None,
                     asr_repair: bool = False):
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

    stress_dict_path — тот же словарь ударений "слово": "сл+ово" (JSON), что
    и у run_silero (см. load_stress_dictionary) — здесь применяется на
    стороне клиента, ДО построения SSML/отправки на сервер, поэтому у
    слов из словаря приоритет над silero-stress. Сервис (silero_rest_service.py)
    сам ударения не расставляет: при server_accent=false он озвучивает текст
    ровно с теми "+", что прислал клиент.

    smart_emotion — LLM (см. classify_paragraph_emotions_llm, тот же
    механизм, что и --yandex-smart-emotion) определяет эмоцию КАЖДОГО
    абзаца главы по смыслу текста, из набора SILERO_EMOTION_PROSODY.
    В отличие от Yandex (там отдельный параметр emotion запроса), у Silero
    нет такого API — вместо этого абзац оборачивается в SSML
    <prosody rate=".." pitch="..> (см. text_to_ssml, paragraph_prosody),
    поверх обычных пауз/усиления вопросов-восклицаний. Классификация
    делается ОДИН РАЗ НА ГЛАВУ (не на каждый max_len-фрагмент), чтобы LLM
    видела достаточно контекста и не тратился лишний запрос на каждый
    маленький кусок. При ошибке LLM — глава озвучивается как обычно, без
    эмоциональной окраски (не срывая синтез).
    emotion_attribution — {"api_key":...,"model":...,"provider":...,
    "folder_id":...} для этого LLM-вызова; по умолчанию (None) — тот же
    attribution, что и для атрибуции говорящих (см. run_yandex — тот же
    подход, разнесено намеренно, чтобы smart_emotion не требовал заодно
    включённой атрибуции диалогов).
    """
    import io
    import time
    import traceback as tb_module
    import requests
    import numpy as np
    import wave as wave_mod

    stress_dict = load_stress_dictionary(stress_dict_path, book_stress_overrides_path)
    pron_subs = load_pronunciation_substitutions()
    if stress_dict:
        print(f"Ручной словарь ударений: {len(stress_dict)} слов(а) "
              "(forced_stress_words.json + manual overrides) — для них нейросеть не вызывается.")
    vad_model, vad_get_timestamps = _load_silero_vad() if use_vad_trim else (None, None)

    # stress_engine применяется ЦЕЛИКОМ на клиенте, ДО отправки на сервис:
    # словарь -> silero-stress -> грамматический шлюз -> санитайзер. Сервис
    # (silero_rest_service.py) ударения не расставляет и сохраняет все "+".
    stress_engine = normalize_stress_engine(stress_engine)
    reset_stress_review(label=Path(outdir).name)
    set_strip_monosyllables(strip_monosyllables)
    if stress_review is not None:
        set_stress_review(stress_review)
    silero_stress_accentor = _load_silero_stress() if stress_engine_needs_silero(stress_engine) else None
    _client_layers = stress_layers_are_client_side(stress_engine, silero_stress_accentor)
    set_keep_hyphens(_client_layers)

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

    def _accented_text_for_rest(plain_text_chunk: str) -> str:
        """Prepares the text BEFORE it goes to silero_rest_service.py: the user's dictionary words,
        then silero-stress, the grammar gate and the sanitizer. The service keeps every "+"."""
        # Digits -> words BEFORE the stress layers, so "десять" gets its own stress mark (the
        # service would otherwise convert them after the layers, unstressed, or not at all).
        plain_text_chunk, _pron_held = protect_pronunciation_substitutions(plain_text_chunk, pron_subs)
        plain_text_chunk = protect_designator_letters(plain_text_chunk, _pron_held)
        plain_text_chunk = numbers_to_words_ru(plain_text_chunk)
        if _client_layers:
            protected, placeholders = apply_stress_dictionary_protected(plain_text_chunk, stress_dict)
            protected = apply_stress_layers(protected, stress_engine, silero_stress_accentor)
            return latin_to_cyrillic_ru(restore_pronunciation_substitutions(restore_stress_placeholders(protected, placeholders), _pron_held))
        return latin_to_cyrillic_ru(restore_pronunciation_substitutions(
            strip_monosyllable_stress(apply_stress_dictionary(plain_text_chunk, stress_dict)), _pron_held))

    # Заполняется заново для каждой главы (см. основной цикл ниже) —
    # {абзац_исходного_текста: {"rate":..,"pitch":..}} по классификации LLM.
    _chapter_para_prosody: dict = {}

    # ASR-автопочинка (asr_autorepair.py): гул/растяжка/проглоченный кусок -> пересинтез кусков
    # с другими паузами. Любой сбой ASR оставляет озвучку как есть.
    _asr_fixer = None
    if asr_repair:
        try:
            from asr_autorepair import AsrRepair
            _asr_fixer = AsrRepair(outdir, log=log)
            if not _asr_fixer.ok:
                _asr_fixer = None
        except Exception as e_asr:
            log(f"asr_repair недоступен ({e_asr}) — озвучка без автопочинки.")
            _asr_fixer = None

    def synth_via_ssml(plain_text_chunk: str, chunk_speaker: str, ssml_tweak=None) -> bytes:
        enrich_stress_dictionary_online(plain_text_chunk, stress_dict, stress_dict_path,
                                         enabled=use_stress_online)
        # paragraph_prosody передаётся ПО ПОЗИЦИИ абзаца (не по тексту) —
        # расстановка ударений ("+") меняет текст абзаца, поэтому сопоставлять
        # эмоцию нужно по номеру абзаца в чанке, а не по совпадению строк.
        # Абзацы plain_text_chunk и accented-текста (который дальше режется
        # на абзацы внутри text_to_ssml) идут в том же порядке 1:1 — сама
        # расстановка ударений абзацы не добавляет и не убирает.
        para_prosody_list = None
        if _chapter_para_prosody:
            plain_paras = [p for p in plain_text_chunk.split("\n") if p.strip()]
            para_prosody_list = [_prosody_for_line(p, _chapter_para_prosody) for p in plain_paras]
        ssml = text_to_ssml(
            _accented_text_for_rest(plain_text_chunk),
            sentence_break_ms=sentence_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
            emphasize=emphasize,
            paragraph_prosody=para_prosody_list,
        )
        if ssml_tweak is not None:
            ssml = ssml_tweak(ssml)

        def request_ssml(one_ssml: str) -> bytes:
            resp = _get_with_retry(ssml_endpoint, {
                "text_to_speech": one_ssml,
                "speaker": chunk_speaker,
                "sample_rate": sample_rate,
                "raw_ssml": "true",
                "strip_mono": "true" if strip_monosyllables else "false",
                "server_accent": "false" if _client_layers else "true",
            })
            level = resp.headers.get("X-Synthesis-Level", "as-is")
            if level != "as-is":
                log(f"сервис использовал упрощённый вариант синтеза: {level}")
            return resp.content

        if isinstance(ssml, list):
            # автопочинка (asr_autorepair): кусок разрезан на независимые запросы, звук склеивается
            from asr_autorepair import join_wavs
            return join_wavs([request_ssml(part) for part in ssml], getattr(ssml, "gap_ms", 270))
        return request_ssml(ssml)

    def synth_via_plain_text(plain_text_chunk: str, chunk_speaker: str) -> bytes:
        enrich_stress_dictionary_online(plain_text_chunk, stress_dict, stress_dict_path,
                                         enabled=use_stress_online)
        resp = _get_with_retry(plain_endpoint, {
            "text_to_speech": strip_letter_marks(_accented_text_for_rest(plain_text_chunk)),
            "speaker": chunk_speaker,
            "sample_rate": sample_rate,
            "strip_mono": "true" if strip_monosyllables else "false",
            "server_accent": "false" if _client_layers else "true",
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
            if _asr_fixer is not None:
                wav_bytes = _asr_fixer.check_and_fix(
                    plain_text_chunk, wav_bytes,
                    lambda tweak: synth_via_ssml(plain_text_chunk, chunk_speaker, tweak),
                    tag=f"гл.{idx} фрагм.{chunk_no}", first_chunk=(chunk_no == 1))
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
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        if vad_model is not None:
            audio, had_speech = trim_silence(audio, sample_rate, vad_model, vad_get_timestamps)
            if not had_speech:
                log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] ПРЕДУПРЕЖДЕНИЕ: silero-vad не "
                    f"нашёл речи во фрагменте {plain_text_chunk[:80]!r} — похоже на пропуск, "
                    f"проверьте результат на слух.")
        return apply_fade(audio, sample_rate)

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        reset_stress_context()
        out_path = outdir / fname

        fingerprint = _params_fingerprint(
            text, mode="silero_rest", model=model_id, speaker=speaker, sample_rate=sample_rate,
            sentence_break_ms=sentence_break_ms, paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms, emphasize=emphasize, max_len=max_len,
            dialogue_speakers=",".join(dialogue_speakers) if dialogue_speakers else "",
            attribution=(attribution.get("provider", ""), attribution.get("model", "")) if attribution else "",
            stress_dict_digest=stress_dict_digest(stress_dict, text), use_vad_trim=use_vad_trim,
            stress_engine=stress_engine, smart_emotion=smart_emotion,
            **({"pron_digest": _pd} if (_pd := pronunciation_digest(pron_subs, text)) else {}),
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

        _sg = _stall_guard_start(f"chapter {idx}: preparation + first fragment")
        run_stress_review_for_chapter(text, stress_dict, title)

        _chapter_para_prosody = {}
        if smart_emotion and _llm_provider_allowed(((emotion_attribution if emotion_attribution is not None else attribution) or {}).get("provider")):
            chapter_paras = [p for p in text.split("\n") if p.strip()]
            _emo_creds = emotion_attribution if emotion_attribution is not None else attribution
            try:
                emo_labels = classify_paragraph_emotions_llm(
                    chapter_paras, allowed_emotions=list(SILERO_EMOTION_PROSODY.keys()),
                    api_key=(_emo_creds or {}).get("api_key", ""),
                    model=(_emo_creds or {}).get("model", DEFAULT_ATTRIBUTION_MODEL),
                    provider=(_emo_creds or {}).get("provider", DEFAULT_ATTRIBUTION_PROVIDER),
                    folder_id=(_emo_creds or {}).get("folder_id", ""),
                    ollama_host=(_emo_creds or {}).get("ollama_host", ""),
                    log_fn=log,
                )
                _chapter_para_prosody = {
                    p: pr for p, pr in zip(chapter_paras, smooth_emotion_prosody(emo_labels))
                }
                log(f"smart_emotion: определены эмоции для {len(chapter_paras)} абзац(ев) главы.")
            except Exception as e:
                log(f"smart_emotion не удался ({e}) — глава озвучивается без эмоциональной окраски.")
                _chapter_para_prosody = {}

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
            _stall_guard_stop(_sg)
            audio_parts.append(audio)
            audio_parts.append(pause)
            if on_progress:
                on_progress(pos, total, chunk_no, chunks_total)

        _stall_guard_stop(_sg)
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
        if _asr_fixer is not None:
            _asr_fixer.chapter_done(idx)

        if play:
            print("  Проигрывание...")
            (play_fn or play_file)(out_path)

    if _asr_fixer is not None:
        _asr_fixer.close()
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
    # Раньше в конце всегда печаталось "Готово", даже если половина
    # фрагментов не синтезировалась и была заменена тишиной - по этому
    # сообщению было невозможно понять, что что-то пошло не так, не читая
    # весь лог целиком. Считаем неудачи по ходу дела и меняем финальное
    # сообщение, если они были.
    failure_stats = {"failed_fragments": 0, "silent_fragments": 0, "chapters_with_failures": set()}

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
            failure_stats["failed_fragments"] += 1
            failure_stats["chapters_with_failures"].add(idx)
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
            failure_stats["silent_fragments"] += 1
            failure_stats["chapters_with_failures"].add(idx)
            log(f"[Гл.{idx} '{title}', фрагмент {chunk_no}] ПРЕДУПРЕЖДЕНИЕ: сервис вернул "
                f"пустой или практически беззвучный WAV (без ошибки HTTP) — если это "
                f"повторяется на каждом фрагменте, скорее всего профиль голоса "
                f"{chunk_voice!r} не загрузился на сервисе (см. cosyvoice_rest_service.log).")
            return audio
        # Fade-in/out на границах — убирает щелчки/потрескивание на стыках
        # кусков при склейке (см. apply_fade); тишину (ветка выше) фейдить
        # не нужно.
        return apply_fade(audio, sample_rate)

    selection = _select_chapters(chapters, start=start, chapter_indices=chapter_indices,
                                  char_ranges=char_ranges)
    total = len(selection) or 1

    for pos, (idx, title, text) in enumerate(selection, 1):
        if should_stop and should_stop():
            print("Остановлено пользователем (после предыдущей главы).")
            break
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        reset_stress_context()
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

    total_bad = failure_stats["failed_fragments"] + failure_stats["silent_fragments"]
    if total_bad > 0:
        n_chapters = len(failure_stats["chapters_with_failures"])
        print(
            f"\nЗАВЕРШЕНО С ОШИБКАМИ: не удалось озвучить {total_bad} фрагмент(ов) "
            f"({failure_stats['failed_fragments']} — ошибка синтеза, "
            f"{failure_stats['silent_fragments']} — пустой/беззвучный ответ) "
            f"в {n_chapters} глав(ах). На месте этих фрагментов в готовых .wav — тишина. "
            f"Файлы сохранены в: {outdir.resolve()}"
        )
        print(f"Подробности по каждому фрагменту — в логе: {log_path.resolve()}")
    else:
        print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")
        print(f"Подробный лог ошибок (если были): {log_path.resolve()}")

    # Возвращаем статистику ошибок, чтобы вызывающий код (GUI) не писал
    # "Озвучка завершена" бодрым тоном, если на самом деле часть фрагментов
    # заменена тишиной - раньше run_cosyvoice ничего не возвращал, и GUI
    # не могла отличить полный успех от "завершилось, но с дырами".
    return {
        "failed_fragments": failure_stats["failed_fragments"],
        "silent_fragments": failure_stats["silent_fragments"],
        "chapters_with_failures": len(failure_stats["chapters_with_failures"]),
    }


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
    d = _app_paths.user_file("piper_voices")
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
        reset_stress_context()
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

def _stress_selftest() -> None:
    """--stress-selftest: the real chain (user dictionary -> silero-stress -> grammar gate ->
    sanitizer) on a few hard sentences, printed so the result can be checked by eye."""
    silero = _load_silero_stress()
    stress_dict = load_stress_dictionary()
    print(f"manual dictionary: {len(stress_dict)} word(s); silero-stress: {'yes' if silero else 'no'}")
    for sentence in ("Он сказал себе, что всё хорошо, и подошел к двери.",
                     "Хотел я зайти в замок, но ключ в замок не подошел.",
                     "Все пришли, но это все, что он знал.",
                     "Когда-нибудь кто-то скажет: дай-ка мне сети и земли."):
        prot, ph = apply_stress_dictionary_protected(sentence, stress_dict)
        out = apply_stress_layers(prot, "silero", silero)
        print(restore_stress_placeholders(out, ph))


# --- «Анализ омографов книги» (Stress Dictionary tab, gui_stress_editor.py; no console command) ---------
# Scan = the SAME chain the audio gets (digits -> words, ё marks, silero-stress primed with the previous
# paragraph, grammar gate, sanitizer), chapter by chapter, WITHOUT the user's dictionaries: what the models
# decide for each occurrence of a homograph is recorded as a PHRASE key (the word plus 1-2 neighbours).
# Never a flat single-word entry - that would flatten the other readings. forced_stress_words.json still
# wins at synthesis time.
_BBD_VOWEL_RE = re.compile(r"[аеёиоуыэюя]", re.I)
# extra homographs that are not in the ambiguous list (one-vowel words are filtered out of it)
_EXTRA_HOMOGRAPHS = ("все",)
# context windows (offsets of the first/last word relative to the target), tried in this order:
# "мы все побежали", "в этой земли", "земли лены харрингтон", then wider ones
_BBD_WINDOWS = ((-1, 1), (-2, 0), (0, 2), (-2, 1), (-1, 2), (-2, 2), (-1, 0), (0, 1))


def _bbd_norm(w: str) -> str:
    return w.lower().replace("ё", "е")


def homograph_targets() -> set:
    """Plain lower-case spellings (е for ё) of the doubtful words: the grammar gate's paradigm table
    (земли, сети, трети, голоса, корпуса...), the ambiguous-stress list and _EXTRA_HOMOGRAPHS (все)."""
    t = set(_EXTRA_HOMOGRAPHS)
    try:
        import grammar_gate
        t.update(_bbd_norm(k) for k in grammar_gate.PARADIGM_TABLE)
    except Exception:
        pass
    try:
        t.update(_bbd_norm(w) for w in _load_ambiguous_stress_words())
    except Exception:
        pass
    return t


def _bbd_value(out_tok: str, target: bool = False) -> "tuple[str, bool]":
    """(lower-case form, reading known?) for one output token. One-vowel words lose their mark; a
    multi-syllable word counts as known when it carries a '+' or ё. A TARGET keeps/gets its mark even
    with one vowel ("все" -> "вс+е", "вс+ё"), because that mark is exactly the reading we record."""
    t = out_tok.lower()
    plain = t.replace("+", "")
    vowels = list(_BBD_VOWEL_RE.finditer(plain))
    if target and "+" not in t and "ё" not in plain and len(vowels) == 1:
        i = vowels[0].start()
        return plain[:i] + "+" + plain[i:], True
    if len(vowels) <= 1 and "ё" not in plain:
        return plain, False
    return t, ("+" in t or "ё" in plain)


def scan_book_stress(chapters, silero_accentor, targets=None, log_fn=print, cancel=None) -> "list[list[tuple]]":
    """Runs every paragraph of every chapter through the default pipeline (so the context priming works
    exactly as in synthesis) and returns, per paragraph, word tuples
    (key, value, is_target_reading, joined_to_next, chapter_no): key = source spelling lower-cased,
    value = pipeline output lower-cased (with '+'), joined_to_next = only whitespace separates it from
    the next word (a phrase key may continue). `cancel` (anything with is_set()) aborts with InterruptedError."""
    import time as _time
    t0 = _time.time()
    targets = targets or set()
    set_keep_hyphens(True)
    paras: list = []
    total = len(chapters)
    for ci, (title, text) in enumerate(chapters, 1):
        reset_stress_context()                      # first paragraph of a chapter has no left context
        first, n_words, n_hits = len(paras), 0, 0
        for para in numbers_to_words_ru(text or "").split("\n"):
            if not para.strip():
                continue
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            protected, ph = apply_stress_dictionary_protected(para, {})     # only the ё marks
            marked = restore_stress_placeholders(apply_stress_layers(protected, "silero", silero_accentor), ph)
            src = [(m.group(0), m.start(), m.end()) for m in _STRESS_WORD_RE.finditer(para)]
            out_tokens = _OUT_WORD_RE.findall(marked)
            mapping = _align_plain([_bbd_norm(w) for w, _s, _e in src], out_tokens)
            row = []
            for i, (w, s, e) in enumerate(src):
                j = mapping.get(i)
                is_t = j is not None and "ё" not in w.lower() and _bbd_norm(w) in targets
                value, known = _bbd_value(out_tokens[j], is_t) if j is not None else (w.lower(), False)
                hit = is_t and known
                n_hits += hit
                nxt = src[i + 1][1] if i + 1 < len(src) else None
                joined = nxt is not None and para[e:nxt] != "" and para[e:nxt].strip() == ""
                row.append((w.lower(), value, hit, joined, ci))
            n_words += len(row)
            if row:
                paras.append(row)
        log_fn(f"  [{ci}/{total}] {title[:50]!r}: {len(paras) - first} paragraphs, {n_words} words, "
               f"{n_hits} homograph occurrence(s) ({_time.time() - t0:.0f} s)")
    return paras


def derive_homograph_phrases(paras, max_phrases_per_word: int = 60,
                             only_multi: bool = True) -> "tuple[dict, dict]":
    """(entries, stats). entries = {"мы все побежали": "мы вс+е побежали"}: phrase keys only. With
    only_multi (default) a word is kept only when the book reads it in at least two ways. For each
    occurrence the first window (1-2 words around it) whose words occur with ONE reading of the target in
    the whole book is used, so the keys never contradict each other."""
    readings: dict = {}                    # key -> {reading: [count, {chapters}]}
    occ: dict = {}                         # key -> [(para index, word index)]
    for pi, row in enumerate(paras):
        for wi, (key, value, hit, _j, ch) in enumerate(row):
            if hit:
                rec = readings.setdefault(key, {}).setdefault(value, [0, set()])
                rec[0] += 1
                rec[1].add(ch)
                occ.setdefault(key, []).append((pi, wi))
    chosen_words = [k for k, v in readings.items() if len(v) > 1 or not only_multi]

    def window(pi, wi, lo, hi):
        row = paras[pi]
        a, b = wi + lo, wi + hi
        if a < 0 or b >= len(row) or any(not row[k][3] for k in range(a, b)):
            return None
        toks = row[a:b + 1]
        return (" ".join(_bbd_norm(t[0]) for t in toks), " ".join(t[0] for t in toks),
                " ".join(t[1] for t in toks))

    phrases: dict = {}                     # normalised key -> [display key, value, count, reading]
    conflicted: set = set()
    unresolved: dict = {}
    for key in chosen_words:
        places = occ[key]
        owners_by_window = []
        for lo, hi in _BBD_WINDOWS:
            owners: dict = {}
            for pi, wi in places:
                w = window(pi, wi, lo, hi)
                if w:
                    owners.setdefault(w[0], set()).add(paras[pi][wi][1])
            owners_by_window.append(owners)
        picked: dict = {}
        for pi, wi in places:
            reading = paras[pi][wi][1]
            for (lo, hi), owners in zip(_BBD_WINDOWS, owners_by_window):
                w = window(pi, wi, lo, hi)
                if w and len(owners[w[0]]) == 1:
                    rec = picked.setdefault(w[0], [w[1], w[2], 0, reading, paras[pi][wi][4]])
                    rec[2] += 1
                    break
            else:
                unresolved[key] = unresolved.get(key, 0) + 1
        by_reading: dict = {}                       # cap per word, fair between readings
        for nkey, rec in picked.items():
            by_reading.setdefault(rec[3], []).append((nkey, rec))
        queues = [sorted(v, key=lambda x: -x[1][2]) for v in by_reading.values()]
        kept = []
        while queues and len(kept) < max_phrases_per_word:
            for q in list(queues):
                if q and len(kept) < max_phrases_per_word:
                    kept.append(q.pop(0))
                if not q:
                    queues.remove(q)
        for nkey, rec in kept:
            old = phrases.get(nkey)
            if nkey in conflicted:
                continue
            if old and old[1] != rec[1]:           # two homographs side by side disagree on the neighbour
                conflicted.add(nkey)
                phrases.pop(nkey, None)
            elif not old:
                phrases[nkey] = rec

    entries = {phrases[n][0]: phrases[n][1] for n in sorted(phrases)}
    items = [{"key": phrases[n][0], "value": phrases[n][1], "count": phrases[n][2], "chapter": phrases[n][4]}
             for n in sorted(phrases)]
    stats = {
        "items": items,
        "homograph_occurrences": sum(len(v) for v in occ.values()),
        "homograph_words_found": len(readings),
        "homograph_words_multi": sum(1 for v in readings.values() if len(v) > 1),
        "phrase_keys": len(entries),
        "unresolved_occurrences": sum(unresolved.values()),
        "unresolved_words": sorted(unresolved, key=lambda k: -unresolved[k])[:10],
        "words_report": {k: {r: v[0] for r, v in sorted(readings[k].items(), key=lambda x: -x[1][0])}
                         for k in sorted(chosen_words, key=lambda k: -sum(v[0] for v in readings[k].values()))[:15]},
    }
    return entries, stats


def _grammar_gate_available() -> bool:
    try:
        import grammar_gate
        return bool(grammar_gate._load())
    except Exception:
        return False


def analyze_chapters_homographs(chapters, max_phrases_per_word: int = 60, only_multi: bool = True,
                                progress_cb=None, cancel=None) -> "tuple[dict, dict]":
    """Engine of the «Анализ омографов книги» button (gui_stress_editor.py; there is no console command):
    scans the already loaded chapters [(title, text)] with the default stress pipeline and returns
    (entries, stats) - entries = {"мы все побежали": "мы вс+е побежали"}, stats["items"] = per phrase
    {key, value, count, chapter}. Runs in a GUI worker thread; `progress_cb(str)` gets a line per chapter,
    `cancel.is_set()` aborts (InterruptedError). Raises RuntimeError when silero-stress is missing."""
    silero = _load_silero_stress()
    if silero is None:
        raise RuntimeError("silero-stress не установлен - нечем разрешать ударения")
    targets = homograph_targets()
    log = progress_cb or (lambda m: None)
    log(f"Глав: {len(chapters)}; слов-омографов в списке: {len(targets)}; "
        f"грамматический шлюз: {'вкл' if _grammar_gate_available() else 'ВЫКЛ (падежные омографы не разбираются)'}")
    paras = scan_book_stress(chapters, silero, targets, log, cancel)
    entries, stats = derive_homograph_phrases(paras, max_phrases_per_word, only_multi)
    stats["chapters"] = len(chapters)
    stats["words_total"] = sum(len(r) for r in paras)
    return entries, stats


SUBCOMMANDS = ("fetch-models", "selftest", "serve-silero", "gui")


def run_subcommand(argv: "list[str]") -> int:
    """Internal orchestrator: every non-synthesis phase is a subcommand of this one program
    (`fb2_reader[.exe] <subcommand> ...`), so a frozen build needs no side scripts or python.exe.
    Callable from the GUI as well (it spawns `[sys.executable, "serve-silero"]` when frozen)."""
    import app_paths
    ap = argparse.ArgumentParser(prog="fb2_reader", description="fb2_reader orchestrator")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-models", help="download grammar-gate models (and optionally the stress corpus)")
    f.add_argument("--corpus", action="store_true", help="also fetch stress_corpus_ru.tsv.gz")
    sub.add_parser("selftest", help="check that silero-stress, the grammar gate and the asset paths resolve")
    sv = sub.add_parser("serve-silero", help="run the Silero REST service in this process")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=5010)
    sub.add_parser("gui", help="graphical interface")
    ns = ap.parse_args(argv)

    if app_paths.is_frozen():
        os.chdir(app_paths.app_dir())          # relative paths (logs, outputs) land next to the exe
        app_paths.seed_defaults()
    if ns.command == "fetch-models":
        import grammar_gate
        print("grammar models ->", grammar_gate.fetch_models())
        if ns.corpus:
            print("corpus: run the corpus download from the GUI/README (word-form only; not needed for training)")
        return 0
    if ns.command == "selftest":
        import grammar_gate
        print(app_paths.describe())
        ok = True
        ss = _load_silero_stress()
        print("silero-stress:", "ok" if ss is not None else "MISSING (pip install silero-stress)")
        ok = ss is not None
        print("grammar gate:", "ON" if grammar_gate._load() else "OFF (packages/models missing)")
        if grammar_gate._load():
            print(grammar_gate.grammar_gate("Он сидел у воды."))
        return 0 if ok else 1
    if ns.command == "serve-silero":
        import uvicorn
        import silero_rest_service
        uvicorn.run(silero_rest_service.app, host=ns.host, port=ns.port, log_level="info")
        return 0
    if ns.command == "gui":
        from fb2_reader_gui import run_gui
        run_gui()
        return 0
    return 2


def main():
    import multiprocessing
    multiprocessing.freeze_support()          # frozen exe: worker processes must not re-run main()
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        sys.exit(run_subcommand(sys.argv[1:]))
    try:
        import app_paths as _ap
        if _ap.is_frozen():
            os.chdir(_ap.app_dir())
            _ap.seed_defaults()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Озвучивание FB2-книг на русском языке")
    ap.add_argument("book", type=Path, nargs="?", help="путь к .fb2 или .fb2.zip файлу")
    ap.add_argument("--gui", action="store_true", help="запустить графический интерфейс")
    ap.add_argument("--mode", choices=["online", "offline", "silero", "silero_rest", "cosyvoice", "piper",
                                        "yandex", "qwen_tts", "qwen_tts_local"],
                     default="silero",
                     help="silero = нейросетевой голос локально через torch.hub; "
                          "silero_rest = синтез через Silero-REST-Service с интонационными паузами "
                          "(SSML: паузы на запятых/тире, границах предложений и абзацев, интонация "
                          "вопросов/восклицаний); cosyvoice = локальный сервис CosyVoice (GPU, "
                          "клонирование голоса по образцу — см. install_cosyvoice.bat); "
                          "piper = Piper TTS (локально, CPU, быстро, без клонирования — "
                          "см. --piper-voice); "
                          "yandex = облачный Yandex SpeechKit (платно, нужны "
                          "--yandex-api-key и --yandex-folder-id); "
                          "qwen_tts = облачный Qwen3-TTS через DashScope (платно, нужен "
                          "--qwen-api-key; не проверено на живом аккаунте — см. комментарий у "
                          "QWEN_TTS_VOICES в коде); online = gTTS (интернет); "
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
                          "предложений (через <prosody> в silero_rest-режиме, через лёгкую "
                          "обработку звука в silero-режиме) — по умолчанию усиление включено")
    ap.add_argument("--no-accent", action="store_true",
                     help="не расставлять ударения автоматически в silero-режиме "
                          "(по умолчанию расставляются)")
    ap.add_argument("--no-yo", action="store_true",
                     help="не заменять 'е' на 'ё' там, где нужно, в silero-режиме "
                          "(по умолчанию заменяется)")
    ap.add_argument("--stress-selftest", action="store_true",
                     help="accentuate a few test sentences through the full pipeline "
                          "(dictionary -> silero-stress -> grammar gate) and exit")
    ap.add_argument("--keep-monosyllable-stress", action="store_true",
                     help="do not strip the '+' mark from one-vowel words")
    ap.add_argument("--asr-repair", action="store_true",
                    help="silero_rest: проверять каждый кусок распознаванием (whisper в .venv_cosyvoice3) и пересинтезировать "
                         "куски с гулом/растяжкой/проглоченным текстом с другими паузами (см. asr_autorepair.py)")
    ap.add_argument("--no-stress-review", action="store_true",
                     help="do not write the per-chapter review list "
                          "review_list.json (same as AUDIOBOOK_STRESS_REVIEW=0)")
    ap.add_argument("--stress-engine", type=str, default="silero",
                     choices=["silero", "none", "ruaccent", "silero_stress", "hybrid", "ollama", "lmstudio"],
                     help="'silero' (default) - dictionary -> silero-stress -> grammar gate -> "
                          "sanitizer; 'none' - switch automatic stress off. Legacy values "
                          "(ruaccent/silero_stress/hybrid/ollama/lmstudio) are accepted and mean 'silero'.")
    ap.add_argument("--stress-ollama-model", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--ollama-host", type=str, default="",
                     help="адрес локального LLM-сервера только для --attribution-provider "
                          "ollama/lmstudio (в расстановке ударений LLM не участвует)")
    ap.add_argument("--stress-dict", type=str, default="",
                     help="свой словарь ударений (JSON: {\"слово\": \"сл+ово\"}) для "
                          "silero-режима — приоритетнее silero-stress, для имён/терминов книги. "
                          "По умолчанию: stress_dictionary.json рядом со скриптом, если есть")
    ap.add_argument("--no-vad-trim", action="store_true",
                     help="не обрезать тишину по краям фрагментов через silero-vad в "
                          "silero/silero_rest-режимах (по умолчанию обрезается, если пакет "
                          "silero-vad установлен) — используется только фиксированный fade")
    ap.add_argument("--stress-online", action="store_true",
                     help="искать ударения слов, которых нет в --stress-dict, в "
                          "офлайн-корпусе (stress_corpus_ru.tsv.gz) и в интернете "
                          "(Викисловарь) в silero/silero_rest-режимах, и дописывать найденное "
                          "в словарь. ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО: и офлайн-корпус, и Викисловарь "
                          "на практике не различают грамматические формы одного слова с разным "
                          "ударением (например 'катера' — 'ка́тера' в родительном падеже "
                          "единственного числа, но 'катера́' во множественном — а в словарь "
                          "может попасть не тот вариант) и содержат отдельные готовые ошибки "
                          "даже на простых словах ('пока', 'именно') — включённое по "
                          "умолчанию, это раньше портило произношение обычных слов, которые "
                          "silero-stress и без словаря обычно угадывает по контексту правильно. "
                          "Включайте только если специально хотите пополнить словарь и готовы "
                          "потом вручную проверять/чистить stress_dictionary.json.")
    ap.add_argument("--no-stress-online", action="store_true",
                     help="(оставлено для обратной совместимости, ничего не делает — "
                          "автопоиск в интернете/офлайн-корпусе теперь и так выключен по "
                          "умолчанию; используйте просто без --stress-online)")
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
                          "good, neutral, evil — единая на всю озвучку. Игнорируется, если "
                          "задан --yandex-smart-emotion")
    ap.add_argument("--yandex-smart-emotion", action="store_true",
                     help="вместо единой --yandex-emotion — LLM (тот же ключ/провайдер, что и "
                          "--attribution-provider) сама определяет эмоцию КАЖДОГО абзаца по "
                          "смыслу текста (режим yandex, только для голосов ermil/jane; для "
                          "остальных голосов Yandex эмоции не поддерживает вообще, флаг "
                          "тихо игнорируется)")
    ap.add_argument("--smart-emotion", action="store_true",
                     help="режим silero_rest: LLM (тот же ключ/провайдер, что и "
                          "--attribution-provider) определяет эмоцию КАЖДОГО абзаца по смыслу "
                          "текста и озвучивает его через SSML <prosody rate/pitch> (см. "
                          "SILERO_EMOTION_PROSODY) — тоньше, чем фиксированные паузы/усиление "
                          "вопросов-восклицаний. При ошибке LLM глава озвучивается как обычно.")
    ap.add_argument("--yandex-speed", type=float, default=1.0,
                     help="скорость речи Yandex SpeechKit, от 0.1 до 3.0 (по умолчанию 1.0)")
    ap.add_argument("--yandex-dialogue-voices", type=str, default="",
                     help="список голосов через запятую (например: jane,filipp,zahar) — если "
                          "задан, реплики диалогов (абзацы, начинающиеся с тире) будут по "
                          "очереди озвучены этими голосами вместо голоса --yandex-voice, "
                          "чтобы диалоги не звучали одним монотонным голосом; по умолчанию "
                          "не задано — вся книга одним голосом, как раньше")
    ap.add_argument("--qwen-api-key", type=str, default="",
                     help="API-ключ DashScope (режим qwen_tts); можно также задать переменной "
                          "окружения DASHSCOPE_API_KEY")
    ap.add_argument("--qwen-voice", type=str, default=QWEN_TTS_DEFAULT_VOICE,
                     choices=list(QWEN_TTS_VOICES.keys()),
                     help="голос Qwen3-TTS (режим qwen_tts)")
    ap.add_argument("--qwen-dialogue-voices", type=str, default="",
                     help="список голосов Qwen3-TTS через запятую для чередования на репликах "
                          "диалогов — аналог --yandex-dialogue-voices")
    ap.add_argument("--qwen-local-url", type=str, default=QWEN_TTS_LOCAL_DEFAULT_URL,
                     help=f"адрес локального Gradio-сервера Qwen3-TTS (режим qwen_tts_local), "
                          f"по умолчанию {QWEN_TTS_LOCAL_DEFAULT_URL}")
    ap.add_argument("--qwen-local-voice", type=str, default=QWEN_TTS_LOCAL_DEFAULT_VOICE,
                     choices=list(QWEN_TTS_LOCAL_VOICES.keys()),
                     help="голос для локального Qwen3-TTS (режим qwen_tts_local) — список "
                          "голосов у локального сервера свой, отличается от облачного "
                          "(--qwen-voice); см. QWEN_TTS_LOCAL_VOICES в коде")
    ap.add_argument("--check-model-updates", action="store_true",
                     help="проверить на GitHub, не появилась ли более новая модель "
                          "Silero для русского языка, и выйти")
    ap.add_argument("--attribution-provider", type=str, default=DEFAULT_ATTRIBUTION_PROVIDER,
                     choices=list(ATTRIBUTION_PROVIDERS.keys()),
                     help="сервис для определения, какой персонаж говорит каждую реплику "
                          "диалога: yandexgpt (работает из РФ без VPN, есть бесплатный лимит), "
                          "gemini (Google, бесплатно, но часто недоступен из РФ), anthropic "
                          "(Claude, платно) или ollama (локально на вашей машине, бесплатно, "
                          "без ключа, нужен запущенный Ollama — см. --ollama-host и "
                          f"--attribution-model). По умолчанию {DEFAULT_ATTRIBUTION_PROVIDER}.")
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
    if args.no_stress_review:
        globals()["STRESS_REVIEW_ENABLED"] = False

    if args.stress_selftest:
        _stress_selftest()
        return

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
    attribution_folder_id = args.attribution_folder_id or args.yandex_folder_id \
        or os.environ.get("YANDEX_FOLDER_ID", "")
    attribution = {
        "api_key": attribution_api_key, "model": attribution_model,
        "provider": attribution_provider, "folder_id": attribution_folder_id,
        "ollama_host": args.ollama_host,
    } if (attribution_api_key or attribution_provider in ("ollama", "lmstudio", "builtin")) else None

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
                   attribution=attribution,
                   stress_engine=args.stress_engine,
                   emphasize=not args.no_emphasis,
                   stress_dict_path=args.stress_dict or None,
                   use_vad_trim=not args.no_vad_trim,
                   use_stress_online=args.stress_online,
                   book_stress_overrides_path=book_manual_stress_overrides_path(args.book),
                   strip_monosyllables=not args.keep_monosyllable_stress)
    elif args.mode == "silero_rest":
        run_silero_rest(chapters, args.outdir, args.start, args.speaker, args.sample_rate, args.play,
                         args.rest_url, args.sentence_break_ms, args.paragraph_break_ms,
                         args.comma_break_ms, emphasize=not args.no_emphasis, model_id=args.model,
                         chapter_indices=chapter_indices, dialogue_speakers=dialogue_speakers,
                         attribution=attribution,
                         stress_dict_path=args.stress_dict or None,
                         use_vad_trim=not args.no_vad_trim,
                         use_stress_online=args.stress_online,
                         book_stress_overrides_path=book_manual_stress_overrides_path(args.book),
                         stress_engine=args.stress_engine,
                         smart_emotion=args.smart_emotion,
                         strip_monosyllables=not args.keep_monosyllable_stress,
                         asr_repair=args.asr_repair)
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
                   attribution=attribution, smart_emotion=args.yandex_smart_emotion)
    elif args.mode == "qwen_tts":
        api_key = args.qwen_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        dialogue_voices = [v.strip() for v in args.qwen_dialogue_voices.split(",") if v.strip()] or None
        run_qwen_tts(chapters, args.outdir, args.start, args.play, api_key,
                     voice=args.qwen_voice,
                     chapter_indices=chapter_indices, dialogue_voices=dialogue_voices,
                     attribution=attribution)
    elif args.mode == "qwen_tts_local":
        dialogue_voices = [v.strip() for v in args.qwen_dialogue_voices.split(",") if v.strip()] or None
        run_qwen_tts_local(chapters, args.outdir, args.start, args.play, args.qwen_local_url,
                            voice=args.qwen_local_voice,
                            chapter_indices=chapter_indices, dialogue_voices=dialogue_voices,
                            attribution=attribution)
    else:
        run_offline(chapters, args.start, args.rate, args.voice, voice_id=args.voice,
                    chapter_indices=chapter_indices, dialogue_voice_ids=dialogue_voice_ids,
                    attribution=attribution, outdir=args.outdir)


if __name__ == "__main__":
    # `import fb2_reader` from the GUI must reuse this running module, not load a 2nd copy
    sys.modules.setdefault("fb2_reader", sys.modules["__main__"])
    main()
