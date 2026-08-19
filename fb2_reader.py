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
  pip install torch torchaudio omegaconf numpy

  # для режима silero_rest (клиент к REST-сервису):
  pip install requests numpy

  # для остальных режимов:
  pip install gTTS pyttsx3 pygame lxml pydub

Для offline-режима (pyttsx3) на Linux дополнительно нужен espeak-ng:
  sudo apt install espeak-ng

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

Доступные голоса Silero (--speaker): aidar (муж.), baya (жен.),
kseniya (жен.), xenia (жен.), eugene (муж.), random (случайный на каждой фразе).
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

try:
    from lxml import etree
    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAVE_LXML = False


FB2_NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}


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


def synth_online(text: str, out_path: Path, lang="ru"):
    from gtts import gTTS
    # gTTS ограничивает длину — разбиваем на куски по предложениям
    max_len = 4500
    chunks = split_text(text, max_len)
    if len(chunks) == 1:
        gTTS(text=chunks[0], lang=lang).save(str(out_path))
        return
    # склеиваем несколько mp3 через pydub, если он есть; иначе сохраняем по частям
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        tmp_files = []
        for i, chunk in enumerate(chunks):
            tmp = out_path.with_suffix(f".part{i}.mp3")
            gTTS(text=chunk, lang=lang).save(str(tmp))
            combined += AudioSegment.from_mp3(str(tmp))
            tmp_files.append(tmp)
        combined.export(str(out_path), format="mp3")
        for t in tmp_files:
            t.unlink(missing_ok=True)
    except ImportError:
        # Без pydub — сохраняем части отдельными файлами
        for i, chunk in enumerate(chunks):
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


def play_file(path: Path):
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def run_online(chapters, outdir: Path, play: bool, start: int, voice_lang: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for idx, (title, text) in enumerate(chapters, 1):
        if idx < start:
            continue
        fname = f"{idx:03d}_{sanitize_filename(title)}.mp3"
        out_path = outdir / fname
        print(f"[{idx}/{len(chapters)}] Озвучиваю: {title} -> {fname}")
        synth_online(text, out_path, lang=voice_lang)
        if play:
            print("  Проигрывание...")
            play_file(out_path)


def run_silero(chapters, outdir: Path, start: int, speaker: str, sample_rate: int, play: bool):
    """Озвучка через Silero TTS — нейросетевой русский голос,
    хорошо передаёт интонацию и паузы по знакам препинания.
    При первом запуске модель (~50-100 МБ) скачивается и кэшируется torch.hub.
    """
    import torch

    print("Загружаю модель Silero TTS (при первом запуске — скачивание)...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    device = torch.device("cpu")
    model.to(device)

    outdir.mkdir(parents=True, exist_ok=True)

    # Silero имеет ограничение на длину текста за один вызов — разбиваем по абзацам/предложениям
    max_len = 900

    for idx, (title, text) in enumerate(chapters, 1):
        if idx < start:
            continue
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname
        print(f"[{idx}/{len(chapters)}] Озвучиваю: {title} -> {fname}")

        chunks = split_text(text, max_len)
        import numpy as np
        audio_parts = []
        pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)  # пауза между кусками

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                audio = model.apply_tts(
                    text=chunk,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
                audio_parts.append(audio.numpy())
                audio_parts.append(pause)
            except Exception as e:
                print(f"  Пропускаю фрагмент из-за ошибки синтеза: {e}")

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        import wave
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        if play:
            print("  Проигрывание...")
            play_file(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


# --------------------------------------------------------------------------
# SSML: расстановка интонационных пауз, вопросов/восклицаний
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SOFT_PAUSE_RE = re.compile(r"([,;:]|—|--)\s+")


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def text_to_ssml(text: str, sentence_break_ms: int = 320,
                  paragraph_break_ms: int = 550, comma_break_ms: int = 180) -> str:
    """Превращает обычный текст главы в SSML-документ для Silero.

    * абзацы -> <p>, между ними длинная пауза (paragraph_break_ms);
    * предложения -> <s>, между ними пауза покороче (sentence_break_ms);
      знаки "?" и "!" сохраняются как есть — по ним Silero строит
      вопросительную/восклицательную интонацию;
    * внутри предложения после запятых/тире/двоеточий/точек с запятой
      добавляется короткая пауза <break/> (comma_break_ms), имитирующая
      естественную интонационную паузу при чтении.
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
            s_chunks.append(f"<s>{escaped}</s>")
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
                     comma_break_ms: int, max_len: int = 700):
    """Озвучка через Silero-REST-Service (см. https://github.com/Flokss/Silero-REST-Service).

    Текст каждой главы автоматически превращается в SSML с интонационными
    паузами (см. text_to_ssml) и отправляется на эндпоинт /getssmlwav
    (нужен патч сервиса, добавляющий поддержку ssml_text/SSML — обычный
    /getwav SSML не понимает).
    """
    import io
    import requests
    import numpy as np
    import wave as wave_mod

    rest_url = rest_url.rstrip("/")
    endpoint = f"{rest_url}/getssmlwav"
    outdir.mkdir(parents=True, exist_ok=True)

    def synth_ssml_chunk(plain_text_chunk: str) -> bytes:
        ssml = text_to_ssml(
            plain_text_chunk,
            sentence_break_ms=sentence_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
        )
        resp = requests.get(endpoint, params={
            "text_to_speech": ssml,
            "speaker": speaker,
            "sample_rate": sample_rate,
            "raw_ssml": "true",
        }, timeout=300)
        resp.raise_for_status()
        return resp.content

    for idx, (title, text) in enumerate(chapters, 1):
        if idx < start:
            continue
        fname = f"{idx:03d}_{sanitize_filename(title)}.wav"
        out_path = outdir / fname
        print(f"[{idx}/{len(chapters)}] Озвучиваю (silero_rest): {title} -> {fname}")

        chunks = _split_paragraphs_for_ssml(text, max_len)
        pause = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
        audio_parts = []

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                wav_bytes = synth_ssml_chunk(chunk)
                with wave_mod.open(io.BytesIO(wav_bytes), "rb") as wf:
                    n = wf.getnframes()
                    pcm = wf.readframes(n)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                audio_parts.append(audio)
                audio_parts.append(pause)
            except Exception as e:
                print(f"  Пропускаю фрагмент из-за ошибки синтеза: {e}")

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)
        with wave_mod.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            pcm = (full_audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)

        if play:
            print("  Проигрывание...")
            play_file(out_path)

    print(f"\nГотово. Файлы сохранены в: {outdir.resolve()}")


def run_offline(chapters, start: int, rate: int, voice_hint: str):
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)

    # Пытаемся найти русский голос
    voices = engine.getProperty("voices")
    chosen = None
    for v in voices:
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

    for idx, (title, text) in enumerate(chapters, 1):
        if idx < start:
            continue
        print(f"\n[{idx}/{len(chapters)}] {title}")
        engine.say(text)
        engine.runAndWait()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Озвучивание FB2-книг на русском языке")
    ap.add_argument("book", type=Path, help="путь к .fb2 или .fb2.zip файлу")
    ap.add_argument("--mode", choices=["online", "offline", "silero", "silero_rest"], default="silero",
                     help="silero = нейросетевой голос локально через torch.hub; "
                          "silero_rest = синтез через Silero-REST-Service с интонационными паузами "
                          "(SSML: паузы на запятых/тире, границах предложений и абзацев, интонация "
                          "вопросов/восклицаний); online = gTTS (интернет); offline = pyttsx3 (без интернета)")
    ap.add_argument("--outdir", type=Path, default=Path("audiobook_output"),
                     help="папка для сохранения аудио (online и silero режимы)")
    ap.add_argument("--play", action="store_true",
                     help="сразу проигрывать главы после озвучки (online и silero режимы)")
    ap.add_argument("--start", type=int, default=1, help="с какой главы начать (1 = с начала)")
    ap.add_argument("--rate", type=int, default=170, help="скорость речи для offline-режима (слов/мин)")
    ap.add_argument("--voice", type=str, default="", help="подсказка имени голоса для offline-режима")
    ap.add_argument("--speaker", type=str, default="xenia",
                     help="голос для silero-режима: aidar, baya, kseniya, xenia, eugene, random (по умолчанию xenia)")
    ap.add_argument("--sample-rate", type=int, default=48000,
                     help="частота дискретизации для silero-режима (8000/24000/48000)")
    ap.add_argument("--list", action="store_true", help="только показать список глав и выйти")
    ap.add_argument("--rest-url", type=str, default="http://localhost:5010",
                     help="адрес Silero-REST-Service для режима silero_rest")
    ap.add_argument("--sentence-break-ms", type=int, default=320,
                     help="пауза между предложениями в silero_rest-режиме (мс)")
    ap.add_argument("--paragraph-break-ms", type=int, default=550,
                     help="пауза между абзацами в silero_rest-режиме (мс)")
    ap.add_argument("--comma-break-ms", type=int, default=180,
                     help="пауза на запятых/тире/двоеточиях в silero_rest-режиме (мс)")
    args = ap.parse_args()

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

    if args.mode == "online":
        run_online(chapters, args.outdir, args.play, args.start, voice_lang="ru")
        print(f"\nГотово. Файлы сохранены в: {args.outdir.resolve()}")
    elif args.mode == "silero":
        run_silero(chapters, args.outdir, args.start, args.speaker, args.sample_rate, args.play)
    elif args.mode == "silero_rest":
        run_silero_rest(chapters, args.outdir, args.start, args.speaker, args.sample_rate, args.play,
                         args.rest_url, args.sentence_break_ms, args.paragraph_break_ms,
                         args.comma_break_ms)
    else:
        run_offline(chapters, args.start, args.rate, args.voice)


if __name__ == "__main__":
    main()