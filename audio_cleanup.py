#!/usr/bin/env python3
# coding: utf-8
"""
audio_cleanup.py — умный аудио-компрессор и "чистильщик" тишины для уже
готовых озвученных глав (любой TTS-движок этого проекта: silero, cosyvoice,
piper, yandex, qwen_tts и т.д. — работает с готовым .wav/.mp3, ей всё равно,
чем он был синтезирован).

Решает две типичные проблемы "сырой" озвучки, особенно у локальных
движков вроде XTTSv2/CosyVoice3:

1. НЕРОВНАЯ ГРОМКОСТЬ — модель то говорит тише, то громче в пределах одной
   главы (особенно на стыках фрагментов/предложений, которые синтезировались
   отдельными вызовами модели и потом склеены). Слушать это на скорости
   1.5x-2x утомительно — то не слышно, то бьёт по ушам.
2. АНОМАЛЬНЫЕ ПАУЗЫ — модель иногда "задумывается"/оставляет тишину по
   несколько секунд там, где в тексте была просто точка или запятая.
   Раздувает длительность книги и сбивает темп при прослушивании.

Что делает скрипт:

- Ищет в тексте аудио неречевые участки (детектор тишины по dBFS,
  pydub.silence.detect_nonsilent) — получает список "речевых" кусков.
- Для КАЖДОГО речевого куска меряет его громкость (dBFS) и подтягивает её
  к общей целевой громкости (--target-dbfs), с ограничением максимального
  усиления/ослабления (--max-gain-db) — чтобы не "накачать" шум там, где
  и так было тихо специально (шёпот в диалоге и т.п.), и не выкрутить один
  громкий выкрик до клиппинга. Это и есть "динамическая нормализация" —
  в отличие от обычного pydub.normalize() (который смотрит только на ОДИН
  глобальный пик по всему файлу), здесь громкость выравнивается ЛОКАЛЬНО,
  кусок за куском. На стыке куска и подтянутой громкости — короткий
  fade (--fade-ms), чтобб не было слышно "щелчка" смены уровня.
- Любую паузу МЕЖДУ речевыми кусками длиннее --max-silence-ms обрезает до
  --target-silence-ms — но не вырезает начисто (не склеивает встык, это
  звучит неестественно и обрубает дыхание/интонацию), а берёт кусочек
  самой этой тишины (естественный "фоновый шум" записи) с её начала и
  конца и склеивает их — так остаётся естественная пауза без обрыва
  интонации, просто короче.
- Опционально (--trim-edges) обрезает лишнюю тишину в самом начале и конце
  файла (до --target-silence-ms).

По умолчанию НИЧЕГО не перезаписывает — исходные файлы остаются как есть,
результат пишется рядом с суффиксом (--suffix, по умолчанию "_clean") или
в отдельную папку (--outdir). Это соответствует общему подходу проекта:
не трогать то, что уже готово, без явного запроса (--in-place).

Использование:
    python3 audio_cleanup.py "audiobook_output/Моя книга"
    python3 audio_cleanup.py "audiobook_output/Моя книга" --in-place
    python3 audio_cleanup.py файл1.wav файл2.wav --target-dbfs -18 --max-silence-ms 600

Зависимости: pydub (уже используется в проекте, ставится install.bat).
Для .mp3 дополнительно нужен ffmpeg в PATH (как и везде в проекте, где
используется pydub) — pydub сам сообщит понятной ошибкой, если его нет.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
except ImportError:
    print("Не найден пакет pydub. Поставьте зависимости через install.bat "
          "(pydub уже входит в основной набор пакетов проекта).")
    sys.exit(1)


AUDIO_EXTS = (".wav", ".mp3")


def find_audio_files(paths: "list[str]") -> "list[Path]":
    """Принимает список файлов и/или папок с командной строки и разворачивает
    его в плоский список аудиофайлов (.wav/.mp3), папки читает нерекурсивно —
    ровно так же, как остальные скрипты этого проекта раскладывают главы
    (один файл на главу в одной папке)."""
    files: "list[Path]" = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS
                and not f.stem.endswith("_clean")  # не обрабатываем свои же прошлые результаты повторно
            )
            if not found:
                print(f"В папке {p} не нашлось .wav/.mp3 файлов — пропускаю.")
            files.extend(found)
        elif p.is_file():
            if p.suffix.lower() not in AUDIO_EXTS:
                print(f"Пропускаю {p} — не .wav/.mp3.")
                continue
            files.append(p)
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {p} не найден, пропускаю.")
    return files


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _shrink_silence(gap: "AudioSegment", target_ms: int) -> "AudioSegment":
    """Ужимает кусок тишины между репликами до target_ms, сохраняя
    естественный "фоновый шум" записи, а не вставляя мёртвую цифровую
    тишину — берёт начало и конец исходного гэпа (там, где затухает/только
    начинается тишина после/перед речью — естественнее на слух, чем
    середина) и короткой перекрёстной растушёвкой (fade) склеивает их."""
    if len(gap) <= target_ms:
        return gap
    half = target_ms // 2
    head = gap[:half]
    tail = gap[len(gap) - (target_ms - half):]
    fade = min(30, half, target_ms - half)
    if fade > 0:
        head = head.fade_out(fade)
        tail = tail.fade_in(fade)
    return head + tail


def clean_audio(
    audio: "AudioSegment",
    *,
    target_dbfs: float,
    max_gain_db: float,
    silence_thresh_db: float,
    min_silence_len_ms: int,
    max_silence_ms: int,
    target_silence_ms: int,
    fade_ms: int,
    trim_edges: bool,
) -> "AudioSegment":
    """Основная логика: см. подробное описание в шапке файла. Возвращает
    новый AudioSegment, исходный не модифицирует."""
    # detect_nonsilent работает от АБСОЛЮТНОГО порога в dBFS (не от пика
    # файла) — silence_thresh_db задаётся относительно 0 dBFS, как и dBFS
    # самого сегмента (типичная тихая цифровая тишина у TTS обычно ниже
    # -50 dBFS, обычная речь -25..-15 dBFS — поэтому разумный порог по
    # умолчанию около -40 dBFS отделяет одно от другого с запасом).
    spans = detect_nonsilent(
        audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db
    )
    if not spans:
        print("    Речь не обнаружена (весь файл ниже порога тишины) — оставляю без изменений.")
        return audio

    pieces: "list[AudioSegment]" = []

    # Ведущая тишина (до первого речевого куска)
    if spans[0][0] > 0:
        lead = audio[: spans[0][0]]
        pieces.append(_shrink_silence(lead, target_silence_ms) if trim_edges else lead)

    for i, (start, end) in enumerate(spans):
        chunk = audio[start:end]
        if len(chunk) > 0 and chunk.dBFS != float("-inf"):
            gain = _clamp(target_dbfs - chunk.dBFS, -max_gain_db, max_gain_db)
            if abs(gain) > 0.1:
                chunk = chunk.apply_gain(gain)
                f = min(fade_ms, len(chunk) // 2)
                if f > 0:
                    chunk = chunk.fade_in(f).fade_out(f)
        pieces.append(chunk)

        # Пауза ДО следующего речевого куска (или до конца файла, если это
        # последний кусок — тогда это "хвостовая" тишина).
        gap_end = spans[i + 1][0] if i + 1 < len(spans) else len(audio)
        gap = audio[end:gap_end]
        is_trailing = i + 1 >= len(spans)
        if len(gap) > max_silence_ms:
            if is_trailing and not trim_edges:
                pieces.append(gap)
            else:
                pieces.append(_shrink_silence(gap, target_silence_ms))
        else:
            pieces.append(gap)

    result = pieces[0]
    for piece in pieces[1:]:
        result += piece
    return result


def process_file(path: Path, args: argparse.Namespace) -> "Path | None":
    print(f"[{path.name}]")
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        print(f"    ОШИБКА: не удалось прочитать файл ({e}) — пропускаю.")
        return None

    before_ms = len(audio)
    before_dbfs = audio.dBFS

    cleaned = clean_audio(
        audio,
        target_dbfs=args.target_dbfs,
        max_gain_db=args.max_gain_db,
        silence_thresh_db=args.silence_thresh_db,
        min_silence_len_ms=args.min_silence_len_ms,
        max_silence_ms=args.max_silence_ms,
        target_silence_ms=args.target_silence_ms,
        fade_ms=args.fade_ms,
        trim_edges=args.trim_edges,
    )

    after_ms = len(cleaned)
    after_dbfs = cleaned.dBFS
    saved_s = (before_ms - after_ms) / 1000.0
    print(f"    Длительность: {before_ms/1000:.1f}с -> {after_ms/1000:.1f}с "
          f"({'-' if saved_s >= 0 else '+'}{abs(saved_s):.1f}с), "
          f"громкость: {before_dbfs:.1f} -> {after_dbfs:.1f} dBFS")

    if args.in_place:
        out_path = path
    elif args.outdir:
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / path.name
    else:
        out_path = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")

    fmt = "mp3" if out_path.suffix.lower() == ".mp3" else "wav"
    export_kwargs = {"format": fmt}
    if fmt == "mp3":
        export_kwargs["bitrate"] = "192k"
    try:
        cleaned.export(out_path, **export_kwargs)
    except Exception as e:
        print(f"    ОШИБКА: не удалось сохранить {out_path} ({e}).")
        return None
    print(f"    Сохранено: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Динамическая нормализация громкости + обрезка аномальных пауз "
                     "в уже готовых озвученных главах (.wav/.mp3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("paths", nargs="+",
                     help="файлы и/или папки с .wav/.mp3 (папка читается нерекурсивно, "
                          "один уровень — как обычно раскладываются главы этим проектом)")
    ap.add_argument("--target-dbfs", type=float, default=-20.0,
                     help="целевая громкость речевых кусков (dBFS)")
    ap.add_argument("--max-gain-db", type=float, default=12.0,
                     help="максимальное усиление/ослабление одного куска (дБ) — "
                          "ограничивает, чтобы не выкручивать специально тихие места "
                          "(шёпот) и не заводить шум там, где и так было тихо")
    ap.add_argument("--silence-thresh-db", type=float, default=-40.0,
                     help="порог, ниже которого участок считается тишиной (dBFS)")
    ap.add_argument("--min-silence-len-ms", type=int, default=200,
                     help="минимальная длина участка, чтобы вообще считать его тишиной "
                          "(короткие естественные микропаузы внутри слов не трогаем)")
    ap.add_argument("--max-silence-ms", type=int, default=800,
                     help="паузы длиннее этого порога считаются аномальными и обрезаются")
    ap.add_argument("--target-silence-ms", type=int, default=300,
                     help="до какой длины ужимать аномально длинные паузы")
    ap.add_argument("--fade-ms", type=int, default=15,
                     help="длина fade in/out на стыке при изменении громкости куска (мс) — "
                          "убирает слышимый \"щелчок\" смены уровня")
    ap.add_argument("--trim-edges", action="store_true",
                     help="также обрезать лишнюю тишину в самом начале и конце файла "
                          "(по умолчанию не трогается — так безопаснее для склейки глав)")
    ap.add_argument("--in-place", action="store_true",
                     help="ПЕРЕЗАПИСАТЬ исходные файлы (по умолчанию выключено — "
                          "результат пишется рядом с суффиксом --suffix)")
    ap.add_argument("--outdir", type=str, default="",
                     help="папка для результатов (вместо суффикса рядом с исходником)")
    ap.add_argument("--suffix", type=str, default="_clean",
                     help="суффикс имени файла для результата (игнорируется при --in-place/--outdir)")
    args = ap.parse_args()

    if args.in_place and args.outdir:
        ap.error("--in-place и --outdir несовместимы — выберите что-то одно.")

    files = find_audio_files(args.paths)
    if not files:
        print("Нечего обрабатывать.")
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}")
    ok = 0
    for f in files:
        if process_file(f, args) is not None:
            ok += 1
    print(f"\nГотово: {ok}/{len(files)} файлов обработано успешно.")


if __name__ == "__main__":
    main()
