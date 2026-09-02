"""Скачивает офлайн-корпус ударений русского языка (~1.68 млн слов и
словоформ) и кладёт его рядом со скриптом как stress_corpus_ru.tsv.gz —
именно этот файл использует fb2_reader.py (см. fetch_stress_from_offline_corpus
/ _load_offline_stress_corpus) как первый, самый быстрый источник ударений
для слов, которых нет в собственном stress_dictionary.json, — до того, как
(при необходимости) обращаться к Викисловарю через интернет. Наличие этого
файла резко снижает число обращений к Викисловарю на целую книгу, а
значит — и шанс упереться в его лимит запросов (HTTP 429).

Источник данных: https://github.com/Koziev/NLP_Datasets (Stress/all_accents.zip,
собран по Википедии/Викисловарю + грамматическому словарю словоформ).

Вызывается из install.bat при установке. Можно запустить и вручную:
    .venv\\Scripts\\python.exe download_stress_corpus.py

Ничего не бросает наружу и не завершает процесс с ошибкой: если скачать
не получилось (нет интернета, GitHub недоступен, репозиторий переехал и
т.п.), просто печатается предупреждение и скрипт завершается успешно —
fb2_reader.py и без этого файла работает, только реже находит ударения
мгновенно и локально (без офлайн-корпуса RUAccent/Викисловарь всё равно
подстрахуют)."""

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/Koziev/NLP_Datasets/master/Stress/all_accents.zip"
OUTPUT_PATH = Path(__file__).resolve().parent / "stress_corpus_ru.tsv.gz"
TIMEOUT = 120


def main() -> int:
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 1_000_000:
        print(f"Офлайн-корпус ударений уже на месте: {OUTPUT_PATH} "
              f"({OUTPUT_PATH.stat().st_size} байт) — пропускаю скачивание.")
        return 0

    print("Скачиваю офлайн-корпус ударений русского языка (~10 МБ, один раз, "
          f"источник: {SOURCE_URL}) ...")
    try:
        req = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "audiobook-fb2-reader-installer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            tsv_names = [n for n in zf.namelist() if n.lower().endswith(".tsv")]
            if not tsv_names:
                raise ValueError(f"в архиве не нашлось .tsv-файла (есть: {zf.namelist()})")
            with zf.open(tsv_names[0]) as f:
                raw = f.read()

        # Исходный формат — "слово<TAB>слово_с_^_перед_ударной_буквой".
        # Оставляем как есть (переводом "^" -> "+" занимается сам
        # fb2_reader.py при загрузке, _load_offline_stress_corpus) и просто
        # пережимаем в gzip, чтобы файл в репозитории/дистрибутиве весил
        # разумно (~10 МБ вместо ~75 МБ).
        tmp_path = OUTPUT_PATH.with_suffix(".tmp")
        with gzip.open(tmp_path, "wb", compresslevel=9) as gz:
            gz.write(raw)
        tmp_path.replace(OUTPUT_PATH)

        print(f"Готово: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} байт).")
        return 0
    except Exception as e:
        print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось скачать офлайн-корпус ударений "
              f"({type(e).__name__}: {e}). Это не критично — программа "
              "продолжит работать без него, ударения будут расставляться "
              "RUAccent'ом и (если доступен интернет) Викисловарём. Можно "
              "попробовать позже: .venv\\Scripts\\python.exe download_stress_corpus.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
