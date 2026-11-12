r"""Какие главы затронуты правками словарей (для выборочного пересинтеза).

Запуск из корня:
  .venv\Scripts\python.exe -I stress-listening-session\affected_chapters.py <git-ревизия> [--book pepel-pobedy.fb2]
Сравнивает forced_stress_words.json, <книга>.manual_stress_overrides.json и pronunciation_substitutions.json
в рабочей копии с ревизией, ищет изменённые ключи (слова и фразы, е = ё, любые пробелы) в тексте каждой главы.
Печатает в файл affected.txt рядом (cwd): «<номер в списке, 1-based = --chapters> <название>: ключи».
НОМЕР В СПИСКЕ = N главы + 1 (предисловие = 1), он же NNN в имени NNN_Глава N.wav.
Ограничение: фразовые ключи входят в отпечаток КАЖДОЙ главы, поэтому программа сама пересинтезировала бы всю книгу;
этот список — «где правка реально слышна».
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fb2_reader as f  # noqa: E402


def main():
    rev = sys.argv[1]
    book = "pepel-pobedy.fb2"
    if "--book" in sys.argv:
        book = sys.argv[sys.argv.index("--book") + 1]
    files = ("forced_stress_words.json", Path(book).with_suffix(".manual_stress_overrides.json").name,
             "pronunciation_substitutions.json")
    changed = {}
    for p in files:
        raw = subprocess.run(["git", "show", f"{rev}:{p}"], capture_output=True, encoding="utf-8", cwd=ROOT).stdout
        old = json.loads(raw) if raw.strip() else {}
        new = json.loads((ROOT / p).read_text(encoding="utf-8"))
        for k, v in new.items():
            if not k.startswith("_") and old.get(k) != v:
                changed[k] = p
    _, chapters = f.parse_fb2(ROOT / book)
    norm = lambda s: re.sub(r"\s+", " ", s.lower().replace("ё", "е"))
    out = [f"changed entries: {len(changed)}"]
    for i, (title, text) in enumerate(chapters):
        nx = norm(re.sub(r"[^\w\s-]", " ", text))
        hit = [k for k in changed
               if re.search(r"(?<![\w-])" + re.escape(norm(k)).replace(r"\ ", " ") + r"(?![\w-])", nx)]
        if hit:
            out.append(f"{i + 1} {title}: {', '.join(hit)}")
    Path("affected.txt").write_text("\n".join(out), encoding="utf-8")
    print(len(out) - 1, "chapters")


if __name__ == "__main__":
    main()
