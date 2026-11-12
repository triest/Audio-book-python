"""Сухой прогон цепочки ударений по главам книги + список НОВЫХ слов с «+».

Запуск из корня проекта:
  .venv\Scripts\python.exe -I stress-chapter-review\dry_run_chapters.py ВЫХОДНАЯ_ПАПКА 20 45 [--book pepel-pobedy.fb2]

В ВЫХОДНОЙ_ПАПКЕ: chN.txt (глава с «+»), words_new.txt (уникальные слова с «+», которых ещё не было
в words_seen.txt этой папки), words_seen.txt (накопительный список уже просмотренных слов).
Номер N = индекс в списке глав = «Глава N» в pepel-pobedy.fb2 (проверено для 15, 20, 28, 40, 45).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fb2_reader as f  # noqa: E402


def main():
    args = sys.argv[1:]
    book = ROOT / "pepel-pobedy.fb2"
    if "--book" in args:
        i = args.index("--book")
        book = ROOT / args[i + 1]
        del args[i:i + 2]
    out_dir = Path(args[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(a) for a in args[1:]]
    overrides = book.with_suffix(".manual_stress_overrides.json")
    _, chapters = f.parse_fb2(book)
    seen_path = out_dir / "words_seen.txt"
    seen = set(seen_path.read_text(encoding="utf-8").split()) if seen_path.exists() else set()
    found = set()
    for n in nums:
        title, text = chapters[n]
        out, info = f.stress_dry_run(text, book_overrides_path=str(overrides) if overrides.exists() else None)
        (out_dir / f"ch{n}.txt").write_text(title + "\n" + out, encoding="utf-8")
        found.update(w.lower() for w in re.findall(r"[А-Яа-яЁё\-]*\+[А-Яа-яЁё\-+]*", out))
        print(n, info, file=sys.stderr)
    new = sorted(found - seen)
    (out_dir / "words_new.txt").write_text(" ".join(new), encoding="utf-8")
    seen_path.write_text(" ".join(sorted(seen | found)), encoding="utf-8")
    print(f"слов с «+»: {len(found)}, новых: {len(new)}", file=sys.stderr)


if __name__ == "__main__":
    main()
