r"""Фильтр списка новых слов после dry_run_chapters.py: оставляет слова, где ударение цепочки
расходится с независимым корпусом (stress_corpus_ru.tsv.gz), плюс слова, которых в корпусе нет.

Запуск из корня:
  .venv\Scripts\python.exe -I stress-listening-session\flag_vs_corpus.py ПАПКА_dry_run 14 15
Читает ПАПКА/words_new.txt и ПАПКА/chN.txt, пишет ПАПКА/flag_new.txt (расхождения + контекст)
и ПАПКА/miss_new.txt (нет в корпусе, без «ё»). Читать файлы через Read: консоль портит кириллицу.
Корпус хранит ОДНО чтение омографа и не знает «ё»: расхождение — повод посмотреть контекст, не ошибка.
"""
import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    d = Path(sys.argv[1])
    ns = [int(a) for a in sys.argv[2:]]
    corpus = {}
    for line in gzip.open(ROOT / "stress_corpus_ru.tsv.gz", "rt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            corpus[p[0].lower()] = p[1].lower().replace("^", "+")
    words = (d / "words_new.txt").read_text(encoding="utf-8").split()
    texts = {n: (d / f"ch{n}.txt").read_text(encoding="utf-8") for n in ns}
    flagged = []
    for w in words:
        c = corpus.get(w.replace("+", ""))
        if c and c.replace("ё", "е") != w.replace("ё", "е"):
            if "ё" in w and "+" not in c:
                continue
            ctx = ""
            for n, t in texts.items():
                m = re.search(r"(?<![а-яё+])" + re.escape(w) + r"(?![а-яё+])", t, re.I)
                if m:
                    ctx = f"ch{n}: " + t[max(0, m.start() - 45):m.end() + 30].replace("\n", " ")
                    break
            flagged.append(f"{w} (корпус {c}) | {ctx}")
    (d / "flag_new.txt").write_text("\n".join(flagged), encoding="utf-8")
    miss = [w for w in words if w.replace("+", "") not in corpus and "ё" not in w]
    (d / "miss_new.txt").write_text(" ".join(miss), encoding="utf-8")
    print(len(words), len(flagged), len(miss))


if __name__ == "__main__":
    main()
