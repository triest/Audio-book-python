#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_ambiguous_stress.py — предварительный текстовый анализ книги на предмет
слов, у которых ударение зависит от контекста (омографы), ДО озвучки.

Идея: вместо того чтобы ловить неправильные ударения на слух в 16-часовой
аудиокниге, этот скрипт сканирует ИСХОДНЫЙ ТЕКСТ .fb2 за секунды и находит
все места, где встречаются слова из ambiguous_stress_words_ru.txt (общий
список омографов) и из встроенных в fb2_reader.py списков
_CASE_AMBIGUOUS_NOUNS / _LEMMA_HOMOGRAPHS (слова, для которых программа уже
пытается сама разрешить ударение по контексту).

Использование:
    python check_ambiguous_stress.py книга.fb2 [--out отчёт.txt] [--top 50]

Отчёт делится на два раздела:
  1) "Автоматически разрешаемые слова" — те, что входят в
     _CASE_AMBIGUOUS_NOUNS/_LEMMA_HOMOGRAPHS. Для каждого вхождения показан
     выбор, который сделает программа (gen_sg/nom_pl/лемма) и фрагмент
     предложения — можно быстро глазами проверить, не ошиблась ли автоматика,
     не дожидаясь синтеза.
  2) "Слова без автоматики" — которые есть в общем списке омографов
     (ambiguous_stress_words_ru.txt), но программа для них просто отдаёт
     решение RUAccent целиком, без подсказки. Список отсортирован по частоте
     встречаемости в книге — стоит в первую очередь послушать/проверить самые
     частые, а не пытаться вручную разобрать всю книгу.

Ничего не меняет ни в тексте, ни в словарях — только отчёт для чтения.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fb2_reader as fr  # noqa: E402


def sentence_context(text: str, start: int, end: int, radius: int = 60) -> str:
    """Небольшой фрагмент вокруг найденного слова, обрезанный по границам
    предложения там, где это удобно, иначе просто по числу символов."""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\n", " ")
    snippet = " ".join(snippet.split())
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def analyze_book(book_path: Path, top: int = 200, max_examples: int = 3,
                  progress_cb=None) -> "tuple[str, dict]":
    """Основная логика отчёта, вынесена отдельно от CLI (main), чтобы её
    можно было напрямую вызвать из GUI (fb2_reader_gui.py), без запуска
    отдельного процесса. Возвращает (текст_отчёта, статистика), ничего не
    пишет на диск сама — это решает вызывающий код. progress_cb(str),
    если передан, получает короткие текстовые статусы по ходу работы (для
    вывода в журнал GUI).

    stats, помимо счётчиков, содержит "plain_words" — список [(слово,
    частота, [(глава, контекст), ...]), ...], отсортированный по убыванию
    частоты — GUI использует его, чтобы построить диалог ручной простановки
    ударений, не разбирая заново текстовый отчёт."""
    def report_progress(msg):
        if progress_cb:
            progress_cb(msg)

    report_progress(f"Читаю книгу: {book_path}")
    book_title, chapters = fr.parse_fb2(book_path)
    report_progress(f"«{book_title}», глав: {len(chapters)}")

    ambiguous_words = fr._load_ambiguous_stress_words()
    auto_case_words = set(fr._CASE_AMBIGUOUS_NOUNS.keys())
    auto_lemma_words = set(fr._LEMMA_HOMOGRAPHS.keys())
    auto_words = auto_case_words | auto_lemma_words
    # слова, для которых пользователь уже сам указал ударение раньше — они
    # больше не "сомнительные", повторно спрашивать не нужно. Проверяем ОБА
    # источника: общий (manual_stress_overrides.json, действует для всех
    # книг сразу) и книжный (книга.manual_stress_overrides.json рядом с
    # книгой — ударения, назначенные конкретно для этой книги, могут не
    # совпадать с общими, если слово в разных книгах читается по-разному).
    manually_answered_words = (
        set(fr._load_manual_stress_overrides().keys())
        | set(fr._load_manual_stress_overrides(
            fr.book_manual_stress_overrides_path(book_path)).keys())
    )

    # слова из общего списка омографов, для которых НЕТ автоматики И которые
    # ещё не были разрешены пользователем вручную ранее
    plain_ambiguous_words = ambiguous_words - auto_words - manually_answered_words

    auto_hits = defaultdict(list)   # word -> [(chapter, context, choice), ...]
    plain_counts = Counter()
    plain_examples = defaultdict(list)  # word -> [(chapter, context), ...]

    for chapter_title, chapter_text in chapters:
        for m in fr._STRESS_WORD_RE.finditer(chapter_text):
            word_lower = m.group(0).lower()
            if word_lower in auto_words:
                if word_lower in auto_case_words:
                    prefix = chapter_text[:m.start()]
                    has_gen = bool(fr._GENITIVE_CONTEXT_RE.search(prefix)
                                   or fr._COUNTING_CONTEXT_RE.search(prefix))
                    has_pl = bool(fr._PLURAL_CONTEXT_RE.search(prefix)
                                  or fr._PLURAL_ADJECTIVE_CONTEXT_RE.search(prefix))
                    if has_gen and not has_pl:
                        choice_desc = "род.п. ед.ч. (по предлогу/числительному рядом)"
                    elif has_pl and not has_gen:
                        choice_desc = "им.п. мн.ч. (по местоимению/прилагательному рядом)"
                    else:
                        choice = fr._resolve_case_ambiguous_noun(word_lower)
                        choice_desc = {
                            "gen_sg": "род.п. ед.ч. (по pymorphy3, без явной подсказки рядом)",
                            "nom_pl": "им.п. мн.ч. (по pymorphy3, без явной подсказки рядом)",
                            None: "НЕ УДАЛОСЬ ОПРЕДЕЛИТЬ — слово останется как есть",
                        }[choice]
                elif word_lower in auto_lemma_words:
                    repl = fr._resolve_lemma_homograph(word_lower)
                    choice_desc = f"выбран вариант: {repl!r}" if repl else "НЕ УДАЛОСЬ ОПРЕДЕЛИТЬ"
                ctx = sentence_context(chapter_text, m.start(), m.end())
                auto_hits[word_lower].append((chapter_title, ctx, choice_desc))
            elif word_lower in plain_ambiguous_words:
                plain_counts[word_lower] += 1
                if len(plain_examples[word_lower]) < max_examples:
                    ctx = sentence_context(chapter_text, m.start(), m.end())
                    plain_examples[word_lower].append((chapter_title, ctx))

    lines = []
    lines.append(f"Отчёт по книге: {book_title}")
    lines.append(f"Файл: {book_path}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("РАЗДЕЛ 1. Слова, для которых программа САМА пытается разрешить")
    lines.append("ударение по контексту (см. _CASE_AMBIGUOUS_NOUNS/_LEMMA_HOMOGRAPHS")
    lines.append("в fb2_reader.py). Проверьте глазами, что выбор не выглядит странно —")
    lines.append("это можно сделать быстрее, чем слушать всю книгу целиком.")
    lines.append("=" * 78)
    if not auto_hits:
        lines.append("(в тексте не встретилось ни одного такого слова)")
    for word in sorted(auto_hits):
        occurrences = auto_hits[word]
        lines.append("")
        lines.append(f'"{word}" — встречается {len(occurrences)} раз(а):')
        shown = occurrences[:max_examples]
        for chapter_title, ctx, choice_desc in shown:
            lines.append(f"  [{chapter_title}] {choice_desc}")
            lines.append(f"    {ctx}")
        if len(occurrences) > len(shown):
            lines.append(f"  … и ещё {len(occurrences) - len(shown)} раз(а)")

    lines.append("")
    lines.append("=" * 78)
    lines.append("РАЗДЕЛ 2. Слова из общего списка омографов БЕЗ автоматики — программа")
    lines.append("целиком полагается на RUAccent, готовых правил для них нет. Отсортировано")
    lines.append("по частоте: сначала проверяйте самые частые — это даёт наибольший эффект")
    lines.append("при наименьших усилиях.")
    lines.append("=" * 78)
    if not plain_counts:
        lines.append("(в тексте не встретилось ни одного такого слова)")
    for word, count in plain_counts.most_common(top):
        lines.append("")
        lines.append(f'"{word}" — {count} раз(а)')
        for chapter_title, ctx in plain_examples[word]:
            lines.append(f"  [{chapter_title}] {ctx}")

    remaining = len(plain_counts) - min(len(plain_counts), top)
    if remaining > 0:
        lines.append("")
        lines.append(f"… и ещё {remaining} слов(о) реже встречающихся не показано "
                      f"(увеличьте --top, чтобы увидеть все).")

    lines.append("")
    lines.append("=" * 78)
    lines.append("ИТОГО")
    lines.append("=" * 78)
    lines.append(f"Автоматически обрабатываемых слов (раздел 1): {len(auto_hits)} уникальных, "
                  f"{sum(len(v) for v in auto_hits.values())} вхождений всего")
    lines.append(f"Слов без автоматики (раздел 2): {len(plain_counts)} уникальных, "
                  f"{sum(plain_counts.values())} вхождений всего")
    if manually_answered_words:
        lines.append(f"Уже отвечено вручную раньше (не показано в разделе 2, общий "
                      f"словарь + этой книги): {len(manually_answered_words)} слов(о)")
    lines.append("")
    lines.append("Если какое-то слово из раздела 2 регулярно встречается и его ударение")
    lines.append("реально зависит от контекста в этой книге (не только теоретически) —")
    lines.append("пришлите его мне вместе с примером из отчёта, и я добавлю для него")
    lines.append("правило в _CASE_AMBIGUOUS_NOUNS или _LEMMA_HOMOGRAPHS, как со «звезда»")
    lines.append("и «граф». Слова, которые встречаются 1 раз и не искажают смысл при")
    lines.append("любом ударении, обычно можно спокойно пропустить.")

    stats = {
        "book_title": book_title,
        "auto_unique": len(auto_hits),
        "auto_total": sum(len(v) for v in auto_hits.values()),
        "plain_unique": len(plain_counts),
        "plain_total": sum(plain_counts.values()),
        "plain_words": [
            (word, count, plain_examples[word])
            for word, count in plain_counts.most_common()
        ],
        "book_overrides_path": str(fr.book_manual_stress_overrides_path(book_path)),
    }
    return "\n".join(lines), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", type=Path, help="Путь к .fb2 (или .fb2.zip) файлу книги")
    ap.add_argument("--out", type=Path, default=None,
                     help="Куда сохранить отчёт (по умолчанию рядом с книгой, .ambiguous_stress_report.txt)")
    ap.add_argument("--top", type=int, default=200,
                     help="Сколько самых частых неавтоматических слов показать целиком с примерами (по умолчанию 200)")
    ap.add_argument("--max-examples", type=int, default=3,
                     help="Сколько примеров контекста показывать на каждое слово (по умолчанию 3)")
    args = ap.parse_args()

    if not args.book.exists():
        print(f"Файл не найден: {args.book}")
        return 1

    out_path = args.out
    if out_path is None:
        out_path = Path(str(args.book.with_suffix("")) + ".ambiguous_stress_report.txt")

    report_text, stats = analyze_book(args.book, top=args.top, max_examples=args.max_examples,
                                       progress_cb=print)

    out_path.write_text(report_text, encoding="utf-8")
    print(f"Отчёт сохранён: {out_path}")
    print(f"Раздел 1 (автоматика): {stats['auto_unique']} уникальных слов")
    print(f"Раздел 2 (без автоматики): {stats['plain_unique']} уникальных слов, "
          f"{stats['plain_total']} вхождений всего")
    return 0


if __name__ == "__main__":
    sys.exit(main())
