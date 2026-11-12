---
name: stress-debug
description: Отладка модуля ударений — прогон всех проверок (test_accentuation, test_book_pipeline, stress_benchmark, check_stress_versions), разбор падений теста (устаревшее ожидание vs. реальная ошибка цепочки). Использовать на «отладь ударения», «доведи ударения до конца», «тесты ударений красные».
---

# Отладка ударений

Из корня проекта, `set PYTHONIOENCODING=utf-8` (иначе вывод в cp1251 нечитаем; в bash — `export`). Вывод пиши в файл и `grep -a`.

## Прогон
```
.venv\Scripts\python.exe tools\test_accentuation.py      # ~390 проверок; итог «N passed, M failed»
.venv\Scripts\python.exe tools\test_book_pipeline.py     # реальная книга, без синтеза; «ALL OK»
.venv\Scripts\python.exe tools\stress_benchmark.py       # золото 67/68; MISS «феномен» — известный
.venv\Scripts\python.exe tools\check_stress_versions.py  # пакеты, SHA-256, офлайн
```
`grep -a "FAIL\|passed"` по логу теста.

## Разбор падения: сначала решить, что устарело — тест или код
1. Тест делает ту же цепочку, что озвучка: `run_chain` = словарь → `apply_stress_layers` → `restore_stress_placeholders` → `strip_monosyllable_stress`. Если тест ждёт «лет» без «+», а получил «л+ет» — смотри `_MONO_KEEP_STRESS` (там «лет», «пять», «сто» намеренно сохраняют метку, чтобы не было «лёт»).
2. Ожидания `XENIA_CASES` читают РЕАЛЬНЫЙ `forced_stress_words.json` / `stress_dictionary.json`. Если словарь правили позже теста (коммит «fix stress»), а ожидание старое — сверь со стандартным ударением и словарём (тр+анспорт, тк+аней, основн+ое, брон+я, пл+ена) и обнови тест, не словарь.
3. `apply_stress_layers` возвращает `_text_guard(out, global_prose_sanitizer(out), "sanitizer")`; тест «sanitizer is LAST» сверяет эту строку — при смене хвоста функции обновляй обе стороны.
4. Значения сид-словаря: по одной «+» на часть слова через дефис (`мал+иново-б+ело-голуб+ая`).
5. Настоящая ошибка цепочки (слово не то, а ожидание верное) → правило в `_PROSE_WORDS` (слово с одним чтением) или запись в `forced_stress_words.json`; после — снова все четыре прогона и проверка корпуса (тест `_PROSE_WORDS vs independent corpus`).

## Правила
- Не подгонять код под устаревший тест и наоборот, не сверив с корпусом/словарём.
- Не затирать тесты через `git checkout`-скрипты вслепую: при правке срезами по `s.index` проверяй `count==1` (однажды продублировался кусок файла).
- Ударение в аудио подтверждает только слух пользователя; тесты его не проверяют.
