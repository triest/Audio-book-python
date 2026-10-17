---
name: stress-book-run
description: Полный (инкрементальный) прогон книги через silero_rest в audiobook_output5 после правок словарей ударений, проверка готовых глав (звук, ASR-сверка, A/B-фрагменты для прослушивания) и разбор жалобы «в главе N слово читается не так». Использовать, когда пользователь просит «запусти всю книгу», «проверь готовые главы», «в главе 45 снова X с неверным ударением».
---

# Прогон книги и проверка готовых глав

Дополняет `stress-chapter-review` (поиск ошибок сухим прогоном) и `stress-dictionary-audit` (разбор словаря).

## 1. Прогон всей книги (инкрементально)

Из корня проекта, в фоне (`run_in_background`), лог в scratchpad. Сервис Silero должен слушать порт 5010 (`netstat -ano | grep :5010`):

```
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe fb2_reader.py pepel-pobedy.fb2 --mode silero_rest --model v5_6_ru --speaker xenia ^
  --sample-rate 48000 --sentence-break-ms 270 --paragraph-break-ms 490 --comma-break-ms 120 --outdir audiobook_output5
```

- Параметры взяты из `*.wav.meta.json` готовых глав (mode/model/speaker/sample_rate/breaks). Меняешь параметры — пересинтезируется ВСЯ книга.
- Папка `audiobook_output5`: главы с совпавшим отпечатком пропускаются («Пропускаю (уже озвучено…)»), пересинтезируются только главы, у которых изменился хэш затронутых записей словарей/подмен. Поэтому правка словаря → пересинтез 3–7 глав, не 50.
- `python.exe -I` с `fb2_reader.py` НЕ работает (не находит `app_paths`): флаг `-I` только для скриптов из scratchpad/`stress-chapter-review`, которые сами добавляют корень в `sys.path`.
- `--help` падает на cp1251 — перенаправляй в файл с `PYTHONIOENCODING=utf-8`.
- Статус «failed» у фоновой задачи-обёртки (`nohup … &` + `tail`) НЕ значит, что синтез упал: проверь `tasklist`/`wmic process … | grep fb2_reader` и хвост лога. Ждать конца — фоновый `until ! tasklist | grep -q <PID>; do sleep 20; done` (голый `sleep` блокируется).
- Файл главы N называется `NNN_Глава N.wav`, где NNN = N + 1 (предисловие = 001).
- Итог: `grep -a "Озвучиваю"` в логе (список пересинтезированных глав), ошибки — `silero_rest_client.log` в outdir.

## 2. Проверка готовых глав

- **Звук** (без зависимостей): длительность, доля тишины, самая длинная пауза, паузы > 2 с, клиппинг, `stress_dict_digest` в метаданных. Норма для книги: тишина ~20%, самая длинная пауза ~1,2 с, клиппинга нет.
- **ASR-сверка:** `.venv\Scripts\python.exe tools\chapter_asr_check.py audiobook_output5 28 45 --watch одержат,ввергла` — покрытие/WER/CER/пропуски/зацикливание + фрагменты ±3 с вокруг слов в `audiobook_output5\_asr\clips`. Нужен `faster-whisper` (ставит `install.bat`; без него скрипт не запустится, вручную `pip` не ставить). ASR ударение НЕ проверяет.
- **Ударение в аудио проверяет только слух пользователя.** Сделай A/B-страницу: `tools\audio_stress_review.py make <папка> --from-json items.json`, `items.json` = `[{"sentence","word","candidates":["одерж+ат","од+ержат"]}]`, открыть `index.html`. Голос xenia, модель v5_6_ru.

## 3. «В главе N слово X читается не так»

1. Найти, что реально уходит в Silero. Сухой прогон одного абзаца может отличаться от реальной нарезки на куски (`_split_paragraphs_for_ssml(text, 700)`) — повтори цепочку по кускам подряд после `reset_stress_context()`: `apply_stress_dictionary_protected` → `apply_stress_layers(…, "silero", silero)` → `restore_stress_placeholders`, как в `_accented_text_for_rest` (`fb2_reader.py` ~5511).
2. Просмотреть ВСЕ источники по порядку приоритета: `forced_stress_words.json` → `pepel-pobedy.manual_stress_overrides.json` (фразы) → `manual_stress_overrides.json` → `stress_dictionary.json` (29 тыс. слов, там же «самой» → `с+амой`) → silero-stress/шлюз.
3. **Файл правок книги может содержать ошибки, созданные кнопкой «Анализ омографов книги»**: фразы вроде `на графа белой → на граф+а б+елой` читаются неверно (граф-титул = `гр+афа`), и они ПЕРЕБИВАЮТ модель. Проверять `grep` по корню слова во всех словарях и править значения Python-скриптом (`json.dump(ensure_ascii=False, indent=2)`), затем коммитить.
4. Если в текст ушло верное ударение, а пользователь слышит другое — не гадать: сказать, что текст верный, спросить минуту в аудио/версию файла (мог слушать файл до пересинтеза), дать A/B-пару.

## Правила

- Пересинтез и запись в `audiobook_output5` — только по просьбе «запусти книгу»; без просьбы только сухой прогон.
- Коммит/пуш — только по просьбе пользователя; не коммитить `fb2_reader_settings.json`, `dialogue_characters.json`, `audiobook_output*/`.
- Зависимости (faster-whisper и др.) — только строкой в `install.bat`.
- Сообщать факты (что в логе, что в метриках), без заверений «теперь точно звучит верно» — это подтверждает слух.
