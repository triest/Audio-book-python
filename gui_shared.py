"""Общие константы и мелкие функции, на которые опираются и fb2_reader_gui.py
(класс AudiobookApp), и его миксины (gui_service_management.py,
gui_voice_profiles.py, gui_player.py, gui_ui_builder.py).

ВАЖНО: этот модуль НЕ импортирует fb2_reader_gui — специально, чтобы разорвать
циклический импорт. Раньше миксины делали `from fb2_reader_gui import (...)`,
что при запуске программы напрямую (`python fb2_reader_gui.py`, как делает
run_gui.bat) ломалось с ImportError: когда fb2_reader_gui.py на строке импорта
миксинов ещё не дописан до конца (выполняется как __main__, а не как модуль
fb2_reader_gui в sys.modules), миксин пытался заново импортировать
fb2_reader_gui с диска — и упирался в тот же самый импорт миксина по кругу.
Поэтому всё, что нужно и AudiobookApp, и миксинам, вынесено сюда, в модуль без
обратных зависимостей."""

import sys
import tkinter as tk
from pathlib import Path


def _app_dir() -> Path:
    """Папка, где лежит программа. В обычном запуске (python fb2_reader_gui.py)
    это папка со скриптом. В собранной PyInstaller-версии (.exe) __file__
    указывает на временную папку распаковки, которая меняется при каждом
    запуске — поэтому там нужно брать папку, где лежит сам .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _make_logs_dir() -> Path:
    """Отдельная папка "logs" рядом с программой — куда пишется файл
    журнала. Если её почему-то не удаётся создать там (например, программа
    лежит в защищённой системной папке без прав на запись — на некоторых
    компьютерах это Program Files), используем временную папку системы как
    запасной вариант, чтобы журнал всё равно куда-то писался, а не терялся
    молча."""
    primary = _app_dir() / "logs"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        probe = primary / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return primary
    except Exception:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "fb2_reader_logs"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return fallback


LOGS_DIR = _make_logs_dir()
LOG_FILE_PATH = LOGS_DIR / "fb2_reader_gui.log"
_log_file_handle = None
_log_file_open_failed = False


LOG_FILE_MAX_BYTES = 2_000_000  # ~2 МБ — после этого старый файл переносится в .1, чтобы не расти бесконечно

# Должно совпадать с SERVICE_VERSION в cosyvoice_rest_service.py. Программа
# сверяет его через /health и, если уже запущенный сервис отвечает старой
# версией, сама перезапускает его - иначе после обновления файлов сервис
# продолжал бы молча работать со старым кодом, пока пользователь не
# перезапустит компьютер вручную (именно так сломалось удаление голосов).
COSYVOICE_EXPECTED_SERVICE_VERSION = "2026-08-23.10"

# Движки синтеза, которые умеет запускать cosyvoice_rest_service.py (см. его
# ENGINE/TTS_ENGINE) - код (ключ словаря) идёт в переменную окружения
# TTS_ENGINE процесса сервиса и сравнивается со значением, которое сервис
# сам сообщает в /health ("engine"), человекочитаемая подпись - то, что
# видит пользователь в выпадающем списке на вкладке «Голоса CosyVoice».
COSYVOICE_ENGINE_LABELS = {
    "f5": "F5-TTS-Russian",
    "espeech": "ESpeech RL-V2 (F5-based, меньше шума, рекомендуется)",
    "f5winter": "F5-TTS-Russian winter (свежий чекпоинт, с ударениями)",
    "xtts": "XTTS-v2 (запасной, шумнее)",
    "cosyvoice3": "CosyVoice 3 (экспериментально, отдельное окружение)",
}
COSYVOICE_ENGINE_CODES = {label: code for code, label in COSYVOICE_ENGINE_LABELS.items()}
COSYVOICE_DEFAULT_ENGINE = "espeech"

# CosyVoice3, в отличие от остальных движков (f5/espeech/f5winter/xtts),
# запускается подпроцессом в СВОЁМ окружении .venv_cosyvoice3 (не в общем
# .venv_cosyvoice - см. cosyvoice3_rest_service.py и install.bat) - у него
# несовместимый со всем остальным набор фиксированных версий зависимостей.
# Порт и ожидаемая версия сервиса поэтому тоже отдельные.
COSYVOICE3_DEFAULT_REST_URL = "http://localhost:5012"
COSYVOICE3_EXPECTED_SERVICE_VERSION = "2026-08-24.15-transcribe-window-fix"


def _write_log_file(s: str):
    """Дублирует всё, что попадает в окно «Журнал» (и вообще все ошибки —
    см. _install_error_logging), в текстовый файл в папке logs/ рядом с
    программой — окно журнала в tkinter не всегда удобно копировать мышью,
    а из обычного файла (Блокнотом или чем угодно) можно скопировать что
    угодно без проблем.

    Файл ДОПИСЫВАЕТСЯ между запусками программы (а не перезаписывается с
    нуля каждый раз) — раньше он открывался в режиме "w" (перезапись), из-за
    чего при каждом новом запуске программы весь журнал предыдущего запуска
    стирался; если ошибка случалась в одном запуске, а посмотреть файл
    получалось только после следующего — важные строки уже пропадали.
    Теперь только при СТАРТЕ программы, если файл вырос больше ~2 МБ, он
    переименовывается в fb2_reader_gui.log.1 (затирая предыдущий .1) — так
    файл не растёт бесконечно, но и не обнуляется просто от перезапуска."""
    global _log_file_handle, _log_file_open_failed
    if _log_file_handle is None:
        try:
            if LOG_FILE_PATH.exists() and LOG_FILE_PATH.stat().st_size > LOG_FILE_MAX_BYTES:
                rotated = LOG_FILE_PATH.with_suffix(LOG_FILE_PATH.suffix + ".1")
                try:
                    rotated.unlink(missing_ok=True)
                    LOG_FILE_PATH.rename(rotated)
                except Exception:
                    pass  # не критично - просто продолжим дописывать в тот же файл
            _log_file_handle = open(LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)
            _log_file_handle.write(f"\n=== Запуск программы: {LOG_FILE_PATH} ===\n")
        except Exception:
            _log_file_handle = False  # не удалось открыть — больше не пробуем
            _log_file_open_failed = True
    if _log_file_handle:
        try:
            _log_file_handle.write(s)
        except Exception:
            pass


def _log_exception_to_file(context: str, exc_info=None):
    """Пишет в файл журнала полный traceback (не только текст ошибки) —
    используется и глобальным обработчиком необработанных исключений, и
    отдельными местами в коде, где ошибка перехватывается вручную (запуск
    сервисов, сетевые запросы и т.п.), чтобы для диагностики всегда было
    видно, что именно произошло и где."""
    import traceback
    if exc_info is None:
        exc_info = sys.exc_info()
    tb_text = "".join(traceback.format_exception(*exc_info)) if exc_info[0] else ""
    _write_log_file(f"\n--- ОШИБКА ({context}) ---\n{tb_text}\n")


def add_context_menu(widget):
    """Добавляет виджету Entry стандартное контекстное меню правой кнопкой
    мыши (Вырезать/Копировать/Вставить/Выделить всё) — у полей tkinter его
    нет по умолчанию, из-за чего вставка правой кнопкой мыши (в отличие от
    Ctrl+V) может выглядеть так, будто ничего не работает."""
    menu = tk.Menu(widget, tearoff=0)
    menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
    menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
    menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
    menu.add_separator()
    menu.add_command(label="Выделить всё", command=lambda: widget.select_range(0, "end"))

    def show_menu(event):
        widget.focus_set()
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    widget.bind("<Button-3>", show_menu)
    return widget
