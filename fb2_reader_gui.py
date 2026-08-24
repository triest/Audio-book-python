#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Графический интерфейс для fb2_reader.py — выбор книги, голоса и параметров озвучки."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fb2_reader import (
    DEFAULT_ATTRIBUTION_MODEL, ATTRIBUTION_PROVIDERS, DEFAULT_ATTRIBUTION_PROVIDER,
    TTS_MODES,
    YANDEX_VOICES,
    PIPER_VOICES,
    PIPER_DEFAULT_VOICE,
    COSYVOICE_DEFAULT_REST_URL,
    cosyvoice_list_voices,
    list_offline_voices,
    parse_fb2,
    play_file,
    run_cosyvoice,
    run_offline,
    run_online,
    run_piper,
    run_silero,
    run_silero_rest,
    run_yandex,
)
from silero_config import (
    DEFAULT_SILERO_MODEL,
    SILERO_MODELS,
    SPEAKER_LABELS,
    add_model_to_config,
    check_for_model_updates,
    fetch_package_url,
    speaker_choices,
)


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


class TextRedirector(io.TextIOBase):
    """Перенаправляет print() в текстовое поле GUI (и дублирует в файл
    журнала — см. _write_log_file)."""

    def __init__(self, widget: tk.Text, tag: str = "log"):
        self.widget = widget
        self.tag = tag

    def write(self, s: str) -> int:
        if not s:
            return 0
        _write_log_file(s)
        self.widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str):
        self.widget.configure(state="normal")
        self.widget.insert("end", s, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


SETTINGS_PATH = _app_dir() / "fb2_reader_settings.json"

# Какие поля запоминаются между запусками программы (имя атрибута-переменной
# -> ничего больше не нужно, tk.Variable сама знает свой тип через .get()).
SETTINGS_FIELDS = [
    "mode_var", "model_var", "start_var", "outdir_var", "play_var",
    "sample_rate_var", "rest_url_var", "auto_start_rest_var", "rate_var",
    "cosyvoice_rest_url_var", "cosyvoice_engine_var",
    "sentence_break_var", "paragraph_break_var", "comma_break_var",
    "accent_var", "yo_var", "emphasis_var",
    "yandex_api_key_var", "yandex_folder_id_var", "yandex_voice_var", "yandex_speed_var",
    "dialogue_var", "attribution_var", "attribution_provider_var",
    "attribution_api_key_var", "attribution_model_var", "attribution_folder_id_var",
]


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


class AudiobookApp(tk.Tk):
    def __init__(self, initial_book: Path | None = None):
        super().__init__()
        self.title("FB2 Audiobook — озвучивание книг")
        self.minsize(920, 640)
        self.geometry("1020x720")

        self.book_path: Path | None = None
        self.book_title = ""
        self.chapters: list[tuple[str, str]] = []
        self._worker: threading.Thread | None = None
        self._stop_requested = False
        self._offline_voices: list[dict] = []
        self._cosyvoice_voices: list[str] = []
        self._piper_voice_keys: list[str] = []
        self._rest_proc: subprocess.Popen | None = None
        self._cosyvoice_proc: subprocess.Popen | None = None
        self._cosyvoice3_proc: subprocess.Popen | None = None
        self._mixer_ready = False

        # Запоминаем последнюю папку, открытую в каждом отдельном диалоге
        # выбора файла/папки (книга, папка вывода, образец голоса CosyVoice
        # и т.п.) - каждый ключ своей собственный, независимо от остальных,
        # чтобы не приходилось каждый раз заново переходить в нужную папку.
        # Сохраняется в fb2_reader_settings.json вместе с остальными
        # настройками (см. _load_settings/_save_settings).
        self._last_dirs: dict = {}

        # --- состояние встроенного плеера ---
        self._player_path: Path | None = None
        self._player_duration = 0.0     # секунд
        self._player_position = 0.0     # секунд — сохранённая позиция (когда не играет)
        self._player_state = "stopped"  # "stopped" | "paused" | "playing"
        self._player_play_wall_start = 0.0
        self._player_seek_dragging = False
        self._player_tick_job: str | None = None
        self._player_block_event: threading.Event | None = None
        self._player_load_token = 0     # чтобы отбросить устаревший расчёт длительности,
                                         # если пользователь успел выбрать другую главу
        self._player_updating_scale = False  # True, пока сама программа двигает ползунок

        self._build_ui()
        self._bind_events()
        self._load_settings()

        if initial_book and initial_book.exists():
            self.load_book(initial_book)

    def _load_settings(self):
        """Восстанавливает сохранённые настройки (в т.ч. API-ключ Yandex,
        Folder ID и остальные поля) из предыдущего запуска, если файл
        настроек существует. Ошибки чтения тихо игнорируются — при первом
        запуске файла ещё нет, это нормально."""
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for name in SETTINGS_FIELDS:
            if name not in data:
                continue
            var = getattr(self, name, None)
            if var is None:
                continue
            try:
                var.set(data[name])
            except Exception:
                pass
        last_dirs = data.get("_last_dirs")
        if isinstance(last_dirs, dict):
            self._last_dirs = {str(k): str(v) for k, v in last_dirs.items()}
        # mode_var/model_var могли поменяться — обновить зависящие от них
        # виджеты (список голосов, доступность полей и т.п.)
        self._on_model_changed()
        self._on_mode_changed()
        self._on_dialogue_toggle()
        self._on_attribution_toggle()
        self._on_attribution_provider_change(reset_model=False)

    def _save_settings(self):
        """Сохраняет текущие настройки в JSON-файл рядом со скриптом —
        вызывается при закрытии окна и перед стартом озвучки, чтобы
        API-ключ и остальные поля не терялись между запусками программы."""
        data = {}
        for name in SETTINGS_FIELDS:
            var = getattr(self, name, None)
            if var is None:
                continue
            try:
                data[name] = var.get()
            except Exception:
                pass
        data["_last_dirs"] = self._last_dirs
        try:
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            self.log(f"Не удалось сохранить настройки: {e}")

    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Озвучка")

        voices_tab = ttk.Frame(self.notebook)
        self.notebook.add(voices_tab, text="Голоса CosyVoice")

        outer = ttk.Frame(main_tab, padding=10)
        outer.pack(fill="both", expand=True)

        # --- верх: выбор файла ---
        file_row = ttk.Frame(outer)
        file_row.pack(fill="x", pady=(0, 8))

        ttk.Label(file_row, text="Книга FB2:").pack(side="left")
        self.book_var = tk.StringVar()
        add_context_menu(
            ttk.Entry(file_row, textvariable=self.book_var)
        ).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(file_row, text="Обзор…", command=self.browse_book).pack(side="left")
        ttk.Button(file_row, text="Загрузить", command=self.reload_book).pack(side="left", padx=(6, 0))

        paned = ttk.Panedwindow(outer, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- левая колонка: главы ---
        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=3)

        self.title_label = ttk.Label(left, text="Книга не выбрана", font=("Segoe UI", 11, "bold"))
        self.title_label.pack(anchor="w", pady=(0, 4))

        self.chapters_info = ttk.Label(left, text="Глав: —")
        self.chapters_info.pack(anchor="w", pady=(0, 6))

        chapters_frame = ttk.LabelFrame(left, text="Главы", padding=6)
        chapters_frame.pack(fill="both", expand=True)

        self.chapters_list = tk.Listbox(
            chapters_frame,
            activestyle="none",
            exportselection=False,
            font=("Consolas", 10),
            selectmode=tk.EXTENDED,
        )
        scroll = ttk.Scrollbar(chapters_frame, orient="vertical", command=self.chapters_list.yview)
        self.chapters_list.configure(yscrollcommand=scroll.set)
        self.chapters_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.chapters_list.bind("<<ListboxSelect>>", self._on_chapters_selection_change)

        ttk.Label(
            left,
            text="Можно выделить несколько глав (Ctrl/Shift+клик) — тогда озвучатся "
                 "только они, а «Начать с главы» справа будет проигнорировано. Если "
                 "выделена ровно одна глава, ниже можно озвучить только её часть.",
            wraplength=340, foreground="#555", justify="left",
        ).pack(anchor="w", pady=(4, 0))

        self.chapter_range_frame = ttk.LabelFrame(left, text="Кусок главы (необязательно)", padding=6)
        self.chapter_range_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(self.chapter_range_frame, text="С символа:").grid(row=0, column=0, sticky="w")
        self.char_from_var = tk.IntVar(value=0)
        self.char_from_spin = ttk.Spinbox(
            self.chapter_range_frame, from_=0, to=0, textvariable=self.char_from_var, width=8
        )
        self.char_from_spin.grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(self.chapter_range_frame, text="До символа:").grid(row=0, column=2, sticky="w")
        self.char_to_var = tk.IntVar(value=0)
        self.char_to_spin = ttk.Spinbox(
            self.chapter_range_frame, from_=0, to=0, textvariable=self.char_to_var, width=8
        )
        self.char_to_spin.grid(row=0, column=3, sticky="w", padx=(4, 0))
        self.char_range_hint = ttk.Label(
            self.chapter_range_frame, text="Выделите ровно одну главу, чтобы включить.",
            foreground="#888",
        )
        self.char_range_hint.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        for w in (self.char_from_spin, self.char_to_spin):
            w.configure(state="disabled")

        # --- правая колонка: настройки ---
        # Настроек стало много (голоса, диалоги, атрибуция, интонация...) —
        # они не помещаются целиком по высоте на многих экранах, поэтому
        # верхняя часть (все LabelFrame с настройками) идёт в прокручиваемую
        # область, а кнопки/прогресс/журнал внизу закреплены и видны всегда.
        right = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(right, weight=2)

        # --- низ (закреплён): кнопки, прогресс-бар, журнал ---
        btn_row = ttk.Frame(right)
        btn_row.pack(side="bottom", fill="x", pady=(10, 6))

        self.start_btn = ttk.Button(btn_row, text="Начать озвучку", command=self.start_synthesis)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btn_row, text="Остановить", command=self.request_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        # --- встроенный плеер: играет уже озвученную главу (выбранную в
        # списке слева) или очередную главу сразу после её озвучки (если
        # включено "Проигрывать после озвучки каждой главы"). Раньше
        # проигрывание шло через системный проигрыватель/winsound без
        # какой-либо возможности остановить, поставить на паузу или
        # перемотать — теперь звук идёт через pygame.mixer, которым можно
        # управлять прямо отсюда: кнопка play/пауза, стоп и полоса
        # прокрутки (перемотка кликом/перетаскиванием).
        player_frame = ttk.LabelFrame(right, text="Проигрывание", padding=(8, 4))
        player_frame.pack(side="bottom", fill="x", pady=(0, 6))

        self.player_title_label = ttk.Label(player_frame, text="Глава не выбрана", foreground="#555")
        self.player_title_label.pack(anchor="w", fill="x")

        player_seek_row = ttk.Frame(player_frame)
        player_seek_row.pack(fill="x", pady=(4, 0))
        self.player_pos_var = tk.DoubleVar(value=0.0)
        self.player_scale = ttk.Scale(
            player_seek_row, from_=0, to=1, orient="horizontal",
            variable=self.player_pos_var, command=self._on_player_scale_move, state="disabled",
        )
        self.player_scale.pack(side="left", fill="x", expand=True)
        self.player_scale.bind("<ButtonPress-1>", self._on_player_seek_start)
        self.player_scale.bind("<ButtonRelease-1>", self._on_player_seek_commit)
        self.player_time_label = ttk.Label(player_seek_row, text="", width=12, anchor="e")
        self.player_time_label.pack(side="left", padx=(6, 0))

        player_controls_row = ttk.Frame(player_frame)
        player_controls_row.pack(fill="x", pady=(4, 0))
        self.player_play_btn = ttk.Button(
            player_controls_row, text="▶ Играть", command=self._player_play_pause,
            state="disabled", width=12,
        )
        self.player_play_btn.pack(side="left")
        self.player_stop_btn = ttk.Button(
            player_controls_row, text="⏹ Стоп", command=self._player_stop, state="disabled", width=10,
        )
        self.player_stop_btn.pack(side="left", padx=(6, 0))

        self.progress_label = ttk.Label(right, text="", foreground="#555")
        self.progress_label.pack(side="bottom", fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1000, value=0)
        self.progress.pack(side="bottom", fill="x", pady=(2, 2))

        log_frame = ttk.LabelFrame(right, text="Журнал", padding=6)
        log_frame.pack(side="bottom", fill="both", expand=True, pady=(8, 0))

        log_buttons_row = ttk.Frame(log_frame)
        log_buttons_row.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(
            log_buttons_row, text="Копировать всё", command=self._copy_log_to_clipboard
        ).pack(side="left")
        ttk.Button(
            log_buttons_row, text="Открыть файл журнала", command=self._open_log_file
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            log_buttons_row, text=f"Файл: logs\\{LOG_FILE_PATH.name}", foreground="#555"
        ).pack(side="left", padx=(10, 0))

        log_text_row = ttk.Frame(log_frame)
        log_text_row.pack(side="top", fill="both", expand=True)
        self.log_text = tk.Text(log_text_row, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_text_row, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        # --- верх (прокручивается): все настройки ---
        scroll_holder = ttk.Frame(right)
        scroll_holder.pack(side="top", fill="both", expand=True)

        settings_canvas = tk.Canvas(scroll_holder, highlightthickness=0)
        settings_scroll = ttk.Scrollbar(scroll_holder, orient="vertical", command=settings_canvas.yview)
        settings_canvas.configure(yscrollcommand=settings_scroll.set)
        settings_canvas.pack(side="left", fill="both", expand=True)
        settings_scroll.pack(side="right", fill="y")

        right_scroll = ttk.Frame(settings_canvas)
        settings_canvas_window = settings_canvas.create_window((0, 0), window=right_scroll, anchor="nw")

        def _on_right_scroll_configure(_event=None):
            settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

        def _on_settings_canvas_configure(event):
            settings_canvas.itemconfigure(settings_canvas_window, width=event.width)

        right_scroll.bind("<Configure>", _on_right_scroll_configure)
        settings_canvas.bind("<Configure>", _on_settings_canvas_configure)

        def _on_settings_mousewheel(event):
            # Windows/macOS: event.delta; Linux: Button-4/5 обрабатываются отдельно ниже
            settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_settings_wheel(_event=None):
            settings_canvas.bind_all("<MouseWheel>", _on_settings_mousewheel)
            settings_canvas.bind_all("<Button-4>", lambda e: settings_canvas.yview_scroll(-1, "units"))
            settings_canvas.bind_all("<Button-5>", lambda e: settings_canvas.yview_scroll(1, "units"))

        def _unbind_settings_wheel(_event=None):
            settings_canvas.unbind_all("<MouseWheel>")
            settings_canvas.unbind_all("<Button-4>")
            settings_canvas.unbind_all("<Button-5>")

        settings_canvas.bind("<Enter>", _bind_settings_wheel)
        settings_canvas.bind("<Leave>", _unbind_settings_wheel)

        settings = ttk.LabelFrame(right_scroll, text="Настройки озвучки", padding=10)
        settings.pack(fill="x")

        row = 0
        ttk.Label(settings, text="Режим:").grid(row=row, column=0, sticky="w", pady=4)
        self.mode_var = tk.StringVar(value=TTS_MODES["silero"])
        self._mode_labels = list(TTS_MODES.values())
        self._mode_keys = list(TTS_MODES.keys())
        self.mode_combo = ttk.Combobox(
            settings,
            textvariable=self.mode_var,
            values=self._mode_labels,
            state="readonly",
            width=28,
        )
        self.mode_combo.grid(row=row, column=1, sticky="ew", pady=4)
        self.mode_desc = ttk.Label(settings, text=TTS_MODES["silero"], wraplength=280, foreground="#555")
        self.mode_desc.grid(row=row + 1, column=1, sticky="w", pady=(0, 6))

        row += 2
        self.model_label = ttk.Label(settings, text="Модель:")
        self.model_label.grid(row=row, column=0, sticky="w", pady=4)
        self._model_keys = list(SILERO_MODELS.keys())
        self._model_labels = [SILERO_MODELS[k]["title"] for k in self._model_keys]
        default_model_idx = self._model_keys.index(DEFAULT_SILERO_MODEL)
        self.model_var = tk.StringVar(value=self._model_labels[default_model_idx])
        self.model_row = ttk.Frame(settings)
        self.model_row.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.model_combo = ttk.Combobox(
            self.model_row,
            textvariable=self.model_var,
            values=self._model_labels,
            state="readonly",
            width=22,
        )
        self.model_combo.pack(side="left", fill="x", expand=True)
        self.check_updates_btn = ttk.Button(
            self.model_row, text="Обновления…", command=self.check_model_updates
        )
        self.check_updates_btn.pack(side="left", padx=(6, 0))

        row += 1
        ttk.Label(settings, text="Голос:").grid(row=row, column=0, sticky="w", pady=4)
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(settings, textvariable=self.voice_var, state="readonly", width=28)
        self.voice_combo.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(settings, text="Обновить системные", command=self.refresh_offline_voices).grid(
            row=row, column=2, padx=(6, 0), pady=4
        )

        row += 1
        ttk.Label(settings, text="Начать с главы:").grid(row=row, column=0, sticky="w", pady=4)
        self.start_var = tk.IntVar(value=1)
        self.start_spin = ttk.Spinbox(settings, from_=1, to=1, textvariable=self.start_var, width=8)
        self.start_spin.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Label(settings, text="Папка вывода:").grid(row=row, column=0, sticky="w", pady=4)
        out_row = ttk.Frame(settings)
        out_row.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.outdir_var = tk.StringVar(value=str(Path("audiobook_output").resolve()))
        add_context_menu(
            ttk.Entry(out_row, textvariable=self.outdir_var)
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="…", width=3, command=self.browse_outdir).pack(side="left", padx=(4, 0))

        row += 1
        self.play_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text="Проигрывать после озвучки каждой главы", variable=self.play_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=4
        )

        row += 1
        self.sample_rate_label = ttk.Label(settings, text="Частота (Гц):")
        self.sample_rate_label.grid(row=row, column=0, sticky="w", pady=4)
        self.sample_rate_var = tk.StringVar(value="48000")
        self.sample_rate_combo = ttk.Combobox(
            settings,
            textvariable=self.sample_rate_var,
            values=["8000", "24000", "48000"],
            state="readonly",
            width=10,
        )
        self.sample_rate_combo.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        self.rest_url_label = ttk.Label(settings, text="REST URL:")
        self.rest_url_label.grid(row=row, column=0, sticky="w", pady=4)
        self.rest_url_var = tk.StringVar(value="http://localhost:5010")
        self.rest_url_entry = add_context_menu(ttk.Entry(settings, textvariable=self.rest_url_var))
        self.rest_url_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)

        row += 1
        self.auto_start_rest_var = tk.BooleanVar(value=True)
        self.auto_start_rest_check = ttk.Checkbutton(
            settings,
            text="Запускать сервис Silero REST автоматически (без консоли)",
            variable=self.auto_start_rest_var,
        )
        self.auto_start_rest_check.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)

        self.cosyvoice_rest_url_var = tk.StringVar(value=COSYVOICE_DEFAULT_REST_URL)
        self.cosyvoice_engine_var = tk.StringVar(value=COSYVOICE_ENGINE_LABELS[COSYVOICE_DEFAULT_ENGINE])
        row += 1
        self.cosyvoice_tab_hint = ttk.Label(
            settings,
            text="Список голосов и добавление нового голоса из аудиофайла — "
                 "на вкладке «Голоса CosyVoice» вверху окна.",
            foreground="#555", wraplength=280, justify="left",
        )
        self.cosyvoice_tab_hint.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))

        row += 1
        self.rate_label = ttk.Label(settings, text="Скорость (offline):")
        self.rate_label.grid(row=row, column=0, sticky="w", pady=4)
        self.rate_var = tk.IntVar(value=170)
        self.rate_spin = ttk.Spinbox(settings, from_=80, to=300, textvariable=self.rate_var, width=8)
        self.rate_spin.grid(row=row, column=1, sticky="w", pady=4)

        settings.columnconfigure(1, weight=1)

        # --- Разные голоса для диалогов (silero / silero_rest / offline) ---
        dlg_frame = ttk.LabelFrame(
            right_scroll, text="Разные голоса для диалогов (реплики персонажей)", padding=10
        )
        dlg_frame.pack(fill="x", pady=(8, 0))
        self._dialogue_frame = dlg_frame

        self.dialogue_var = tk.BooleanVar(value=False)
        self.dialogue_check = ttk.Checkbutton(
            dlg_frame, text="Включить", variable=self.dialogue_var,
            command=self._on_dialogue_toggle,
        )
        self.dialogue_check.pack(anchor="w")

        self.dialogue_hint = ttk.Label(
            dlg_frame,
            text="Реплики (абзацы, начинающиеся с тире) звучат по очереди голосами "
                 "из списка ниже, весь остальной текст — основным голосом сверху "
                 "(«Голос:»). Это чередование, а не привязка голоса к конкретному "
                 "персонажу — для этого нужен полноценный анализ текста.",
            foreground="#555", justify="left", wraplength=340,
        )
        self.dialogue_hint.pack(anchor="w", pady=(2, 4))

        dlg_list_row = ttk.Frame(dlg_frame)
        dlg_list_row.pack(fill="both", expand=True)
        self.dialogue_list = tk.Listbox(
            dlg_list_row, selectmode=tk.EXTENDED, exportselection=False, height=6, state="disabled",
        )
        dlg_list_scroll = ttk.Scrollbar(dlg_list_row, orient="vertical", command=self.dialogue_list.yview)
        self.dialogue_list.configure(yscrollcommand=dlg_list_scroll.set)
        self.dialogue_list.pack(side="left", fill="both", expand=True)
        dlg_list_scroll.pack(side="right", fill="y")

        self.dialogue_unavailable_label = ttk.Label(
            dlg_frame,
            text="Google TTS (режим online) не поддерживает несколько разных "
                 "русских голосов — используйте Silero, Silero REST или Yandex SpeechKit "
                 "(для Yandex — свой блок «Yandex SpeechKit» ниже).",
            foreground="#a00", justify="left", wraplength=340,
        )
        # прячется/показывается в _update_mode_dependent_widgets

        self._attribution_section = ttk.Frame(dlg_frame)
        self._attribution_section.pack(fill="x")
        attr_section = self._attribution_section

        ttk.Separator(attr_section, orient="horizontal").pack(fill="x", pady=(8, 6))

        self.attribution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            attr_section,
            text="Определять, КАКОЙ персонаж говорит (через LLM)",
            variable=self.attribution_var, command=self._on_attribution_toggle,
        ).pack(anchor="w")

        self.attribution_hint = ttk.Label(
            attr_section,
            text="Без этого голоса из списка выше просто чередуются по кругу — диалог двух "
                 "персонажей прозвучит разными голосами, но один и тот же персонаж в разных "
                 "сценах/главах может получить разный голос. С этим — LLM читает главу и "
                 "определяет, кто говорит каждую реплику, и голос закрепляется за персонажем "
                 "на всю книгу (словарь сохраняется в dialogue_characters.json рядом с аудио "
                 "— можно поправить руками).",
            foreground="#555", justify="left", wraplength=340,
        )
        self.attribution_hint.pack(anchor="w", pady=(2, 4))

        attr_provider_row = ttk.Frame(attr_section)
        attr_provider_row.pack(fill="x", pady=(2, 2))
        ttk.Label(attr_provider_row, text="Сервис:").pack(side="left")
        self._attribution_provider_titles = [p["title"] for p in ATTRIBUTION_PROVIDERS.values()]
        self._attribution_provider_by_title = {
            p["title"]: key for key, p in ATTRIBUTION_PROVIDERS.items()
        }
        self._attribution_title_by_provider = {
            key: p["title"] for key, p in ATTRIBUTION_PROVIDERS.items()
        }
        self.attribution_provider_var = tk.StringVar(
            value=self._attribution_title_by_provider[DEFAULT_ATTRIBUTION_PROVIDER]
        )
        self.attribution_provider_combo = ttk.Combobox(
            attr_provider_row, textvariable=self.attribution_provider_var,
            values=self._attribution_provider_titles, state="readonly", width=42,
        )
        self.attribution_provider_combo.pack(side="left", padx=(6, 0))
        self.attribution_provider_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._on_attribution_provider_change()
        )

        attr_key_row = ttk.Frame(attr_section)
        attr_key_row.pack(fill="x", pady=(2, 2))
        self.attribution_key_label = ttk.Label(attr_key_row, text="API-ключ:")
        self.attribution_key_label.pack(side="left")
        self.attribution_api_key_var = tk.StringVar(
            value=os.environ.get("GEMINI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.attribution_api_key_entry = add_context_menu(
            ttk.Entry(attr_key_row, textvariable=self.attribution_api_key_var, show="•")
        )
        self.attribution_api_key_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self._attribution_key_show_var = tk.BooleanVar(value=False)

        def _toggle_attribution_key_visibility():
            self.attribution_api_key_entry.configure(
                show="" if self._attribution_key_show_var.get() else "•"
            )

        ttk.Checkbutton(
            attr_key_row, text="показать", variable=self._attribution_key_show_var,
            command=_toggle_attribution_key_visibility,
        ).pack(side="left")

        attr_folder_row = ttk.Frame(attr_section)
        attr_folder_row.pack(fill="x", pady=(2, 2))
        self.attribution_folder_label = ttk.Label(attr_folder_row, text="Folder ID:")
        self.attribution_folder_label.pack(side="left")
        self.attribution_folder_id_var = tk.StringVar(value="")
        self.attribution_folder_id_entry = add_context_menu(
            ttk.Entry(attr_folder_row, textvariable=self.attribution_folder_id_var)
        )
        self.attribution_folder_id_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))

        def _use_yandex_speechkit_creds():
            self.attribution_api_key_var.set(self.yandex_api_key_var.get())
            self.attribution_folder_id_var.set(self.yandex_folder_id_var.get())

        self.attribution_use_yandex_btn = ttk.Button(
            attr_section, text="Взять ключ/Folder ID из настроек Yandex SpeechKit ниже",
            command=_use_yandex_speechkit_creds,
        )
        self.attribution_use_yandex_btn.pack(anchor="w", pady=(2, 2))

        attr_model_row = ttk.Frame(attr_section)
        attr_model_row.pack(fill="x", pady=(2, 0))
        ttk.Label(attr_model_row, text="Модель:").pack(side="left")
        self.attribution_model_var = tk.StringVar(value=DEFAULT_ATTRIBUTION_MODEL)
        self.attribution_model_entry = add_context_menu(
            ttk.Entry(attr_model_row, textvariable=self.attribution_model_var, width=24)
        )
        self.attribution_model_entry.pack(side="left", padx=(6, 0))
        self.attribution_key_hint_label = ttk.Label(
            attr_section, text=ATTRIBUTION_PROVIDERS[DEFAULT_ATTRIBUTION_PROVIDER]["key_hint"],
            foreground="#888", justify="left", wraplength=340,
        )
        self.attribution_key_hint_label.pack(anchor="w", pady=(4, 0))

        self._on_attribution_toggle()

        # --- Yandex SpeechKit: ключ, каталог, голос ---
        yandex_frame = ttk.LabelFrame(right_scroll, text="Yandex SpeechKit (режим yandex)", padding=10)
        yandex_frame.pack(fill="x", pady=(8, 0))

        yrow = 0
        ttk.Label(yandex_frame, text="API-ключ:").grid(row=yrow, column=0, sticky="w", pady=3)
        key_row = ttk.Frame(yandex_frame)
        key_row.grid(row=yrow, column=1, columnspan=2, sticky="ew", pady=3)
        self.yandex_api_key_var = tk.StringVar(value=os.environ.get("YANDEX_API_KEY", ""))
        self.yandex_api_key_entry = add_context_menu(
            ttk.Entry(key_row, textvariable=self.yandex_api_key_var, show="•")
        )
        self.yandex_api_key_entry.pack(side="left", fill="x", expand=True)
        self._yandex_key_show_var = tk.BooleanVar(value=False)

        def _toggle_key_visibility():
            self.yandex_api_key_entry.configure(show="" if self._yandex_key_show_var.get() else "•")

        ttk.Checkbutton(
            key_row, text="показать", variable=self._yandex_key_show_var, command=_toggle_key_visibility
        ).pack(side="left", padx=(6, 0))

        yrow += 1
        ttk.Label(yandex_frame, text="Folder ID (необязательно):").grid(row=yrow, column=0, sticky="w", pady=3)
        self.yandex_folder_id_var = tk.StringVar(value=os.environ.get("YANDEX_FOLDER_ID", ""))
        self.yandex_folder_id_entry = add_context_menu(
            ttk.Entry(yandex_frame, textvariable=self.yandex_folder_id_var)
        )
        self.yandex_folder_id_entry.grid(row=yrow, column=1, columnspan=2, sticky="ew", pady=3)

        yrow += 1
        ttk.Label(yandex_frame, text="Голос:").grid(row=yrow, column=0, sticky="w", pady=3)
        self._yandex_voice_keys = list(YANDEX_VOICES.keys())
        yandex_voice_labels = [f"{k} — {v}" for k, v in YANDEX_VOICES.items()]
        self.yandex_voice_var = tk.StringVar(value=yandex_voice_labels[0])
        self.yandex_voice_combo = ttk.Combobox(
            yandex_frame, textvariable=self.yandex_voice_var, values=yandex_voice_labels,
            state="readonly", width=28,
        )
        self.yandex_voice_combo.grid(row=yrow, column=1, columnspan=2, sticky="ew", pady=3)

        yrow += 1
        ttk.Label(yandex_frame, text="Скорость речи:").grid(row=yrow, column=0, sticky="w", pady=3)
        self.yandex_speed_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            yandex_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.yandex_speed_var, width=8
        ).grid(row=yrow, column=1, sticky="w", pady=3)

        yrow += 1
        ttk.Label(
            yandex_frame,
            text="Платный облачный сервис (после пробного периода) — нужен интернет\n"
                 "на каждую главу. Как получить ключ и Folder ID — см. README.\n"
                 "Разные голоса для диалогов — в блоке «Разные голоса для диалогов» выше.",
            foreground="#555", justify="left",
        ).grid(row=yrow, column=0, columnspan=3, sticky="w", pady=(8, 0))

        yandex_frame.columnconfigure(1, weight=1)
        self._yandex_frame = yandex_frame

        # --- интонация: паузы по знакам препинания, ударения, усиление ---
        intonation = self._intonation_frame = ttk.LabelFrame(
            right_scroll, text="Интонация и паузы (silero / silero_rest / cosyvoice)", padding=10
        )
        intonation.pack(fill="x", pady=(8, 0))

        irow = 0
        ttk.Label(intonation, text="Пауза на запятых/тире, мс:").grid(row=irow, column=0, sticky="w", pady=3)
        self.comma_break_var = tk.IntVar(value=180)
        ttk.Spinbox(intonation, from_=0, to=1000, increment=10, textvariable=self.comma_break_var, width=8).grid(
            row=irow, column=1, sticky="w", pady=3
        )

        irow += 1
        ttk.Label(intonation, text="Пауза между предложениями, мс:").grid(row=irow, column=0, sticky="w", pady=3)
        self.sentence_break_var = tk.IntVar(value=320)
        ttk.Spinbox(intonation, from_=0, to=2000, increment=10, textvariable=self.sentence_break_var, width=8).grid(
            row=irow, column=1, sticky="w", pady=3
        )

        irow += 1
        ttk.Label(intonation, text="Пауза между абзацами, мс:").grid(row=irow, column=0, sticky="w", pady=3)
        self.paragraph_break_var = tk.IntVar(value=550)
        ttk.Spinbox(intonation, from_=0, to=3000, increment=10, textvariable=self.paragraph_break_var, width=8).grid(
            row=irow, column=1, sticky="w", pady=3
        )

        irow += 1
        self.accent_var = tk.BooleanVar(value=True)
        self.accent_check = ttk.Checkbutton(
            intonation, text="Расставлять ударения автоматически (только silero/silero_rest)",
            variable=self.accent_var,
        )
        self.accent_check.grid(row=irow, column=0, columnspan=2, sticky="w", pady=3)

        irow += 1
        self.yo_var = tk.BooleanVar(value=True)
        self.yo_check = ttk.Checkbutton(
            intonation, text='Заменять "е" на "ё" где нужно (только silero/silero_rest)',
            variable=self.yo_var,
        )
        self.yo_check.grid(row=irow, column=0, columnspan=2, sticky="w", pady=3)

        irow += 1
        self.emphasis_var = tk.BooleanVar(value=True)
        self.emphasis_check = ttk.Checkbutton(
            intonation,
            text="Усиливать интонацию «?!» (только silero_rest)",
            variable=self.emphasis_var,
        )
        self.emphasis_check.grid(row=irow, column=0, columnspan=2, sticky="w", pady=3)

        intonation.columnconfigure(1, weight=1)

        self._on_attribution_provider_change(reset_model=False)
        self._populate_silero_voices()
        self._update_mode_dependent_widgets()

        self._build_voices_tab(voices_tab)

    def _build_voices_tab(self, tab: ttk.Frame):
        """Отдельная вкладка «Голоса CosyVoice»: список уже сохранённых
        профилей голоса (они хранятся на самом сервисе CosyVoice REST, в
        cosyvoice_voices.json — сохраняются постоянно и переживают
        перезапуск программы и сервиса) и форма для добавления нового
        голоса из аудиофайла — без всплывающего окна, прямо на вкладке."""
        pad = ttk.Frame(tab, padding=10)
        pad.pack(fill="both", expand=True)

        ttk.Label(
            pad,
            text="Голоса, склонированные из аудиофайлов для режима CosyVoice. "
                 "Каждый добавленный голос сохраняется на сервисе CosyVoice "
                 "(в cosyvoice_voices.json рядом с моделью) и остаётся доступен "
                 "после перезапуска программы и компьютера — его не нужно "
                 "добавлять заново.",
            wraplength=680, justify="left", foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        url_row = ttk.Frame(pad)
        url_row.pack(fill="x", pady=(0, 8))
        ttk.Label(url_row, text="Адрес сервиса CosyVoice:").pack(side="left")
        self.cosyvoice_rest_url_entry = add_context_menu(
            ttk.Entry(url_row, textvariable=self.cosyvoice_rest_url_var, width=32)
        )
        self.cosyvoice_rest_url_entry.pack(side="left", padx=(6, 0))
        self.cosyvoice_refresh_btn = ttk.Button(
            url_row, text="Обновить список", command=self.refresh_cosyvoice_voices
        )
        self.cosyvoice_refresh_btn.pack(side="left", padx=(6, 0))

        engine_row = ttk.Frame(pad)
        engine_row.pack(fill="x", pady=(0, 8))
        ttk.Label(engine_row, text="Движок синтеза:").pack(side="left")
        self.cosyvoice_engine_combo = ttk.Combobox(
            engine_row, textvariable=self.cosyvoice_engine_var, state="readonly",
            width=34, values=list(COSYVOICE_ENGINE_LABELS.values()),
        )
        self.cosyvoice_engine_combo.pack(side="left", padx=(6, 0))
        self.cosyvoice_engine_combo.bind("<<ComboboxSelected>>", self._on_cosyvoice_engine_changed)
        ttk.Label(
            pad,
            text="ESpeech RL-V2 — рекомендуется: тоже F5-TTS-архитектура, но менее шумный "
                 "результат, чем у XTTS-v2 (по независимым сравнениям), и поддерживает "
                 "клонирование (новее версии SFT-95K, которая использовалась раньше). "
                 "Расстановку ударений не использует — несмотря на документацию, "
                 "чекпоинт на практике произносит символ \"+\" как слово \"плюс\" вместо того, "
                 "чтобы ставить ударение. F5-TTS-Russian winter — более свежий "
                 "чекпоинт F5-TTS для русского с полной разметкой ударений (в отличие от "
                 "обычного F5-TTS-Russian ниже). F5-TTS-Russian — тоже неплохо, но без "
                 "ударений и обучена на меньшем датасете. XTTS-v2 — запасной вариант "
                 "(мультиязычная, заметно более шумное аудио). CosyVoice 3 — "
                 "экспериментальный, официально поддерживает русский, но независимых "
                 "отзывов о качестве на русском пока нет; ставится отдельной секцией "
                 "install.bat в своё окружение .venv_cosyvoice3 (несовместимо со всем "
                 "остальным по версиям зависимостей). Голоса, которые вы уже добавили, "
                 "работают со всеми движками без повторного добавления — переклонировать "
                 "не нужно. Смена движка перезапускает сервис CosyVoice (может занять "
                 "время на загрузку модели — у ESpeech и F5-TTS-Russian winter чекпоинты "
                 "больше, ~1.3–2.7 ГБ при первой загрузке; у CosyVoice3 ещё и веса "
                 "модели, ~1-2 ГБ, при первом запуске).",
            wraplength=680, justify="left", foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        body = ttk.Panedwindow(pad, orient="horizontal")
        body.pack(fill="both", expand=True)

        # --- слева: список уже сохранённых голосов ---
        list_frame = ttk.LabelFrame(body, text="Сохранённые голоса", padding=8)
        body.add(list_frame, weight=1)

        list_row = ttk.Frame(list_frame)
        list_row.pack(fill="both", expand=True)
        self.cosyvoice_voices_listbox = tk.Listbox(list_row, exportselection=False, height=14)
        cv_list_scroll = ttk.Scrollbar(list_row, orient="vertical", command=self.cosyvoice_voices_listbox.yview)
        self.cosyvoice_voices_listbox.configure(yscrollcommand=cv_list_scroll.set)
        self.cosyvoice_voices_listbox.pack(side="left", fill="both", expand=True)
        cv_list_scroll.pack(side="right", fill="y")

        ttk.Label(
            list_frame,
            text="Голос, выбранный здесь, используется как основной, если на "
                 "вкладке «Озвучка» стоит режим CosyVoice.",
            wraplength=280, foreground="#555", justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self._cv_delete_btn = ttk.Button(
            list_frame, text="Удалить выбранный голос",
            command=self._delete_selected_cosyvoice_voice,
        )
        self._cv_delete_btn.pack(anchor="w", pady=(6, 0))

        # --- справа: добавление нового голоса из аудио ---
        add_frame = ttk.LabelFrame(body, text="Добавить голос из аудиофайла", padding=8)
        body.add(add_frame, weight=1)

        ttk.Label(
            add_frame,
            text="Короткая (3-10 сек) чистая запись голоса без музыки и шума — "
                 "например, фрагмент записи любимого диктора.",
            wraplength=300, foreground="#555", justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(add_frame, text="Аудиофайл:").grid(row=1, column=0, sticky="w", pady=4)
        self._cv_add_audio_path_var = tk.StringVar()
        add_context_menu(
            ttk.Entry(add_frame, textvariable=self._cv_add_audio_path_var, width=26)
        ).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(add_frame, text="…", width=3, command=self._browse_cosyvoice_audio).grid(
            row=1, column=2, padx=(4, 0), pady=4
        )

        ttk.Label(add_frame, text="Имя голоса:").grid(row=2, column=0, sticky="w", pady=4)
        self._cv_add_name_var = tk.StringVar()
        add_context_menu(
            ttk.Entry(add_frame, textvariable=self._cv_add_name_var, width=26)
        ).grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(add_frame, text="Текст записи:").grid(row=3, column=0, sticky="nw", pady=4)
        self._cv_add_text_widget = tk.Text(add_frame, width=26, height=4, wrap="word")
        self._cv_add_text_widget.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)

        self._cv_transcribe_btn = ttk.Button(
            add_frame, text="Распознать текст (по аудио)", command=self._transcribe_cosyvoice_audio,
        )
        self._cv_transcribe_btn.grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(
            add_frame,
            text="Движок клонирования голоса — F5-TTS-Russian (обучена именно на русской "
                 "речи). Текст записи желателен для лучшего клонирования — если оставить "
                 "пустым, сервис сам распознает его при первом использовании голоса. Для "
                 "лучшего результата берите 6-15 секунд чистой речи одного голоса без "
                 "музыки и шума.",
            wraplength=300, foreground="#555", justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._cv_add_status_label = ttk.Label(add_frame, text="", foreground="#555", wraplength=300, justify="left")
        self._cv_add_status_label.grid(row=6, column=0, columnspan=3, sticky="w")

        self._cv_add_btn = ttk.Button(add_frame, text="Добавить голос", command=self._add_cosyvoice_voice_from_tab)
        self._cv_add_btn.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))

        add_frame.columnconfigure(1, weight=1)

        self.cosyvoice_voices_listbox.bind("<<ListboxSelect>>", self._on_cosyvoice_voices_listbox_select)

        self._refresh_cosyvoice_voices_listbox()

    def _bind_events(self):
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def log(self, message: str):
        _write_log_file(message + "\n")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _copy_log_to_clipboard(self):
        """Копирует весь текст журнала в буфер обмена одной кнопкой — на
        случай, если выделение мышью в самом окне журнала работает
        неудобно (так бывает в некоторых версиях tkinter, когда текст в
        поле переведён в состояние "только для чтения")."""
        text = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("(Весь журнал скопирован в буфер обмена.)")

    def _open_log_file(self):
        """Открывает файл журнала (fb2_reader_gui.log рядом с программой)
        в обычной программе для просмотра текста, откуда его можно
        свободно выделять и копировать — в отличие от окна журнала внутри
        программы."""
        if _log_file_handle:
            try:
                _log_file_handle.flush()
            except Exception:
                pass
        if not LOG_FILE_PATH.exists():
            messagebox.showinfo("Файл журнала", f"Файл ещё не создан: {LOG_FILE_PATH}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(LOG_FILE_PATH))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(LOG_FILE_PATH)])
            else:
                subprocess.Popen(["xdg-open", str(LOG_FILE_PATH)])
        except Exception as e:
            messagebox.showerror("Файл журнала", f"Не удалось открыть файл:\n{LOG_FILE_PATH}\n\n{e}")

    def _get_last_dir(self, key: str) -> str:
        """Последняя папка, открытая в диалоге с этим ключом (свой для
        каждого отдельного диалога — книга, папка вывода, образец голоса и
        т.п.), или пустая строка, если такого диалога ещё не открывали."""
        return self._last_dirs.get(key, "")

    def _remember_last_dir(self, key: str, path: str):
        d = os.path.dirname(path) if os.path.isfile(path) or not os.path.isdir(path) else path
        if d:
            self._last_dirs[key] = d

    def browse_book(self):
        path = filedialog.askopenfilename(
            title="Выберите FB2-книгу",
            initialdir=self._get_last_dir("book") or None,
            filetypes=[
                ("FB2 книги", "*.fb2 *.fb2.zip *.zip"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._remember_last_dir("book", path)
            self.book_var.set(path)
            self.reload_book()

    def browse_outdir(self):
        path = filedialog.askdirectory(
            title="Папка для сохранения аудио",
            initialdir=self._get_last_dir("outdir") or None,
        )
        if path:
            self._remember_last_dir("outdir", path)
            self.outdir_var.set(path)

    def reload_book(self):
        path_str = self.book_var.get().strip()
        if not path_str:
            messagebox.showwarning("Книга", "Укажите путь к файлу FB2.")
            return
        path = Path(path_str)
        if not path.exists():
            messagebox.showerror("Книга", f"Файл не найден:\n{path}")
            return
        self.load_book(path)

    def load_book(self, path: Path):
        try:
            title, chapters = parse_fb2(path)
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e))
            return

        self.book_path = path
        self.book_title = title
        self.chapters = chapters
        self.book_var.set(str(path))
        self.title_label.configure(text=title)
        self.chapters_info.configure(text=f"Глав: {len(chapters)}")
        self.chapters_list.delete(0, "end")
        for i, (ch_title, text) in enumerate(chapters, 1):
            self.chapters_list.insert("end", f"{i:3d}. {ch_title}  ({len(text)} симв.)")
        self.start_spin.configure(to=max(1, len(chapters)))
        self._on_chapters_selection_change()
        self.log(f"Загружена книга «{title}» — {len(chapters)} глав.")

    def _on_chapters_selection_change(self, event=None):
        """Включает/выключает поля "кусок главы" в зависимости от того,
        выделена ли ровно одна глава в списке, и подгоняет их границы под
        длину выбранной главы."""
        sel = self.chapters_list.curselection()
        widgets = (self.char_from_spin, self.char_to_spin)
        if len(sel) == 1 and self.chapters:
            idx = sel[0] + 1
            _, text = self.chapters[idx - 1]
            n = len(text)
            for w in widgets:
                w.configure(state="normal", to=n)
            if self.char_to_var.get() in (0,) or self.char_to_var.get() > n:
                self.char_to_var.set(n)
            self.char_from_spin.configure(to=n)
            self.char_range_hint.configure(
                text=f"Глава «{self.chapters[idx - 1][0]}»: символов {n}. "
                     f"Оставьте 0 и {n}, чтобы озвучить целиком."
            )
        else:
            for w in widgets:
                w.configure(state="disabled")
            self.char_range_hint.configure(
                text="Выделите ровно одну главу, чтобы включить."
                if len(sel) != 1 else self.char_range_hint.cget("text")
            )
        self.start_var.set(1)
        self.refresh_player_for_selection()

    def _populate_silero_voices(self, model_id: str | None = None):
        if model_id is None:
            model_id = self._selected_model_id()
        choices = speaker_choices(model_id)
        self._silero_voice_labels = [f"{k} — {v}" for k, v in choices.items()]
        self._silero_voice_keys = list(choices.keys())

    def _selected_model_id(self) -> str:
        label = self.model_var.get()
        try:
            idx = self._model_labels.index(label)
            return self._model_keys[idx]
        except ValueError:
            return DEFAULT_SILERO_MODEL

    def _on_model_changed(self):
        self._populate_silero_voices()
        if self._current_mode() in ("silero", "silero_rest"):
            self._set_silero_voices()
            self._populate_dialogue_list()

    def _current_mode(self) -> str:
        label = self.mode_var.get()
        try:
            idx = self._mode_labels.index(label)
            return self._mode_keys[idx]
        except ValueError:
            return "silero"

    def refresh_offline_voices(self):
        try:
            self._offline_voices = list_offline_voices()
        except Exception as e:
            messagebox.showerror("Голоса", f"Не удалось получить системные голоса:\n{e}")
            return
        if self._current_mode() == "offline":
            self._set_offline_voices()
            self._populate_dialogue_list()
        self.log(f"Найдено системных голосов: {len(self._offline_voices)}")

    def _set_silero_voices(self):
        self.voice_combo.configure(values=self._silero_voice_labels, state="readonly")
        default_idx = self._silero_voice_keys.index("xenia") if "xenia" in self._silero_voice_keys else 0
        self.voice_combo.current(default_idx)

    def check_model_updates(self):
        """Проверяет на GitHub, не появилась ли более новая модель Silero,
        чем те, что уже есть в списке. Ничего не скачивает — только
        сообщает в журнал; сама проверка идёт в фоновом потоке, чтобы не
        подвешивать окно на время сетевого запроса."""
        self.check_updates_btn.configure(state="disabled")
        self.log("Проверяю обновления моделей Silero…")

        def worker():
            result = check_for_model_updates()
            self.after(0, lambda: self._on_model_updates_checked(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_updates_checked(self, result: dict):
        if not result.get("ok"):
            self.check_updates_btn.configure(state="normal")
            self.log(f"Не удалось проверить обновления: {result.get('error')}")
            return
        new_models = result.get("new_models") or []
        if not new_models:
            self.check_updates_btn.configure(state="normal")
            self.log(f"Новых моделей нет — используется актуальная ({', '.join(result.get('checked', []))}).")
            return

        self.log("Найдены новые модели Silero, которых ещё нет в этой программе: " + ", ".join(new_models))
        add_now = messagebox.askyesno(
            "Обновления моделей",
            "Найдены новые версии модели: " + ", ".join(new_models) +
            "\n\nДобавить их автоматически в программу прямо сейчас?\n\n"
            "Ссылка на файл модели будет взята из GitHub, а список голосов "
            f"по умолчанию скопирован от «{DEFAULT_SILERO_MODEL}» — если у "
            "новой модели голоса другие, это будет видно сразу при первой "
            "попытке озвучить ими (лишний голос просто не сработает, "
            "программа не сломается). Изменение сохранится в silero_config.py "
            "и останется после перезапуска.",
        )
        if not add_now:
            self.check_updates_btn.configure(state="normal")
            return

        self.log("Добавляю новые модели…")

        def worker():
            added, failed = [], []
            for model_id in new_models:
                url = fetch_package_url(model_id)
                if not url:
                    failed.append(model_id)
                    continue
                try:
                    add_model_to_config(model_id, url)
                    added.append(model_id)
                except OSError as e:
                    failed.append(f"{model_id} ({e})")
            self.after(0, lambda: self._on_models_added(added, failed))

        threading.Thread(target=worker, daemon=True).start()

    def _on_models_added(self, added: list, failed: list):
        self.check_updates_btn.configure(state="normal")
        if added:
            # SILERO_MODELS уже обновлён add_model_to_config — перечитываем
            # списки моделей в интерфейсе, чтобы новые модели сразу
            # появились в выпадающем списке без перезапуска программы
            self._model_keys = list(SILERO_MODELS.keys())
            self._model_labels = [SILERO_MODELS[k]["title"] for k in self._model_keys]
            self.model_combo.configure(values=self._model_labels)
            self.log(
                f"Добавлены модели: {', '.join(added)}. Теперь доступны в списке «Модель» "
                "(файл silero_config.py обновлён, менять руками не нужно). "
                "Проверьте список голосов для новой модели при первой озвучке."
            )
            messagebox.showinfo(
                "Модели добавлены", "Добавлены модели: " + ", ".join(added) +
                "\n\nОни уже доступны в списке «Модель»."
            )
        if failed:
            self.log(f"Не удалось добавить: {', '.join(failed)}")
            messagebox.showerror("Модели", "Не удалось добавить: " + ", ".join(failed))

    def _set_offline_voices(self):
        if not self._offline_voices:
            self.refresh_offline_voices()
        labels = []
        for v in self._offline_voices:
            mark = "★ " if v["is_russian"] else ""
            labels.append(f"{mark}{v['name']}")
        if not labels:
            labels = ["(голоса не найдены — будет использован по умолчанию)"]
        self.voice_combo.configure(values=labels, state="readonly")
        self.voice_combo.current(0)

    def _set_online_voice(self):
        self.voice_combo.configure(values=["Google TTS — русский (выбор голоса недоступен)"], state="disabled")
        self.voice_combo.current(0)

    def _set_piper_voices(self):
        self._piper_voice_keys = list(PIPER_VOICES.keys())
        labels = [f"{k} — {v}" for k, v in PIPER_VOICES.items()]
        self.voice_combo.configure(values=labels, state="readonly")
        default_idx = self._piper_voice_keys.index(PIPER_DEFAULT_VOICE) if PIPER_DEFAULT_VOICE in self._piper_voice_keys else 0
        self.voice_combo.current(default_idx)

    def _selected_piper_voice(self) -> str:
        idx = self.voice_combo.current()
        if idx < 0 or idx >= len(self._piper_voice_keys):
            return PIPER_DEFAULT_VOICE
        return self._piper_voice_keys[idx]

    def _set_cosyvoice_voices(self):
        if not self._cosyvoice_voices:
            labels = ["(нет голосов — нажмите «Обновить голоса»)"]
            self.voice_combo.configure(values=labels, state="disabled")
            self.voice_combo.current(0)
            return
        self.voice_combo.configure(values=list(self._cosyvoice_voices), state="readonly")
        default_idx = self._cosyvoice_voices.index("default") if "default" in self._cosyvoice_voices else 0
        self.voice_combo.current(default_idx)

    def refresh_cosyvoice_voices(self):
        """Запрашивает у сервиса CosyVoice REST текущий список профилей
        голоса (GET /voices) в фоновом потоке, чтобы не подвешивать окно
        на время сетевого запроса (сервис может ещё загружать модель)."""
        rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
        self.cosyvoice_refresh_btn.configure(state="disabled")
        self.log(f"Запрашиваю список голосов CosyVoice у {rest_url} …")

        def worker():
            if self._rest_url_is_local(rest_url) and not self._ping_rest_service(rest_url):
                self.after(0, lambda: self.log(
                    "Сервис CosyVoice ещё не запущен — запускаю его автоматически…"
                ))
                self._ensure_cosyvoice_service_running(rest_url)
            try:
                voices = cosyvoice_list_voices(rest_url)
            except Exception:
                voices = []
            self.after(0, lambda: self._on_cosyvoice_voices_fetched(voices))

        threading.Thread(target=worker, daemon=True).start()

    def _on_cosyvoice_voices_fetched(self, voices: list):
        self.cosyvoice_refresh_btn.configure(state="normal")
        self._cosyvoice_voices = voices
        if not voices:
            self.log(
                "Не удалось получить голоса CosyVoice — сервис не отвечает или ещё "
                "загружает модель (первый запуск после старта сервиса может занять "
                "некоторое время). Убедитесь, что сервис запущен, и попробуйте ещё раз. "
                "Если это повторяется — смотрите logs\\fb2_reader_gui.log и "
                "CosyVoice\\cosyvoice_rest_service.log за подробностями."
            )
        else:
            self.log(f"Голоса CosyVoice: {', '.join(voices)}")
        if self._current_mode() == "cosyvoice":
            self._set_cosyvoice_voices()
            self._populate_dialogue_list()
        self._refresh_cosyvoice_voices_listbox()

    def _selected_cosyvoice_voice(self) -> str:
        """ВАЖНО: раньше при idx == -1 (ничего не выбрано в выпадающем
        списке - например, из-за гонки между фоновым обновлением списка
        голосов в refresh_cosyvoice_voices() и стартом синтеза) сюда
        возвращалась буквальная строка "default" - а такого профиля голоса
        у CosyVoice3 никогда не бывает (профили называются как файлы
        образцов, например "002 Пролог"), поэтому /getwav сразу отвечал
        HTTP 400 "Профиль голоса 'default' не найден" на КАЖДОМ фрагменте
        книги подряд - озвучка проходила вхолостую, вставляя тишину везде.
        Теперь при отсутствии валидного выбора в комбобоксе подстраховываемся
        первым РЕАЛЬНЫМ профилем из self._cosyvoice_voices (если список вообще
        не пуст) - это почти наверняка тот голос, который был бы выбран по
        умолчанию при обычном заполнении списка (см. _set_cosyvoice_voices)."""
        idx = self.voice_combo.current()
        if 0 <= idx < len(self._cosyvoice_voices):
            return self._cosyvoice_voices[idx]
        if self._cosyvoice_voices:
            return self._cosyvoice_voices[0]
        return "default"

    def _refresh_cosyvoice_voices_listbox(self):
        """Перезаполняет список на вкладке «Голоса CosyVoice» текущим
        содержимым self._cosyvoice_voices (обновляется через
        refresh_cosyvoice_voices / после добавления нового голоса)."""
        if not hasattr(self, "cosyvoice_voices_listbox"):
            return
        self.cosyvoice_voices_listbox.delete(0, "end")
        for name in self._cosyvoice_voices:
            self.cosyvoice_voices_listbox.insert("end", name)

    def _on_cosyvoice_voices_listbox_select(self, _event=None):
        """Выбор голоса в списке на вкладке «Голоса» делает его основным
        голосом режима CosyVoice на вкладке «Озвучка» (если этот режим
        сейчас выбран) — чтобы не нужно было переключаться между вкладками
        и заново искать нужный голос в выпадающем списке."""
        sel = self.cosyvoice_voices_listbox.curselection()
        if not sel:
            return
        name = self.cosyvoice_voices_listbox.get(sel[0])
        if self._current_mode() == "cosyvoice" and name in self._cosyvoice_voices:
            self.voice_combo.current(self._cosyvoice_voices.index(name))

    def _delete_selected_cosyvoice_voice(self):
        """Удаляет выбранный в списке на вкладке «Голоса CosyVoice» профиль
        голоса — и на сервисе (cosyvoice_voices.json + сам WAV-файл), и в
        списке в программе. Голос "default" (ставится при установке из
        штатного примера CosyVoice) тоже можно удалить, если он не нужен —
        программа не делает для него исключения."""
        sel = self.cosyvoice_voices_listbox.curselection()
        if not sel:
            messagebox.showinfo("Удалить голос", "Сначала выберите голос в списке слева.")
            return
        name = self.cosyvoice_voices_listbox.get(sel[0])
        if not messagebox.askyesno(
            "Удалить голос",
            f"Удалить голос «{name}»? Это действие нельзя отменить — "
            "образец придётся добавлять заново, если он снова понадобится.",
        ):
            return

        rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
        self._cv_delete_btn.configure(state="disabled")

        def worker():
            try:
                if self._rest_url_is_local(rest_url):
                    self.after(0, lambda: self.log(
                        "Проверяю сервис CosyVoice (при необходимости запущу/перезапущу "
                        "его автоматически)…"
                    ))
                    if not self._ensure_cosyvoice_service_running(rest_url):
                        raise RuntimeError(
                            "Не удалось автоматически запустить сервис CosyVoice — "
                            "подробности в журнале выше и в logs\\fb2_reader_gui.log."
                        )
                import requests
                from urllib.parse import quote
                resp = requests.delete(f"{rest_url}/voices/{quote(name)}", timeout=30)
                if resp.status_code != 200:
                    detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                    raise RuntimeError(f"Сервис CosyVoice вернул ошибку {resp.status_code}: {detail}")
                data = resp.json()
                self.after(0, lambda: self._on_cosyvoice_voice_deleted(data))
            except Exception as e:
                _log_exception_to_file("удаление голоса CosyVoice")
                err = str(e)
                self.after(0, lambda t=err: self._on_cosyvoice_voice_delete_failed(t))

        threading.Thread(target=worker, daemon=True).start()

    def _on_cosyvoice_voice_deleted(self, data: dict):
        self._cv_delete_btn.configure(state="normal")
        self._cosyvoice_voices = data.get("voices") or []
        self._refresh_cosyvoice_voices_listbox()
        self.log(f"Голос «{data.get('name')}» удалён.")
        if self._current_mode() == "cosyvoice":
            self._set_cosyvoice_voices()
            self._populate_dialogue_list()

    def _on_cosyvoice_voice_delete_failed(self, err: str):
        self._cv_delete_btn.configure(state="normal")
        messagebox.showerror("Удалить голос", f"Не удалось удалить голос:\n{err}")
        self.log(f"Не удалось удалить голос CosyVoice: {err}")

    def _browse_cosyvoice_audio(self):
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл с образцом голоса",
            initialdir=self._get_last_dir("cosyvoice_audio") or None,
            filetypes=[
                ("Аудио", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._remember_last_dir("cosyvoice_audio", path)
            self._cv_add_audio_path_var.set(path)
            if not self._cv_add_name_var.get().strip():
                self._cv_add_name_var.set(Path(path).stem)

    def _transcribe_cosyvoice_audio(self):
        """Отправляет выбранный аудиофайл сервису CosyVoice (POST
        /transcribe) на распознавание речи через Whisper и подставляет
        результат в поле «Текст записи» — чтобы не печатать транскрипт
        образца вручную. Модель Whisper для распознавания (~500 МБ)
        скачивается сервисом один раз при первом использовании этой кнопки
        и дальше берётся из кэша."""
        rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
        audio_path = self._cv_add_audio_path_var.get().strip()
        if not audio_path:
            messagebox.showwarning("Распознать текст", "Сначала выберите аудиофайл с образцом голоса.")
            return

        self._cv_transcribe_btn.configure(state="disabled")
        self._cv_add_status_label.configure(
            text="Готовлю файл к отправке на распознавание…"
        )
        # Помимо маленькой подписи под кнопкой (её легко не заметить),
        # дублируем происходящее в основной журнал — включая периодические
        # "ещё работаю" сообщения с указанием текущего этапа, пока идёт
        # сам запрос, чтобы не выглядело, будто программа зависла (первый
        # запуск с загрузкой модели распознавания может занимать пару
        # минут без какого-либо ответа).
        try:
            size_kb = Path(audio_path).stat().st_size / 1024
        except Exception:
            size_kb = 0
        self.log(f"Распознаю текст образца «{Path(audio_path).name}» ({size_kb:.0f} КБ) через Whisper…")

        stop_ticker = threading.Event()

        def _fmt_elapsed(sec: int) -> str:
            return f"{sec // 60} мин {sec % 60:02d} сек" if sec >= 60 else f"{sec} сек"

        def ticker():
            waited = 0
            while not stop_ticker.wait(10.0):
                waited += 10
                if waited <= 30:
                    stage = "отправляю файл на сервис / сервис ещё может качать модель Whisper (~500 МБ, разово)"
                else:
                    stage = "модель Whisper распознаёт речь"
                self.after(0, lambda w=waited, s=stage: self.log(
                    f"…распознавание ещё идёт ({_fmt_elapsed(w)}): {s}…"
                ))

        threading.Thread(target=ticker, daemon=True).start()

        def worker():
            try:
                if self._rest_url_is_local(rest_url):
                    if not self._ensure_cosyvoice_service_running(rest_url):
                        raise RuntimeError(
                            "Не удалось запустить сервис CosyVoice — подробности в файле "
                            f"журнала (logs\\{LOG_FILE_PATH.name}) и в самом окне «Журнал» выше."
                        )
                import requests
                self.after(0, lambda: self.log("Отправляю аудиофайл на сервис…"))
                try:
                    with open(audio_path, "rb") as f:
                        resp = requests.post(
                            f"{rest_url}/transcribe",
                            files={"audio": (Path(audio_path).name, f)},
                            timeout=300,
                        )
                except requests.exceptions.ConnectionError as ce:
                    _log_exception_to_file("подключение к сервису CosyVoice (/transcribe)")
                    raise RuntimeError(
                        f"Не удалось подключиться к сервису CosyVoice по адресу {rest_url}."
                    ) from ce
                if resp.status_code != 200:
                    detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                    raise RuntimeError(f"Сервис CosyVoice вернул ошибку {resp.status_code}: {detail}")
                text = resp.json().get("text", "")
                self.after(0, lambda: self._on_cosyvoice_audio_transcribed(text))
            except Exception as e:
                _log_exception_to_file("распознавание речи CosyVoice")
                err = str(e)
                self.after(0, lambda t=err: self._on_cosyvoice_transcribe_failed(t))
            finally:
                stop_ticker.set()

        threading.Thread(target=worker, daemon=True).start()

    def _on_cosyvoice_audio_transcribed(self, text: str):
        self._cv_transcribe_btn.configure(state="normal")
        if text:
            self._cv_add_text_widget.delete("1.0", "end")
            self._cv_add_text_widget.insert("1.0", text)
            self._cv_add_status_label.configure(text=f"Готово: распознано {len(text)} симв. — проверьте и поправьте при необходимости.")
            self.log(f"Текст распознан ({len(text)} символов): {text}")
        else:
            self._cv_add_status_label.configure(text="Не удалось распознать текст (пустой результат) — впишите вручную.")
            self.log("Распознавание вернуло пустой текст — впишите текст образца вручную.")

    def _on_cosyvoice_transcribe_failed(self, err: str):
        self._cv_transcribe_btn.configure(state="normal")
        self._cv_add_status_label.configure(text=f"Ошибка распознавания: {err}")
        self.log(f"ОШИБКА распознавания текста CosyVoice: {err}")
        messagebox.showerror("Распознать текст — ошибка", err)

    def _add_cosyvoice_voice_from_tab(self):
        """Отправляет форму добавления голоса на вкладке «Голоса CosyVoice»
        сервису CosyVoice REST (POST /add_voice) — добавленный голос
        сохраняется на сервисе постоянно (cosyvoice_voices.json) и остаётся
        доступен после перезапуска, поэтому его достаточно добавить один
        раз."""
        rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
        audio_path = self._cv_add_audio_path_var.get().strip()
        name = self._cv_add_name_var.get().strip()
        text = self._cv_add_text_widget.get("1.0", "end").strip()

        if not audio_path:
            messagebox.showwarning("Добавить голос", "Выберите аудиофайл с образцом голоса.")
            return
        if not name:
            messagebox.showwarning("Добавить голос", "Введите имя для нового голоса.")
            return
        # Движок клонирования (XTTS-v2) сам тексту образца не требует —
        # клонирует по одному только аудио. Текст здесь используется
        # исключительно для истории/справки, поэтому пустое поле больше не
        # блокируется предупреждением (было актуально для прежнего движка
        # CosyVoice, у которого без текста образца включался режим
        # "cross-lingual", не понимавший русский язык).

        self._cv_add_btn.configure(state="disabled")
        self._cv_add_status_label.configure(text="Отправляю запись сервису CosyVoice, подождите…")
        try:
            size_kb = Path(audio_path).stat().st_size / 1024
        except Exception:
            size_kb = 0
        self.log(f"Добавляю голос «{name}» ({Path(audio_path).name}, {size_kb:.0f} КБ)…")

        stop_ticker = threading.Event()

        def _fmt_elapsed(sec: int) -> str:
            return f"{sec // 60} мин {sec % 60:02d} сек" if sec >= 60 else f"{sec} сек"

        def ticker():
            waited = 0
            while not stop_ticker.wait(10.0):
                waited += 10
                self.after(0, lambda w=waited: self.log(
                    f"…добавление голоса «{name}» ещё идёт ({_fmt_elapsed(w)}): сервис обрабатывает "
                    "и клонирует запись (при первом запуске здесь же может ещё загружаться модель CosyVoice)…"
                ))

        threading.Thread(target=ticker, daemon=True).start()

        def worker():
            try:
                if self._rest_url_is_local(rest_url) and not self._ping_rest_service(rest_url):
                    self.after(0, lambda: self._cv_add_status_label.configure(
                        text="Сервис CosyVoice ещё не запущен — запускаю автоматически…"
                    ))
                    self.after(0, lambda: self.log("Сервис CosyVoice ещё не запущен — запускаю автоматически…"))
                    if not self._ensure_cosyvoice_service_running(rest_url):
                        raise RuntimeError(
                            "Не удалось запустить сервис CosyVoice — подробности в файле "
                            f"журнала (logs\\{LOG_FILE_PATH.name}) и в самом окне «Журнал» выше."
                        )
                import requests
                self.after(0, lambda: self.log("Сервис запущен, отправляю аудиофайл и текст…"))
                try:
                    with open(audio_path, "rb") as f:
                        resp = requests.post(
                            f"{rest_url}/add_voice",
                            data={"name": name, "text": text},
                            files={"audio": (Path(audio_path).name, f)},
                            timeout=120,
                        )
                except requests.exceptions.ConnectionError as ce:
                    _log_exception_to_file("подключение к сервису CosyVoice (/add_voice)")
                    raise RuntimeError(
                        f"Не удалось подключиться к сервису CosyVoice по адресу {rest_url} "
                        "(соединение отклонено — сервис не запущен или ещё не успел "
                        "подняться). Проверьте, что install.bat полностью и без ошибок "
                        "поставил CosyVoice (папки CosyVoice\\ и .venv_cosyvoice должны "
                        "существовать), и посмотрите файл журнала на подробности."
                    ) from ce
                if resp.status_code != 200:
                    detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                    _write_log_file(f"\n--- /add_voice вернул {resp.status_code} ---\n{detail}\n")
                    raise RuntimeError(f"Сервис CosyVoice вернул ошибку {resp.status_code}: {detail}")
                data = resp.json()
                self.after(0, lambda: self._on_cosyvoice_voice_added_from_tab(data))
            except Exception as e:
                _log_exception_to_file("добавление голоса CosyVoice")
                err = str(e)
                self.after(0, lambda t=err: self._on_cosyvoice_voice_add_failed_in_tab(t))
            finally:
                stop_ticker.set()

        threading.Thread(target=worker, daemon=True).start()

    def _on_cosyvoice_voice_added_from_tab(self, data: dict):
        self._cv_add_btn.configure(state="normal")
        self._cv_add_status_label.configure(text=f"Готово: голос «{data.get('name')}» добавлен.")
        self._cosyvoice_voices = data.get("voices") or self._cosyvoice_voices
        self._refresh_cosyvoice_voices_listbox()
        if self._current_mode() == "cosyvoice":
            self._set_cosyvoice_voices()
            self._populate_dialogue_list()
            if data.get("name") in self._cosyvoice_voices:
                self.voice_combo.current(self._cosyvoice_voices.index(data["name"]))
        self.log(f"Голос «{data.get('name')}» добавлен и сохранён — доступен в списке «Голос».")
        # очищаем форму, чтобы можно было сразу добавить следующий голос
        self._cv_add_audio_path_var.set("")
        self._cv_add_name_var.set("")
        self._cv_add_text_widget.delete("1.0", "end")

    def _on_cosyvoice_voice_add_failed_in_tab(self, error: str):
        self._cv_add_btn.configure(state="normal")
        self._cv_add_status_label.configure(text=f"Ошибка: {error}")
        self.log(f"ОШИБКА добавления голоса CosyVoice: {error}")
        messagebox.showerror("Добавить голос — ошибка", error)

    def _set_yandex_voice_placeholder(self):
        self.voice_combo.configure(values=["см. блок «Yandex SpeechKit» ниже"], state="disabled")
        self.voice_combo.current(0)

    def _on_mode_changed(self):
        mode = self._current_mode()
        self.mode_desc.configure(text=TTS_MODES.get(mode, ""))
        self._update_mode_dependent_widgets()

    def _show_row(self, *widgets, show):
        """Показывает или полностью убирает из сетки (grid_remove/grid)
        группу виджетов одной строки настроек — используется, чтобы под
        текущий режим озвучки не показывались неактуальные для него поля
        (grid_remove, в отличие от grid_forget, запоминает позицию и
        параметры, так что виджет потом можно просто показать снова)."""
        for w in widgets:
            if show:
                w.grid()
            else:
                w.grid_remove()

    def _update_mode_dependent_widgets(self):
        mode = self._current_mode()
        silero_like = mode in ("silero", "silero_rest")
        pauses_supported = mode in ("silero", "silero_rest", "cosyvoice", "piper")
        self.model_combo.configure(state="readonly" if silero_like else "disabled")
        self.rest_url_entry.configure(state="normal" if mode == "silero_rest" else "disabled")
        self.auto_start_rest_check.configure(state="normal" if mode == "silero_rest" else "disabled")
        self.emphasis_check.configure(state="normal" if mode == "silero_rest" else "disabled")

        yandex_field_state = "normal" if mode == "yandex" else "disabled"
        self.yandex_api_key_entry.configure(state=yandex_field_state)
        self.yandex_folder_id_entry.configure(state=yandex_field_state)
        self.yandex_voice_combo.configure(state="readonly" if mode == "yandex" else "disabled")

        # Прячем строки настроек, неактуальные для текущего режима, чтобы
        # не перегружать интерфейс полями, которые всё равно ни на что не
        # влияют в этом режиме.
        self._show_row(self.model_label, self.model_row, show=silero_like)
        self._show_row(self.sample_rate_label, self.sample_rate_combo, show=silero_like or mode == "cosyvoice")
        self._show_row(self.rest_url_label, self.rest_url_entry, show=mode == "silero_rest")
        self._show_row(self.auto_start_rest_check, show=mode == "silero_rest")
        self._show_row(self.rate_label, self.rate_spin, show=mode == "offline")
        self._show_row(self.cosyvoice_tab_hint, show=mode == "cosyvoice")

        # Блок Yandex SpeechKit и блок интонации/пауз (для Silero и
        # CosyVoice — у CosyVoice нет SSML, но паузы вставляются вручную
        # между кусками текста, см. run_cosyvoice) — целиком показываются/
        # прячутся вместе с рамкой.
        self._yandex_frame.pack_forget()
        self._intonation_frame.pack_forget()
        if mode == "yandex":
            self._yandex_frame.pack(fill="x", pady=(8, 0))
        if pauses_supported:
            self._intonation_frame.pack(fill="x", pady=(8, 0))
        # "Усиливать интонацию «?!»" — это SSML-фича самого Silero REST;
        # ударения/"ё" — фичи самого Silero — у CosyVoice ничего этого нет,
        # прячем эти конкретные строки внутри общей рамки интонации.
        self._show_row(self.accent_check, show=silero_like)
        self._show_row(self.yo_check, show=silero_like)
        self._show_row(self.emphasis_check, show=mode == "silero_rest")

        if mode in ("silero", "silero_rest"):
            self._set_silero_voices()
        elif mode == "offline":
            self._set_offline_voices()
        elif mode == "yandex":
            self._set_yandex_voice_placeholder()
        elif mode == "cosyvoice":
            self._set_cosyvoice_voices()
            if not self._cosyvoice_voices:
                self.refresh_cosyvoice_voices()
        elif mode == "piper":
            self._set_piper_voices()
        else:
            self._set_online_voice()

        self._populate_dialogue_list()

    def _selected_silero_speaker(self) -> str:
        idx = self.voice_combo.current()
        if idx < 0:
            return "xenia"
        return self._silero_voice_keys[idx]

    def _selected_offline_voice_id(self) -> str:
        idx = self.voice_combo.current()
        if idx < 0 or idx >= len(self._offline_voices):
            return ""
        return self._offline_voices[idx]["id"]

    def _selected_yandex_voice(self) -> str:
        idx = self.yandex_voice_combo.current()
        if idx < 0:
            return "alena"
        return self._yandex_voice_keys[idx]

    def _dialogue_voice_keys(self):
        """Ключи голосов, которые сейчас показаны в списке «Разные голоса
        для диалогов» — зависят от текущего режима: спикеры Silero,
        системные голоса offline или голоса Yandex. Порядок соответствует
        порядку строк в self.dialogue_list."""
        mode = self._current_mode()
        if mode in ("silero", "silero_rest"):
            return list(self._silero_voice_keys)
        if mode == "offline":
            return [v["id"] for v in self._offline_voices]
        if mode == "yandex":
            return list(self._yandex_voice_keys)
        if mode == "cosyvoice":
            return list(self._cosyvoice_voices)
        if mode == "piper":
            return list(self._piper_voice_keys) or list(PIPER_VOICES.keys())
        return []

    def _populate_dialogue_list(self):
        """Перезаполняет список голосов для диалогов под текущий режим
        (модель Silero тоже влияет на список спикеров) и включает/выключает
        сам блок в зависимости от того, поддерживает ли режим вообще
        несколько голосов (Google TTS — нет)."""
        mode = self._current_mode()
        self.dialogue_list.delete(0, "end")
        supported = mode in ("silero", "silero_rest", "offline", "yandex", "cosyvoice", "piper")

        if mode in ("silero", "silero_rest"):
            for k in self._silero_voice_keys:
                self.dialogue_list.insert("end", f"{k} — {SPEAKER_LABELS.get(k, k)}")
        elif mode == "offline":
            if not self._offline_voices:
                try:
                    self._offline_voices = list_offline_voices()
                except Exception:
                    pass
            for v in self._offline_voices:
                mark = "★ " if v["is_russian"] else ""
                self.dialogue_list.insert("end", f"{mark}{v['name']}")
        elif mode == "yandex":
            for k, v in YANDEX_VOICES.items():
                self.dialogue_list.insert("end", f"{k} — {v}")
        elif mode == "cosyvoice":
            for k in self._cosyvoice_voices:
                self.dialogue_list.insert("end", k)
            if len(self._cosyvoice_voices) < 2:
                supported = False
        elif mode == "piper":
            for k, v in PIPER_VOICES.items():
                self.dialogue_list.insert("end", f"{k} — {v}")
            if len(PIPER_VOICES) < 2:
                supported = False

        if supported:
            self.dialogue_unavailable_label.pack_forget()
            self.dialogue_check.configure(state="normal")
            self._attribution_section.pack(fill="x")
            # по умолчанию выделены все голоса — реальный набор всё равно
            # отфильтруется в _selected_dialogue_voices (основной голос
            # исключается, недоступные голоса модели отсеются в run_*)
            for i in range(self.dialogue_list.size()):
                self.dialogue_list.selection_set(i)
        else:
            self.dialogue_var.set(False)
            self.dialogue_check.configure(state="disabled")
            self.dialogue_unavailable_label.pack(anchor="w", pady=(2, 0))
            self._attribution_section.pack_forget()

        self._on_dialogue_toggle()

    def _on_dialogue_toggle(self):
        self.dialogue_list.configure(
            state="normal" if self.dialogue_var.get() else "disabled"
        )

    def _on_attribution_toggle(self):
        state = "normal" if self.attribution_var.get() else "disabled"
        self.attribution_api_key_entry.configure(state=state)
        self.attribution_model_entry.configure(state=state)
        self.attribution_folder_id_entry.configure(state=state)
        self.attribution_use_yandex_btn.configure(state=state)
        self.attribution_provider_combo.configure(state="readonly" if self.attribution_var.get() else "disabled")

    def _current_attribution_provider_key(self):
        return self._attribution_provider_by_title.get(
            self.attribution_provider_var.get(), DEFAULT_ATTRIBUTION_PROVIDER
        )

    def _on_attribution_provider_change(self, reset_model=True):
        provider = self._current_attribution_provider_key()
        info = ATTRIBUTION_PROVIDERS[provider]
        if reset_model:
            self.attribution_model_var.set(info["default_model"])
        self.attribution_key_hint_label.configure(text=info["key_hint"])
        key_labels = {
            "yandexgpt": "Yandex API-ключ:", "gemini": "Google API-ключ:",
            "anthropic": "Anthropic API-ключ:",
        }
        self.attribution_key_label.configure(text=key_labels.get(provider, "API-ключ:"))
        needs_folder = provider == "yandexgpt"
        if needs_folder:
            self.attribution_folder_label.master.pack(fill="x", pady=(2, 2))
            self.attribution_use_yandex_btn.pack(anchor="w", pady=(2, 2))
        else:
            self.attribution_folder_label.master.pack_forget()
            self.attribution_use_yandex_btn.pack_forget()

    def _selected_attribution(self):
        """{"api_key":..., "model":..., "provider":...}, если включена галочка
        «Определять, какой персонаж говорит» и есть ключ — иначе None
        (тогда голоса для диалогов просто чередуются по кругу, без LLM)."""
        if not self.attribution_var.get():
            return None
        api_key = self.attribution_api_key_var.get().strip()
        if not api_key:
            return None
        provider = self._current_attribution_provider_key()
        model = self.attribution_model_var.get().strip() \
            or ATTRIBUTION_PROVIDERS[provider]["default_model"]
        folder_id = self.attribution_folder_id_var.get().strip()
        return {"api_key": api_key, "model": model, "provider": provider, "folder_id": folder_id}

    def _selected_dialogue_voices(self):
        """Список ключей голосов для диалогов в текущем режиме, если
        включена галочка «Разные голоса для диалогов» — иначе None (значит,
        диалоги звучат тем же голосом, что и вся книга, как раньше).
        Основной голос («Голос:» / голос Yandex) исключается из списка,
        чтобы диалог не совпадал по тембру с авторской речью."""
        if not self.dialogue_var.get():
            return None
        mode = self._current_mode()
        if mode == "yandex":
            main_voice = self._selected_yandex_voice()
        elif mode in ("silero", "silero_rest"):
            main_voice = self._selected_silero_speaker()
        elif mode == "offline":
            main_voice = self._selected_offline_voice_id()
        elif mode == "cosyvoice":
            main_voice = self._selected_cosyvoice_voice()
        elif mode == "piper":
            main_voice = self._selected_piper_voice()
        else:
            return None
        keys = self._dialogue_voice_keys()
        sel = self.dialogue_list.curselection()
        voices = [keys[i] for i in sel if i < len(keys) and keys[i] != main_voice]
        return voices or None

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.start_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.configure(value=0)
            self.progress_label.configure(text="Подготовка…")
        else:
            self.progress.configure(value=0)
            self.progress_label.configure(text="")

    def _on_synthesis_progress(self, chapter_pos: int, chapters_total: int,
                                fragment_done: int, fragments_total: int):
        """Колбэк из run_silero/run_silero_rest/run_online/run_offline —
        считает реальный процент выполнения (не "бегающую" полоску) и
        обновляет прогресс-бар и подпись под ним. Вызывается из рабочего
        потока, поэтому обновление виджетов идёт через self.after."""
        chapters_total = max(1, chapters_total)
        fragments_total = max(1, fragments_total)
        fraction = ((chapter_pos - 1) + (fragment_done / fragments_total)) / chapters_total
        fraction = min(1.0, max(0.0, fraction))
        percent = fraction * 100

        def update():
            self.progress.configure(value=fraction * 1000)
            self.progress_label.configure(
                text=f"Глава {chapter_pos}/{chapters_total} · фрагмент {fragment_done}/{fragments_total} "
                     f"· {percent:.0f}%"
            )

        self.after(0, update)

    @staticmethod
    def _rest_url_is_local(url: str) -> bool:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1", "")

    @staticmethod
    def _ping_rest_service(url: str, timeout: float = 1.5) -> bool:
        """Проверяет, отвечает ли уже сервис Silero REST по этому адресу.
        Любой ответ сервера (даже 404 на несуществующий путь) значит, что
        сервис поднят — ошибка нас интересует только на уровне соединения
        (сервис не запущен / порт никто не слушает)."""
        try:
            urllib.request.urlopen(url.rstrip("/") + "/docs", timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def _ensure_rest_service_running(self, rest_url: str, ready_timeout: float = 600.0) -> bool:
        """Запускает silero_rest_service.py в фоне (без консоли пользователя)
        и ждёт, пока он не начнёт отвечать на rest_url. При первом запуске
        сервис может несколько минут скачивать модель — поэтому таймаут
        большой, а в журнал периодически пишется, что процесс ещё идёт."""
        if getattr(sys, "frozen", False):
            # Собранная в один .exe версия (PyInstaller) не содержит
            # отдельного python.exe — sys.executable указывает на сам этот
            # exe, поэтому "запустить silero_rest_service.py тем же
            # интерпретатором" тут не сработает. В таком случае проще
            # использовать локальный режим Silero (он не требует сервиса)
            # либо запускать программу из исходников через run_gui.bat.
            self.after(0, lambda: self.log(
                "Автозапуск сервиса Silero REST недоступен в собранной .exe-версии "
                "программы. Используйте режим «Silero (локально)» — он не требует "
                "отдельного сервиса, — либо запустите программу из исходников "
                "(run_gui.bat) и включите автозапуск там."
            ))
            return False
        if self._rest_proc is None or self._rest_proc.poll() is not None:
            service_path = _app_dir() / "silero_rest_service.py"
            if not service_path.exists():
                self.after(0, lambda: self.log(f"Не найден файл сервиса: {service_path}"))
                return False
            try:
                popen_kwargs = dict(
                    cwd=str(service_path.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if sys.platform == "win32":
                    # без отдельного окна консоли для сервиса
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._rest_proc = subprocess.Popen(
                    [sys.executable, str(service_path)], **popen_kwargs
                )
            except Exception as e:
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис Silero REST: {t}"))
                return False

            def pump_output():
                proc = self._rest_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис Silero REST] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            if self._ping_rest_service(rest_url):
                self.after(0, lambda: self.log("Сервис Silero REST запущен и готов принимать запросы."))
                return True
            if self._rest_proc is not None and self._rest_proc.poll() is not None:
                self.after(0, lambda: self.log(
                    f"Сервис Silero REST завершился сам собой (код {self._rest_proc.returncode}) "
                    "во время запуска — см. сообщения [сервис Silero REST] выше."
                ))
                return False
            if time.time() - last_notice > 15:
                last_notice = time.time()
                self.after(0, lambda: self.log(
                    "…сервис Silero REST ещё запускается (при первом запуске может скачивать "
                    "модель, это может занять несколько минут)…"
                ))
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис Silero REST не успел запуститься за отведённое время."))
        return False

    @staticmethod
    def _cosyvoice_model_loaded(url: str, timeout: float = 3.0, expected_version: str = None) -> bool:
        """В отличие от _ping_rest_service (который считает сервис рабочим
        по любому HTTP-ответу), здесь мы проверяем именно то, что модель
        CosyVoice реально загрузилась И что запущенный сервис - это
        актуальная версия кода (через отдельный эндпоинт /health).

        Без первой проверки зависший с прошлого раза процесс с ошибкой при
        загрузке модели выглядел бы для программы как "уже запущен и всё
        хорошо", и она бы просто продолжала слать в него запросы, каждый
        раз получая HTTP 500 без единой новой попытки перезапуска.

        Без второй проверки (версия) уже РАБОТАЮЩИЙ (с успешно загруженной
        моделью) сервис, оставшийся с прошлого запуска программы, считался
        бы полностью готовым даже после того, как сам файл
        cosyvoice_rest_service.py обновился на диске — из-за чего новые
        эндпоинты и исправления молча не действовали бы, пока пользователь
        не перезапустит компьютер вручную (именно так один раз сломалось
        удаление голосов: старый процесс просто не знал про /voices/{name}).

        expected_version - какую версию сервиса считать актуальной; по
        умолчанию COSYVOICE_EXPECTED_SERVICE_VERSION (обычный сервис в
        .venv_cosyvoice), но для CosyVoice3 (см. _ensure_cosyvoice3_service_running)
        передаётся COSYVOICE3_EXPECTED_SERVICE_VERSION - у него отдельный
        файл сервиса и своя нумерация версий."""
        if expected_version is None:
            expected_version = COSYVOICE_EXPECTED_SERVICE_VERSION
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("model_loaded"):
                    return False
                # Сверяем строго, включая случай "поля вообще нет" (значит,
                # это ещё более старая версия сервиса, до появления самой
                # этой проверки, - тем более пора перезапустить).
                if data.get("service_version") != expected_version:
                    return False
                return True
        except Exception:
            return False

    @staticmethod
    def _pids_listening_on_port(port: int) -> set:
        no_window = subprocess.CREATE_NO_WINDOW
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, creationflags=no_window,
        )
        pids = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == "TCP":
                local_addr = parts[1]
                if local_addr.endswith(f":{port}") and "LISTENING" in line.upper():
                    pid = parts[-1]
                    if pid.isdigit() and pid != "0":
                        pids.add(pid)
        return pids

    @classmethod
    def _kill_process_on_port(cls, port: int, wait_free_timeout: float = 8.0):
        """Убивает процесс(ы), слушающие данный TCP-порт на localhost —
        используется, чтобы наверняка избавиться от зависшего с прошлого
        раза сервиса CosyVoice (например, оставшегося после того, как
        программу закрыли не тем способом, или процесс пережил закрытие
        родителя). Работает через стандартные системные утилиты Windows
        (netstat/taskkill), без сторонних зависимостей.

        После taskkill Windows не всегда освобождает порт мгновенно —
        поэтому дальше ждём (опрашивая netstat), пока порт реально не
        освободится, вместо того чтобы полагаться на фиксированную паузу
        (слишком короткая пауза раньше приводила к тому, что новый процесс
        сразу падал с "address already in use", т.к. старый ещё не успел
        полностью закрыть сокет)."""
        if sys.platform != "win32":
            return
        try:
            pids = cls._pids_listening_on_port(port)
            no_window = subprocess.CREATE_NO_WINDOW
            for pid in pids:
                subprocess.run(
                    ["taskkill", "/PID", pid, "/F"],
                    creationflags=no_window, capture_output=True,
                )
            if not pids:
                return
            deadline = time.time() + wait_free_timeout
            while time.time() < deadline:
                if not cls._pids_listening_on_port(port):
                    return
                time.sleep(0.5)
        except Exception:
            _log_exception_to_file(f"попытка остановить процесс на порту {port}")

    def _cosyvoice_engine_code(self) -> str:
        """Внутренний код выбранного движка ("f5"/"xtts") по подписи,
        выбранной в выпадающем списке — уходит в переменную окружения
        TTS_ENGINE процесса сервиса."""
        return COSYVOICE_ENGINE_CODES.get(self.cosyvoice_engine_var.get(), COSYVOICE_DEFAULT_ENGINE)

    def _on_cosyvoice_engine_changed(self, event=None):
        """При смене движка в выпадающем списке уже запущенный сервис
        обслуживает запросы старым движком, пока его не перезапустить -
        останавливаем его здесь же, чтобы следующий запрос сам поднял
        сервис заново с новым TTS_ENGINE (см. _ensure_cosyvoice_service_running
        и сверку engine в _cosyvoice_model_loaded).

        CosyVoice3 - отдельный случай: у него свой подпроцесс/окружение и
        свой порт 5012 (см. _ensure_cosyvoice3_service_running), не
        связанный с движками f5/espeech/f5winter/xtts на порту 5011 -
        здесь просто переключаем REST URL на его порт, чтобы не пришлось
        делать это вручную (сам процесс на 5011, если запущен, трогать не
        нужно - при выборе cosyvoice3 к нему никто и не обращается). При
        переключении ОБРАТНО с cosyvoice3 на любой другой движок -
        возвращаем URL по умолчанию, если пользователь не менял его сам
        вручную на что-то своё."""
        engine = self._cosyvoice_engine_code()
        current_url = self.cosyvoice_rest_url_var.get().strip()
        if engine == "cosyvoice3":
            if current_url in ("", COSYVOICE_DEFAULT_REST_URL):
                self.cosyvoice_rest_url_var.set(COSYVOICE3_DEFAULT_REST_URL)
            return
        if current_url == COSYVOICE3_DEFAULT_REST_URL:
            self.cosyvoice_rest_url_var.set(COSYVOICE_DEFAULT_REST_URL)

        if self._cosyvoice_proc is not None and self._cosyvoice_proc.poll() is None:
            self.log(f"Движок CosyVoice изменён на «{self.cosyvoice_engine_var.get()}» — "
                      "перезапускаю сервис…")
            try:
                rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
                from urllib.parse import urlparse
                port = urlparse(rest_url).port or 5011
            except Exception:
                port = 5011
            self._cosyvoice_proc = None
            threading.Thread(target=lambda: self._kill_process_on_port(port), daemon=True).start()

    def _ensure_cosyvoice3_service_running(self, rest_url: str, ready_timeout: float = 2700.0) -> bool:
        """Аналог _ensure_cosyvoice_service_running, но для CosyVoice3 -
        собственный подпроцесс в СВОЁМ окружении .venv_cosyvoice3 (у
        CosyVoice3 несовместимый со всем остальным набор фиксированных
        версий зависимостей - torch==2.3.1 и т.д., см. install.bat,
        секция CosyVoice3), и свой порт (COSYVOICE3_DEFAULT_REST_URL).

        (Изначально это планировалось запускать в Docker-контейнере (см.
        docker/cosyvoice3/) для полной изоляции - но Docker Desktop/WSL2 у
        пользователя оказался слишком капризным, поэтому вместо контейнера -
        обычный подпроцесс, как у остальных движков.)"""
        if getattr(sys, "frozen", False):
            self.after(0, lambda: self.log(
                "Автозапуск сервиса CosyVoice3 недоступен в собранной .exe-версии "
                "программы. Запустите программу из исходников (run_gui.bat)."
            ))
            return False

        cosyvoice3_py = _app_dir() / ".venv_cosyvoice3" / "Scripts" / "python.exe"
        if not cosyvoice3_py.exists():
            alt = _app_dir() / ".venv_cosyvoice3" / "bin" / "python"
            cosyvoice3_py = alt if alt.exists() else cosyvoice3_py
        service_dir = _app_dir() / "CosyVoice3"
        service_path = service_dir / "cosyvoice3_rest_service.py"

        if not cosyvoice3_py.exists() or not service_path.exists():
            self.after(0, lambda: self.log(
                "Движок CosyVoice3 ещё не установлен на этом компьютере. Запустите "
                "install.bat ещё раз (секция CosyVoice3 — отдельная, большая загрузка, "
                "может занять время)."
            ))
            return False

        try:
            from urllib.parse import urlparse
            port = urlparse(rest_url).port or 5012
        except Exception:
            port = 5012

        need_start = self._cosyvoice3_proc is None or self._cosyvoice3_proc.poll() is not None
        if need_start and self._ping_rest_service(rest_url):
            # self._cosyvoice3_proc only tracks a process THIS instance of
            # the GUI started - it's None right after the program (re)starts
            # even if a service from a PREVIOUS run is still alive and
            # holding the port (its parent GUI process can exit without
            # killing it, since it isn't spawned as a detached/job-linked
            # child). Without this check we would spawn a second copy here,
            # which immediately crashes with "[Errno 10048] address already
            # in use" while the old copy just keeps answering /health - so
            # the GUI would loop "model not loaded... service exited...
            # model not loaded..." forever instead of just using it or
            # replacing it once.
            if self._cosyvoice_model_loaded(rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION):
                self.after(0, lambda: self.log(
                    "Сервис CosyVoice3 уже запущен (остался от предыдущего запуска "
                    "программы) и готов принимать запросы."
                ))
                return True
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice3 (видимо, от предыдущего "
                "запуска программы), но версия/модель не подходят — останавливаю его "
                "и запускаю заново…"
            ))
            self._kill_process_on_port(port)
            time.sleep(1.0)
        elif not need_start and self._ping_rest_service(rest_url) and not self._cosyvoice_model_loaded(
            rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION
        ):
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice3, у которого не загружена "
                "модель — останавливаю его и запускаю заново…"
            ))
            self._cosyvoice3_proc = None
            self._kill_process_on_port(port)
            time.sleep(1.0)
            need_start = True
        if need_start:
            try:
                popen_kwargs = dict(
                    cwd=str(service_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=dict(os.environ, COSYVOICE3_PORT=str(port)),
                )
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._cosyvoice3_proc = subprocess.Popen(
                    [str(cosyvoice3_py), str(service_path)], **popen_kwargs
                )
            except Exception as e:
                _log_exception_to_file("запуск сервиса CosyVoice3")
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice3: {t}"))
                return False

            def pump_output():
                proc = self._cosyvoice3_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис CosyVoice3] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            just_exited = self._cosyvoice3_proc is not None and self._cosyvoice3_proc.poll() is not None
            if not just_exited and self._ping_rest_service(rest_url):
                if self._cosyvoice_model_loaded(rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION):
                    self.after(0, lambda: self.log("Сервис CosyVoice3 запущен и готов принимать запросы."))
                    return True
                if time.time() - last_notice > 15:
                    last_notice = time.time()
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice3 отвечает, но модель ещё не загрузилась (первый "
                        "запуск также скачивает веса, ~1-2 ГБ) — жду… Подробности в "
                        "CosyVoice3\\data\\cosyvoice3_rest_service.log."
                    ))
            if just_exited:
                self.after(0, lambda: self.log(
                    f"Сервис CosyVoice3 завершился сам собой (код {self._cosyvoice3_proc.returncode}) "
                    "— см. CosyVoice3\\data\\cosyvoice3_rest_service.log."
                ))
                return False
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис CosyVoice3 не успел запуститься за отведённое время."))
        return False

    def _ensure_cosyvoice_service_running(self, rest_url: str, ready_timeout: float = 900.0) -> bool:
        """Аналог _ensure_rest_service_running для CosyVoice: запускает
        cosyvoice_rest_service.py тем интерпретатором, что установлен
        install_cosyvoice.bat в .venv_cosyvoice (отдельно от основного
        .venv, т.к. у CosyVoice несовместимый со всем остальным набор
        зависимостей), без отдельного окна консоли. Первый запуск может
        занять несколько минут — модель загружается в память GPU.

        Движок CosyVoice3 обрабатывается отдельно (см.
        _ensure_cosyvoice3_service_running) - у него своё окружение
        (.venv_cosyvoice3) и свой порт."""
        if self._cosyvoice_engine_code() == "cosyvoice3":
            # CosyVoice3 при первом запуске ещё и скачивает веса модели
            # (~1-2 ГБ с HuggingFace) поверх обычной загрузки в память GPU,
            # так что 900 сек (15 мин) не хватало на медленном соединении -
            # сервис реально работал (что подтверждали логи), просто GUI
            # переставал ждать раньше, чем он успевал подняться. Даём
            # заметно больше времени именно для cosyvoice3, если вызывающий
            # код не передал явный ready_timeout.
            cv3_timeout = ready_timeout if ready_timeout != 900.0 else 2700.0
            return self._ensure_cosyvoice3_service_running(rest_url, ready_timeout=cv3_timeout)

        if getattr(sys, "frozen", False):
            self.after(0, lambda: self.log(
                "Автозапуск сервиса CosyVoice недоступен в собранной .exe-версии "
                "программы. Запустите программу из исходников (run_gui.bat)."
            ))
            return False

        cosyvoice_py = _app_dir() / ".venv_cosyvoice" / "Scripts" / "python.exe"
        if not cosyvoice_py.exists():
            # на всякий случай также проверяем "не-Windows" раскладку venv
            alt = _app_dir() / ".venv_cosyvoice" / "bin" / "python"
            cosyvoice_py = alt if alt.exists() else cosyvoice_py
        service_dir = _app_dir() / "CosyVoice"
        service_path = service_dir / "cosyvoice_rest_service.py"

        if not cosyvoice_py.exists() or not service_path.exists():
            self.after(0, lambda: self.log(
                "Режим CosyVoice ещё не установлен на этом компьютере. Запустите "
                "install.bat ещё раз (он теперь заодно ставит и CosyVoice — это "
                "большая, отдельная от остального загрузка, может занять время)."
            ))
            return False

        try:
            from urllib.parse import urlparse
            port = urlparse(rest_url).port or 5011
        except Exception:
            port = 5011
        port_retry_done = False

        need_start = self._cosyvoice_proc is None or self._cosyvoice_proc.poll() is not None
        if not need_start and self._ping_rest_service(rest_url) and not self._cosyvoice_model_loaded(rest_url):
            # Сервис отвечает (например, остался запущенным с прошлого
            # раза), но модель CosyVoice у него не загрузилась — такой
            # "зомби"-процесс нужно принудительно остановить и поднять
            # заново, иначе программа будет считать его рабочим и получать
            # HTTP 500 на каждый запрос без единой новой попытки перезапуска.
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice, у которого не загружена "
                "модель (скорее всего остался с прошлого раза, до недавнего "
                "исправления) — останавливаю его и запускаю заново…"
            ))
            self._cosyvoice_proc = None
            self._kill_process_on_port(port)
            time.sleep(1.0)
            need_start = True
        if need_start:
            try:
                popen_env = dict(os.environ)
                popen_env["TTS_ENGINE"] = self._cosyvoice_engine_code()
                popen_kwargs = dict(
                    cwd=str(service_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=popen_env,
                )
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._cosyvoice_proc = subprocess.Popen(
                    [str(cosyvoice_py), str(service_path)], **popen_kwargs
                )
            except Exception as e:
                _log_exception_to_file("запуск сервиса CosyVoice")
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice: {t}"))
                return False

            def pump_output():
                proc = self._cosyvoice_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис CosyVoice] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            # ВАЖНО: сначала проверяем, не завершился ли только что нами
            # же запущенный процесс, и только потом - пинг. Если проверять
            # в обратном порядке, то в ситуации "старый зомби-процесс всё
            # ещё держит порт" пинг будет успешно отвечать (это ответ
            # зомби, а не нашего свежего процесса), и мы никогда не
            # заметим, что наш новый процесс уже упал с "порт занят".
            just_exited = self._cosyvoice_proc is not None and self._cosyvoice_proc.poll() is not None
            if not just_exited and self._ping_rest_service(rest_url):
                # Сервис отвечает - но это ещё не значит, что модель
                # загрузилась (uvicorn начинает принимать запросы даже
                # если startup_event поймал исключение при загрузке модели
                # - см. _cosyvoice_model_loaded). Дожидаемся именно этого,
                # а не просто первого ответа сервера.
                if self._cosyvoice_model_loaded(rest_url):
                    self.after(0, lambda: self.log("Сервис CosyVoice запущен и готов принимать запросы."))
                    return True
                if time.time() - last_notice > 15:
                    last_notice = time.time()
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice отвечает, но модель ещё не загрузилась (или не "
                        "смогла загрузиться) — жду… Если это долго не проходит, смотрите "
                        "подробности в CosyVoice\\cosyvoice_rest_service.log."
                    ))
                time.sleep(1.0)
                continue
            if just_exited:
                if not port_retry_done:
                    # Частый случай: старый ("зомби") процесс ещё не успел
                    # полностью освободить порт к моменту, когда новый уже
                    # пытался на него забиндиться ("address already in
                    # use") - добираем ещё раз, уже с более долгим
                    # ожиданием освобождения порта, и пробуем один раз ещё.
                    port_retry_done = True
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice завершился сразу после запуска — возможно, порт "
                        "ещё не освободился от предыдущего процесса. Пробую ещё раз…"
                    ))
                    self._cosyvoice_proc = None
                    self._kill_process_on_port(port, wait_free_timeout=12.0)
                    time.sleep(1.0)
                    try:
                        popen_env2 = dict(os.environ)
                        popen_env2["TTS_ENGINE"] = self._cosyvoice_engine_code()
                        popen_kwargs = dict(
                            cwd=str(service_dir),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=popen_env2,
                        )
                        if sys.platform == "win32":
                            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                        self._cosyvoice_proc = subprocess.Popen(
                            [str(cosyvoice_py), str(service_path)], **popen_kwargs
                        )

                        def pump_output2():
                            proc = self._cosyvoice_proc
                            if proc is None or proc.stdout is None:
                                return
                            for line in proc.stdout:
                                line = line.rstrip()
                                if line:
                                    self.after(0, lambda l=line: self.log("  [сервис CosyVoice] " + l))

                        threading.Thread(target=pump_output2, daemon=True).start()
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        _log_exception_to_file("повторный запуск сервиса CosyVoice")
                        err = str(e)
                        self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice: {t}"))
                        return False
                self.after(0, lambda: self.log(
                    f"Сервис CosyVoice завершился сам собой (код {self._cosyvoice_proc.returncode}) "
                    "во время запуска — см. сообщения [сервис CosyVoice] выше."
                ))
                return False
            if time.time() - last_notice > 15:
                last_notice = time.time()
                self.after(0, lambda: self.log(
                    "…сервис CosyVoice ещё запускается (модель грузится в память GPU, "
                    "при первом запуске это может занять несколько минут)…"
                ))
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис CosyVoice не успел запуститься за отведённое время."))
        return False

    def request_stop(self):
        self._stop_requested = True
        self.log("Запрошена остановка после текущей главы…")

    def start_synthesis(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Озвучка", "Озвучка уже выполняется.")
            return
        if not self.chapters:
            messagebox.showwarning("Озвучка", "Сначала загрузите книгу.")
            return

        mode = self._current_mode()
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        start = max(1, int(self.start_var.get()))
        play = self.play_var.get()

        sel = self.chapters_list.curselection()
        chapter_indices = sorted(i + 1 for i in sel) if sel else None
        char_ranges = None
        if chapter_indices and len(chapter_indices) == 1:
            idx = chapter_indices[0]
            n = len(self.chapters[idx - 1][1])
            a, b = int(self.char_from_var.get()), int(self.char_to_var.get())
            a = max(0, min(a, n))
            b = max(a, min(b, n)) if b > 0 else n
            if a > 0 or b < n:
                char_ranges = {idx: (a, b)}

        dialogue_voices = self._selected_dialogue_voices()

        if self.attribution_var.get() and not self.attribution_api_key_var.get().strip():
            messagebox.showwarning(
                "Определение говорящих",
                "Включена галочка «Определять, какой персонаж говорит», но не указан "
                "API-ключ — атрибуция будет пропущена, голоса для диалогов просто будут "
                "чередоваться по кругу, как без неё."
            )
        elif (self.attribution_var.get()
              and self._current_attribution_provider_key() == "yandexgpt"
              and not self.attribution_folder_id_var.get().strip()):
            messagebox.showwarning(
                "Определение говорящих",
                "Для YandexGPT нужен ещё и Folder ID (можно взять тот же, что и для Yandex "
                "SpeechKit, кнопкой в разделе диалогов) — без него атрибуция будет пропущена."
            )
        attribution = self._selected_attribution()

        self._save_settings()

        self._stop_requested = False
        self._set_busy(True)
        if chapter_indices:
            self.log(f"\n--- Старт озвучки: режим {mode}, главы {chapter_indices}"
                      f"{' (кусок)' if char_ranges else ''} ---")
        else:
            self.log(f"\n--- Старт озвучки: режим {mode}, с главы {start} ---")

        def worker():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.log_text)
            self._synthesis_had_failures = None
            try:
                if mode == "online":
                    run_online(self.chapters, outdir, play, start, voice_lang="ru",
                               on_progress=self._on_synthesis_progress,
                               chapter_indices=chapter_indices, char_ranges=char_ranges,
                               dialogue_voices=dialogue_voices,
                               should_stop=lambda: self._stop_requested,
                               play_fn=self._embedded_play)
                elif mode == "silero":
                    run_silero(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_silero_speaker(),
                        int(self.sample_rate_var.get()),
                        play,
                        model_id=self._selected_model_id(),
                        sentence_break_ms=int(self.sentence_break_var.get()),
                        paragraph_break_ms=int(self.paragraph_break_var.get()),
                        comma_break_ms=int(self.comma_break_var.get()),
                        put_accent=self.accent_var.get(),
                        put_yo=self.yo_var.get(),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_speakers=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                elif mode == "silero_rest":
                    rest_url = self.rest_url_var.get().strip()
                    if self.auto_start_rest_var.get() and self._rest_url_is_local(rest_url):
                        if not self._ping_rest_service(rest_url):
                            self.after(0, lambda: self.log(
                                "Сервис Silero REST не отвечает — запускаю его автоматически "
                                "(без отдельной консоли)…"
                            ))
                            if not self._ensure_rest_service_running(rest_url):
                                raise RuntimeError(
                                    "Не удалось автоматически запустить сервис Silero REST. "
                                    "Подробности — в журнале выше. Можно также запустить его "
                                    "вручную (см. README) или выбрать режим «Silero (локально)»."
                                )
                    run_silero_rest(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_silero_speaker(),
                        int(self.sample_rate_var.get()),
                        play,
                        rest_url,
                        int(self.sentence_break_var.get()),
                        int(self.paragraph_break_var.get()),
                        int(self.comma_break_var.get()),
                        emphasize=self.emphasis_var.get(),
                        model_id=self._selected_model_id(),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_speakers=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                elif mode == "yandex":
                    run_yandex(
                        self.chapters,
                        outdir,
                        start,
                        play,
                        self.yandex_api_key_var.get().strip(),
                        self.yandex_folder_id_var.get().strip(),
                        voice=self._selected_yandex_voice(),
                        speed=float(self.yandex_speed_var.get()),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_voices=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                elif mode == "cosyvoice":
                    cv_rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
                    if self._rest_url_is_local(cv_rest_url):
                        # Не просто "не отвечает вообще" - вызываем всегда,
                        # т.к. _ensure_cosyvoice_service_running сам умеет
                        # отличить "сервис отвечает, но модель не
                        # загрузилась" (зомби-процесс с прошлого раза) от
                        # по-настоящему готового сервиса, и перезапускает
                        # его в первом случае.
                        self.after(0, lambda: self.log(
                            "Проверяю сервис CosyVoice (при необходимости запущу/перезапущу "
                            "его автоматически, без отдельной консоли)…"
                        ))
                        if not self._ensure_cosyvoice_service_running(cv_rest_url):
                            raise RuntimeError(
                                "Не удалось автоматически запустить сервис CosyVoice. "
                                "Подробности — в журнале выше и в logs\\fb2_reader_gui.log. "
                                "Проверьте, что install.bat успешно поставил CosyVoice."
                            )
                    cv_stats = run_cosyvoice(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_cosyvoice_voice(),
                        int(self.sample_rate_var.get()),
                        play,
                        cv_rest_url,
                        sentence_break_ms=int(self.sentence_break_var.get()),
                        paragraph_break_ms=int(self.paragraph_break_var.get()),
                        comma_break_ms=int(self.comma_break_var.get()),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_voices=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                    if cv_stats:
                        total_bad = cv_stats.get("failed_fragments", 0) + cv_stats.get("silent_fragments", 0)
                        if total_bad > 0:
                            self._synthesis_had_failures = (
                                f"{total_bad} фрагмент(ов) не удалось озвучить "
                                f"({cv_stats.get('failed_fragments', 0)} ошибок синтеза, "
                                f"{cv_stats.get('silent_fragments', 0)} беззвучных ответов) "
                                f"в {cv_stats.get('chapters_with_failures', 0)} глав(ах) — "
                                "там в файле тишина вместо озвучки. Подробности в журнале выше "
                                "и в logs\\fb2_reader_gui.log."
                            )
                elif mode == "piper":
                    run_piper(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_piper_voice(),
                        play,
                        sentence_break_ms=int(self.sentence_break_var.get()),
                        paragraph_break_ms=int(self.paragraph_break_var.get()),
                        comma_break_ms=int(self.comma_break_var.get()),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_voices=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                else:
                    run_offline(
                        self.chapters,
                        start,
                        int(self.rate_var.get()),
                        "",
                        voice_id=self._selected_offline_voice_id(),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_voice_ids=dialogue_voices,
                        attribution=attribution,
                        outdir=outdir,
                        should_stop=lambda: self._stop_requested,
                    )
                if not self._stop_requested:
                    if self._synthesis_had_failures:
                        warn_text = self._synthesis_had_failures
                        self.after(0, lambda t=warn_text: self.log(
                            f"--- Озвучка завершена С ОШИБКАМИ: {t} ---"
                        ))
                        self.after(0, lambda t=warn_text: messagebox.showwarning(
                            "Озвучка завершена с ошибками", t
                        ))
                    else:
                        self.after(0, lambda: self.log("--- Озвучка завершена ---"))
            except Exception as e:
                # Важно: захватываем e через значение по умолчанию (e=e), а
                # не просто по имени из замыкания - Python неявно удаляет
                # переменную except-блока при выходе из него, и к моменту,
                # когда self.after() вызовет эту лямбду асинхронно, имя "e"
                # снаружи уже не будет существовать (NameError).
                err_text = str(e)
                self.after(0, lambda t=err_text: messagebox.showerror("Ошибка", t))
                self.after(0, lambda t=err_text: self.log(f"ОШИБКА: {t}"))
            finally:
                sys.stdout = old_stdout
                self.after(0, lambda: self._set_busy(False))
                self.after(0, self.refresh_player_for_selection)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # --- встроенный плеер (pygame.mixer) ------------------------------------
    #
    # Единое состояние плеера используется в двух сценариях: (1) ручное
    # прослушивание уже озвученной главы, выбранной в списке слева —
    # трек подгружается автоматически при выборе главы (см.
    # _on_chapters_selection_change), кнопка "Играть" включается только
    # если для выбранной главы уже есть аудиофайл; (2) автопроигрывание
    # сразу после озвучки очередной главы (галочка "Проигрывать после
    # озвучки каждой главы") — передаётся в run_* как play_fn и блокирует
    # вызвавший поток (поток озвучки), пока не доиграет или не нажали
    # "Стоп" — так же, как раньше, только теперь управляемо.
    #
    # Модель позиции: play/pause не используют pygame.mixer.music.pause(),
    # а всегда останавливают и заново запускают трек с нужной позиции
    # (play(start=позиция)) — это немного грубее, чем настоящая пауза, но
    # одинаково надёжно работает и для WAV, и для MP3, и делает перемотку
    # (тот же механизм) тривиальной. Для WAV сама SDL_mixer не всегда
    # умеет начинать не с начала — тогда перемотка/возобновление тихо
    # срабатывает как проигрывание с начала (ограничение библиотеки, не
    # ошибка программы).

    def _ensure_mixer(self):
        """Лениво импортирует и инициализирует pygame.mixer — только когда
        реально понадобилось проигрывание, чтобы не тормозить запуск
        программы. Возвращает модуль pygame или None, если pygame не
        установлен (тогда используется запасной вариант — системный
        проигрыватель, без кнопок и перемотки)."""
        try:
            import pygame
        except ImportError:
            return None
        except Exception as e:
            # На некоторых системах модуль ставится, но не импортируется
            # (например, не хватает системной DLL) — это не обычный
            # "не установлен", покажем причину, чтобы было понятнее, что
            # чинить.
            err = f"{type(e).__name__}: {e}"
            self.after(0, lambda t=err: self.log(f"pygame не загрузился ({t})"))
            return None
        if not self._mixer_ready:
            try:
                pygame.mixer.init()
                self._mixer_ready = True
            except Exception as e:
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось инициализировать аудио (pygame.mixer): {t}"))
                return None
        return pygame

    @staticmethod
    def _fmt_player_time(seconds: float) -> str:
        seconds = max(0, int(seconds or 0))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _audio_duration_seconds(self, path: Path) -> float:
        """Длительность аудиофайла в секундах — точно для WAV (заголовок
        файла), приблизительно для остальных форматов через pydub (если
        установлен; иначе 0 — плеер тогда просто не покажет полосу
        прокрутки, но проигрывание всё равно работает)."""
        try:
            if path.suffix.lower() == ".wav":
                import wave
                with wave.open(str(path), "rb") as wf:
                    return wf.getnframes() / float(wf.getframerate())
            from pydub import AudioSegment
            return len(AudioSegment.from_file(str(path))) / 1000.0
        except Exception:
            return 0.0

    def _get_chapter_audio_path(self, idx: int, title: str) -> Path | None:
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        from fb2_reader import sanitize_filename
        for ext in (".wav", ".mp3"):
            p = outdir / f"{idx:03d}_{sanitize_filename(title)}{ext}"
            if p.exists():
                return p
        return None

    def refresh_player_for_selection(self):
        """Подгружает в плеер аудио выбранной в списке главы, если оно уже
        есть — кнопка "Играть" включается, только если файл найден.
        Вызывается при выборе главы в списке и после завершения озвучки
        (чтобы свежесозданный файл сразу стал доступен для прослушивания).
        Ничего не делает, пока идёт активное проигрывание/автопроигрывание
        (чтобы не оборвать его сменой выделения)."""
        if self._player_state == "playing" and self._player_block_event is not None:
            return  # идёт автопроигрывание после синтеза — не перебиваем
        sel = self.chapters_list.curselection()
        path = None
        if len(sel) == 1 and self.chapters:
            idx = sel[0] + 1
            title, _ = self.chapters[idx - 1]
            path = self._get_chapter_audio_path(idx, title)
        if path is None:
            self._clear_player()
            return
        if path == self._player_path:
            return  # уже загружен этот же файл — не сбрасываем позицию
        self._clear_player()
        self.player_title_label.configure(text=f"⏳ {path.name} — читаю длительность…")
        self._player_load_token += 1
        token = self._player_load_token

        def worker():
            duration = self._audio_duration_seconds(path)
            self.after(0, lambda: self._finish_load_player_track(path, duration, token))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_load_player_track(self, path: Path, duration: float, token: int):
        if token != self._player_load_token:
            return  # пользователь успел выбрать другую главу — отбрасываем
        self._player_path = path
        self._player_duration = duration
        self._player_position = 0.0
        self._player_state = "paused"
        self.player_scale.configure(to=max(duration, 0.01), state="normal" if duration else "disabled")
        self.player_pos_var.set(0.0)
        self.player_title_label.configure(text=path.name)
        self.player_time_label.configure(text=f"0:00 / {self._fmt_player_time(duration)}")
        self.player_play_btn.configure(state="normal", text="▶ Играть")
        self.player_stop_btn.configure(state="disabled")

    def _clear_player(self, label: str | None = None):
        pygame = None
        try:
            import pygame as _pg
            pygame = _pg
        except ImportError:
            pass
        if pygame is not None and self._player_state == "playing":
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._stop_player_tick()
        self._player_path = None
        self._player_duration = 0.0
        self._player_position = 0.0
        self._player_state = "stopped"
        self.player_title_label.configure(
            text=label or ("Глава не озвучена — сначала озвучьте её" if self.chapters else "Глава не выбрана")
        )
        self.player_time_label.configure(text="")
        self.player_pos_var.set(0.0)
        self.player_scale.configure(state="disabled")
        self.player_play_btn.configure(state="disabled", text="▶ Играть")
        self.player_stop_btn.configure(state="disabled")

    def _current_player_position(self) -> float:
        if self._player_state == "playing":
            pos = self._player_position + (time.time() - self._player_play_wall_start)
            return min(pos, self._player_duration) if self._player_duration else pos
        return self._player_position

    def _player_play_pause(self):
        pygame = self._ensure_mixer()
        if pygame is None:
            self.log(
                "pygame не установлен — управление проигрыванием (пауза/перемотка) "
                "недоступно. Запустите install.bat ещё раз, чтобы поставить pygame."
            )
            return
        if self._player_path is None:
            return
        if self._player_state == "playing":
            pygame.mixer.music.stop()
            self._player_position = self._current_player_position()
            self._player_state = "paused"
            self.player_play_btn.configure(text="▶ Играть")
            self.player_stop_btn.configure(state="normal")
            self._stop_player_tick()
            return

        try:
            pygame.mixer.music.load(str(self._player_path))
            pygame.mixer.music.play(start=self._player_position)
        except Exception as e:
            self.log(f"Не удалось проиграть файл: {e}")
            return
        self._player_state = "playing"
        self._player_play_wall_start = time.time()
        self.player_play_btn.configure(text="⏸ Пауза")
        self.player_stop_btn.configure(state="normal")
        self._start_player_tick()

    def _player_stop(self):
        pygame = self._ensure_mixer()
        if pygame is not None:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._player_position = 0.0
        self._player_state = "paused" if self._player_path else "stopped"
        self.player_play_btn.configure(text="▶ Играть")
        self.player_stop_btn.configure(state="disabled")
        self._stop_player_tick()
        self._player_updating_scale = True
        self.player_pos_var.set(0.0)
        self._player_updating_scale = False
        if self._player_duration:
            self.player_time_label.configure(text=f"0:00 / {self._fmt_player_time(self._player_duration)}")
        if self._player_block_event is not None:
            self._player_block_event.set()

    def _start_player_tick(self):
        self._stop_player_tick()
        self._player_tick_job = self.after(200, self._player_tick)

    def _stop_player_tick(self):
        if self._player_tick_job is not None:
            try:
                self.after_cancel(self._player_tick_job)
            except Exception:
                pass
            self._player_tick_job = None

    def _player_tick(self):
        self._player_tick_job = None
        if self._player_state != "playing":
            return
        pygame = self._ensure_mixer()
        pos = self._current_player_position()
        finished = pygame is not None and not pygame.mixer.music.get_busy()
        if finished or (self._player_duration and pos >= self._player_duration):
            self._player_position = 0.0
            self._player_state = "paused" if self._player_path else "stopped"
            self.player_play_btn.configure(text="▶ Играть")
            self.player_stop_btn.configure(state="disabled")
            self._player_updating_scale = True
            self.player_pos_var.set(0.0)
            self._player_updating_scale = False
            if self._player_duration:
                self.player_time_label.configure(text=f"0:00 / {self._fmt_player_time(self._player_duration)}")
            if self._player_block_event is not None:
                self._player_block_event.set()
            return
        if not self._player_seek_dragging:
            self._player_updating_scale = True
            self.player_pos_var.set(pos)
            self._player_updating_scale = False
        self.player_time_label.configure(
            text=f"{self._fmt_player_time(pos)} / {self._fmt_player_time(self._player_duration)}"
        )
        self._player_tick_job = self.after(200, self._player_tick)

    def _on_player_scale_move(self, _value):
        # Срабатывает и от перетаскивания мышью, и от программного
        # .set() (тик плеера) — здесь только обновляем подпись времени
        # вживую при перетаскивании; сама перемотка применяется в
        # _on_player_seek_commit (по отпусканию кнопки мыши), не на
        # каждое промежуточное движение.
        if getattr(self, "_player_updating_scale", False) or not self._player_seek_dragging:
            return
        pos = self.player_pos_var.get()
        self.player_time_label.configure(
            text=f"{self._fmt_player_time(pos)} / {self._fmt_player_time(self._player_duration)}"
        )

    def _on_player_seek_start(self, _event):
        if self._player_path is None or not self._player_duration:
            return "break"
        self._player_seek_dragging = True

    def _on_player_seek_commit(self, _event):
        if not self._player_seek_dragging:
            return
        self._player_seek_dragging = False
        if self._player_path is None or not self._player_duration:
            return
        target = max(0.0, min(self._player_duration, self.player_pos_var.get()))
        pygame = self._ensure_mixer()
        was_playing = self._player_state == "playing"
        if pygame is not None:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._player_position = target
        if was_playing and pygame is not None:
            try:
                pygame.mixer.music.load(str(self._player_path))
                pygame.mixer.music.play(start=target)
                self._player_play_wall_start = time.time()
                self._player_state = "playing"
                self._start_player_tick()
            except Exception as e:
                self.log(f"Не удалось перемотать: {e}")
                self._player_state = "paused"
        else:
            self._player_state = "paused"
        self.player_time_label.configure(
            text=f"{self._fmt_player_time(target)} / {self._fmt_player_time(self._player_duration)}"
        )

    def _embedded_play(self, path: Path):
        """play_fn для run_* — автопроигрывание сразу после озвучки главы
        (галочка "Проигрывать после озвучки каждой главы"). Вызывается из
        потока озвучки и должна там же и блокировать выполнение (иначе
        следующая глава начнёт озвучиваться поверх ещё звучащей) — поэтому
        ждём threading.Event, который выставляется по естественному
        окончанию трека или по нажатию "Стоп" (см. _player_stop/_player_tick)."""
        pygame = self._ensure_mixer()
        if pygame is None:
            self.after(0, lambda: self.log(
                "pygame не установлен — играю через системный проигрыватель "
                "(без кнопок паузы/стопа/перемотки здесь). Запустите install.bat "
                "ещё раз, чтобы поставить pygame и получить управление проигрыванием."
            ))
            try:
                play_file(path)
            except Exception as e:
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось проиграть файл: {t}"))
            return

        duration = self._audio_duration_seconds(path)
        done_event = threading.Event()
        self._player_block_event = done_event

        def start_on_ui_thread():
            self._clear_player()
            self._player_load_token += 1
            self._player_path = path
            self._player_duration = duration
            self._player_position = 0.0
            self._player_state = "paused"
            self.player_scale.configure(to=max(duration, 0.01), state="normal" if duration else "disabled")
            self.player_title_label.configure(text=f"▶ {path.name}")
            self._player_play_pause()

        self.after(0, start_on_ui_thread)
        done_event.wait()
        self._player_block_event = None

    def _on_close(self):
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno("Выход", "Озвучка ещё идёт. Закрыть приложение?"):
                return
        self._save_settings()
        if self._rest_proc is not None and self._rest_proc.poll() is None:
            if messagebox.askyesno(
                "Сервис Silero REST",
                "Сервис Silero REST был запущен этой программой и всё ещё работает "
                "в фоне. Остановить его тоже?",
            ):
                try:
                    self._rest_proc.terminate()
                except Exception:
                    pass
        self.destroy()


def _install_error_logging(app: "AudiobookApp"):
    """Ловит и пишет в logs/fb2_reader_gui.log ЛЮБЫЕ ошибки, включая те,
    что раньше проходили мимо — а именно исключения внутри обработчиков
    событий tkinter (нажатия кнопок, выбор в списках и т.п.): tkinter по
    умолчанию просто печатает их в консоль и на этом всё, а в run_gui.bat
    / собранном .exe консоль пользователь обычно не видит, так что такие
    ошибки были фактически невидимы. Теперь любая такая ошибка попадает и
    в окно «Журнал», и в файл журнала — с полным traceback."""

    def on_tk_error(exc_type, exc_value, exc_tb):
        _log_exception_to_file("обработчик события интерфейса", (exc_type, exc_value, exc_tb))
        try:
            app.log(f"ОШИБКА (см. подробности в файле журнала logs\\{LOG_FILE_PATH.name}): {exc_value}")
        except Exception:
            pass

    app.report_callback_exception = on_tk_error

    def on_thread_excepthook(args):
        # Ошибки в фоновых потоках (threading.Thread) тоже раньше были
        # видны только в консоли (или вообще нигде, если её нет) - здесь
        # они точно так же попадают в файл журнала.
        _log_exception_to_file(
            f"фоновый поток {args.thread.name if args.thread else '?'}",
            (args.exc_type, args.exc_value, args.exc_traceback),
        )

    if hasattr(threading, "excepthook"):
        threading.excepthook = on_thread_excepthook

    def on_sys_excepthook(exc_type, exc_value, exc_tb):
        _log_exception_to_file("необработанное исключение", (exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = on_sys_excepthook

    if _log_file_open_failed:
        try:
            app.log(
                f"ПРЕДУПРЕЖДЕНИЕ: не удалось создать файл журнала в {LOGS_DIR} — "
                "проверьте права на запись в эту папку."
            )
        except Exception:
            pass


def run_gui(initial_book: Path | None = None):
    app = AudiobookApp(initial_book=initial_book)
    _install_error_logging(app)
    app.log(f"Файл журнала: {LOG_FILE_PATH}")
    app.mainloop()


if __name__ == "__main__":
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_gui(initial_book=initial)
