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
    list_offline_voices,
    parse_fb2,
    play_file,
    run_offline,
    run_online,
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


class TextRedirector(io.TextIOBase):
    """Перенаправляет print() в текстовое поле GUI."""

    def __init__(self, widget: tk.Text, tag: str = "log"):
        self.widget = widget
        self.tag = tag

    def write(self, s: str) -> int:
        if not s:
            return 0
        self.widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str):
        self.widget.configure(state="normal")
        self.widget.insert("end", s, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


def _app_dir() -> Path:
    """Папка, где лежит программа. В обычном запуске (python fb2_reader_gui.py)
    это папка со скриптом. В собранной PyInstaller-версии (.exe) __file__
    указывает на временную папку распаковки, которая меняется при каждом
    запуске — поэтому там нужно брать папку, где лежит сам .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


SETTINGS_PATH = _app_dir() / "fb2_reader_settings.json"

# Какие поля запоминаются между запусками программы (имя атрибута-переменной
# -> ничего больше не нужно, tk.Variable сама знает свой тип через .get()).
SETTINGS_FIELDS = [
    "mode_var", "model_var", "start_var", "outdir_var", "play_var",
    "sample_rate_var", "rest_url_var", "auto_start_rest_var", "rate_var",
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
        self._rest_proc: subprocess.Popen | None = None

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

        outer = ttk.Frame(self, padding=10)
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
        ttk.Button(btn_row, text="Прослушать выбранную главу", command=self.play_selected_chapter).pack(
            side="right"
        )

        self.progress_label = ttk.Label(right, text="", foreground="#555")
        self.progress_label.pack(side="bottom", fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1000, value=0)
        self.progress.pack(side="bottom", fill="x", pady=(2, 2))

        log_frame = ttk.LabelFrame(right, text="Журнал", padding=6)
        log_frame.pack(side="bottom", fill="both", expand=True, pady=(8, 0))

        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
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
            right_scroll, text="Интонация и паузы (silero / silero_rest)", padding=10
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
        ttk.Checkbutton(intonation, text="Расставлять ударения автоматически", variable=self.accent_var).grid(
            row=irow, column=0, columnspan=2, sticky="w", pady=3
        )

        irow += 1
        self.yo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(intonation, text='Заменять "е" на "ё" где нужно', variable=self.yo_var).grid(
            row=irow, column=0, columnspan=2, sticky="w", pady=3
        )

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

    def _bind_events(self):
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def browse_book(self):
        path = filedialog.askopenfilename(
            title="Выберите FB2-книгу",
            filetypes=[
                ("FB2 книги", "*.fb2 *.fb2.zip *.zip"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self.book_var.set(path)
            self.reload_book()

    def browse_outdir(self):
        path = filedialog.askdirectory(title="Папка для сохранения аудио")
        if path:
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
        self.log(f"Загружена книга «{title}» — {len(chapters)} глав.")

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
        self._show_row(self.sample_rate_label, self.sample_rate_combo, show=silero_like)
        self._show_row(self.rest_url_label, self.rest_url_entry, show=mode == "silero_rest")
        self._show_row(self.auto_start_rest_check, show=mode == "silero_rest")
        self._show_row(self.rate_label, self.rate_spin, show=mode == "offline")

        # Блок Yandex SpeechKit и блок интонации/пауз (актуален только для
        # Silero) — целиком показываются/прячутся вместе с рамкой.
        self._yandex_frame.pack_forget()
        self._intonation_frame.pack_forget()
        if mode == "yandex":
            self._yandex_frame.pack(fill="x", pady=(8, 0))
        if silero_like:
            self._intonation_frame.pack(fill="x", pady=(8, 0))

        if mode in ("silero", "silero_rest"):
            self._set_silero_voices()
        elif mode == "offline":
            self._set_offline_voices()
        elif mode == "yandex":
            self._set_yandex_voice_placeholder()
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
        return []

    def _populate_dialogue_list(self):
        """Перезаполняет список голосов для диалогов под текущий режим
        (модель Silero тоже влияет на список спикеров) и включает/выключает
        сам блок в зависимости от того, поддерживает ли режим вообще
        несколько голосов (Google TTS — нет)."""
        mode = self._current_mode()
        self.dialogue_list.delete(0, "end")
        supported = mode in ("silero", "silero_rest", "offline", "yandex")

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
                self.after(0, lambda: self.log(f"Не удалось запустить сервис Silero REST: {e}"))
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
            try:
                if mode == "online":
                    run_online(self.chapters, outdir, play, start, voice_lang="ru",
                               on_progress=self._on_synthesis_progress,
                               chapter_indices=chapter_indices, char_ranges=char_ranges,
                               dialogue_voices=dialogue_voices)
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
                    )
                if not self._stop_requested:
                    self.after(0, lambda: self.log("--- Озвучка завершена ---"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, lambda: self.log(f"ОШИБКА: {e}"))
            finally:
                sys.stdout = old_stdout
                self.after(0, lambda: self._set_busy(False))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def play_selected_chapter(self):
        if not self.chapters:
            messagebox.showwarning("Проигрывание", "Сначала загрузите книгу.")
            return
        sel = self.chapters_list.curselection()
        if not sel:
            messagebox.showinfo("Проигрывание", "Выберите главу в списке.")
            return
        idx = sel[0] + 1
        title, _ = self.chapters[idx - 1]
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        from fb2_reader import sanitize_filename

        ext = ".wav" if self._current_mode() in ("silero", "silero_rest") else ".mp3"
        fname = f"{idx:03d}_{sanitize_filename(title)}{ext}"
        path = outdir / fname
        if not path.exists():
            alt_ext = ".mp3" if ext == ".wav" else ".wav"
            path = outdir / f"{idx:03d}_{sanitize_filename(title)}{alt_ext}"
        if not path.exists():
            messagebox.showwarning("Проигрывание", f"Аудиофайл не найден:\n{path}\n\nСначала озвучьте главу.")
            return
        try:
            play_file(path)
        except Exception as e:
            messagebox.showerror("Проигрывание", str(e))

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


def run_gui(initial_book: Path | None = None):
    app = AudiobookApp(initial_book=initial_book)
    app.mainloop()


if __name__ == "__main__":
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_gui(initial_book=initial)
