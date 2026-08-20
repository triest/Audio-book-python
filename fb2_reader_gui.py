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


SETTINGS_PATH = Path(__file__).resolve().parent / "fb2_reader_settings.json"

# Какие поля запоминаются между запусками программы (имя атрибута-переменной
# -> ничего больше не нужно, tk.Variable сама знает свой тип через .get()).
SETTINGS_FIELDS = [
    "mode_var", "model_var", "start_var", "outdir_var", "play_var",
    "sample_rate_var", "rest_url_var", "auto_start_rest_var", "rate_var",
    "sentence_break_var", "paragraph_break_var", "comma_break_var",
    "accent_var", "yo_var", "emphasis_var",
    "yandex_api_key_var", "yandex_folder_id_var", "yandex_voice_var", "yandex_speed_var",
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
        )
        scroll = ttk.Scrollbar(chapters_frame, orient="vertical", command=self.chapters_list.yview)
        self.chapters_list.configure(yscrollcommand=scroll.set)
        self.chapters_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # --- правая колонка: настройки ---
        right = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(right, weight=2)

        settings = ttk.LabelFrame(right, text="Настройки озвучки", padding=10)
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
        ttk.Label(settings, text="Модель:").grid(row=row, column=0, sticky="w", pady=4)
        self._model_keys = list(SILERO_MODELS.keys())
        self._model_labels = [SILERO_MODELS[k]["title"] for k in self._model_keys]
        default_model_idx = self._model_keys.index(DEFAULT_SILERO_MODEL)
        self.model_var = tk.StringVar(value=self._model_labels[default_model_idx])
        model_row = ttk.Frame(settings)
        model_row.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=self._model_labels,
            state="readonly",
            width=22,
        )
        self.model_combo.pack(side="left", fill="x", expand=True)
        self.check_updates_btn = ttk.Button(
            model_row, text="Обновления…", command=self.check_model_updates
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
        ttk.Label(settings, text="Частота (Гц):").grid(row=row, column=0, sticky="w", pady=4)
        self.sample_rate_var = tk.StringVar(value="48000")
        ttk.Combobox(
            settings,
            textvariable=self.sample_rate_var,
            values=["8000", "24000", "48000"],
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Label(settings, text="REST URL:").grid(row=row, column=0, sticky="w", pady=4)
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
        ttk.Label(settings, text="Скорость (offline):").grid(row=row, column=0, sticky="w", pady=4)
        self.rate_var = tk.IntVar(value=170)
        self.rate_spin = ttk.Spinbox(settings, from_=80, to=300, textvariable=self.rate_var, width=8)
        self.rate_spin.grid(row=row, column=1, sticky="w", pady=4)

        settings.columnconfigure(1, weight=1)

        # --- Yandex SpeechKit: ключ, каталог, голос ---
        yandex_frame = ttk.LabelFrame(right, text="Yandex SpeechKit (режим yandex)", padding=10)
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
                 "на каждую главу. Как получить ключ и Folder ID — см. README.",
            foreground="#555", justify="left",
        ).grid(row=yrow, column=0, columnspan=3, sticky="w", pady=(4, 0))

        yandex_frame.columnconfigure(1, weight=1)
        self._yandex_frame = yandex_frame

        # --- интонация: паузы по знакам препинания, ударения, усиление ---
        intonation = ttk.LabelFrame(right, text="Интонация и паузы (silero / silero_rest)", padding=10)
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

        # --- кнопки ---
        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(10, 6))

        self.start_btn = ttk.Button(btn_row, text="Начать озвучку", command=self.start_synthesis)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btn_row, text="Остановить", command=self.request_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Прослушать выбранную главу", command=self.play_selected_chapter).pack(
            side="right"
        )

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1000, value=0)
        self.progress.pack(fill="x", pady=(2, 2))
        self.progress_label = ttk.Label(right, text="", foreground="#555")
        self.progress_label.pack(fill="x", pady=(0, 6))

        log_frame = ttk.LabelFrame(right, text="Журнал", padding=6)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=14, wrap="word", state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

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

        if mode in ("silero", "silero_rest"):
            self._set_silero_voices()
        elif mode == "offline":
            self._set_offline_voices()
        elif mode == "yandex":
            self._set_yandex_voice_placeholder()
        else:
            self._set_online_voice()

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
        if self._rest_proc is None or self._rest_proc.poll() is not None:
            service_path = Path(__file__).resolve().parent / "silero_rest_service.py"
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

        self._save_settings()

        self._stop_requested = False
        self._set_busy(True)
        self.log(f"\n--- Старт озвучки: режим {mode}, с главы {start} ---")

        def worker():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.log_text)
            try:
                if mode == "online":
                    run_online(self.chapters, outdir, play, start, voice_lang="ru",
                               on_progress=self._on_synthesis_progress)
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
                    )
                else:
                    run_offline(
                        self.chapters,
                        start,
                        int(self.rate_var.get()),
                        "",
                        voice_id=self._selected_offline_voice_id(),
                        on_progress=self._on_synthesis_progress,
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
