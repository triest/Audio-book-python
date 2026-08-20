#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Графический интерфейс для fb2_reader.py — выбор книги, голоса и параметров озвучки."""

from __future__ import annotations

import io
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fb2_reader import (
    TTS_MODES,
    list_offline_voices,
    parse_fb2,
    play_file,
    run_offline,
    run_online,
    run_silero,
    run_silero_rest,
)
from silero_config import DEFAULT_SILERO_MODEL, SILERO_MODELS, speaker_choices


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

        self._build_ui()
        self._bind_events()

        if initial_book and initial_book.exists():
            self.load_book(initial_book)

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
        ttk.Entry(file_row, textvariable=self.book_var).pack(side="left", fill="x", expand=True, padx=6)
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
        self.model_combo = ttk.Combobox(
            settings,
            textvariable=self.model_var,
            values=self._model_labels,
            state="readonly",
            width=28,
        )
        self.model_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)

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
        ttk.Entry(out_row, textvariable=self.outdir_var).pack(side="left", fill="x", expand=True)
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
        self.rest_url_entry = ttk.Entry(settings, textvariable=self.rest_url_var)
        self.rest_url_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)

        row += 1
        ttk.Label(settings, text="Скорость (offline):").grid(row=row, column=0, sticky="w", pady=4)
        self.rate_var = tk.IntVar(value=170)
        self.rate_spin = ttk.Spinbox(settings, from_=80, to=300, textvariable=self.rate_var, width=8)
        self.rate_spin.grid(row=row, column=1, sticky="w", pady=4)

        settings.columnconfigure(1, weight=1)

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

        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 6))

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

    def _on_mode_changed(self):
        mode = self._current_mode()
        self.mode_desc.configure(text=TTS_MODES.get(mode, ""))
        self._update_mode_dependent_widgets()

    def _update_mode_dependent_widgets(self):
        mode = self._current_mode()
        silero_like = mode in ("silero", "silero_rest")
        self.model_combo.configure(state="readonly" if silero_like else "disabled")
        self.rest_url_entry.configure(state="normal" if mode == "silero_rest" else "disabled")

        if mode in ("silero", "silero_rest"):
            self._set_silero_voices()
        elif mode == "offline":
            self._set_offline_voices()
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

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.start_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

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

        self._stop_requested = False
        self._set_busy(True)
        self.log(f"\n--- Старт озвучки: режим {mode}, с главы {start} ---")

        def worker():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.log_text)
            try:
                if mode == "online":
                    run_online(self.chapters, outdir, play, start, voice_lang="ru")
                elif mode == "silero":
                    run_silero(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_silero_speaker(),
                        int(self.sample_rate_var.get()),
                        play,
                        model_id=self._selected_model_id(),
                    )
                elif mode == "silero_rest":
                    run_silero_rest(
                        self.chapters,
                        outdir,
                        start,
                        self._selected_silero_speaker(),
                        int(self.sample_rate_var.get()),
                        play,
                        self.rest_url_var.get().strip(),
                        320,
                        550,
                        180,
                        True,
                        model_id=self._selected_model_id(),
                    )
                else:
                    run_offline(
                        self.chapters,
                        start,
                        int(self.rate_var.get()),
                        "",
                        voice_id=self._selected_offline_voice_id(),
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
        self.destroy()


def run_gui(initial_book: Path | None = None):
    app = AudiobookApp(initial_book=initial_book)
    app.mainloop()


if __name__ == "__main__":
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_gui(initial_book=initial)
