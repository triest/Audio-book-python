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
    QWEN_TTS_VOICES,
    QWEN_TTS_DEFAULT_VOICE,
    QWEN_TTS_LOCAL_DEFAULT_URL,
    QWEN_TTS_LOCAL_VOICES,
    QWEN_TTS_LOCAL_DEFAULT_VOICE,
    book_manual_stress_overrides_path,
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
    run_qwen_tts,
    run_qwen_tts_local,
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

# Общие константы/функции вынесены в gui_shared.py (см. его docstring) —
# специально, чтобы у миксинов (gui_service_management.py,
# gui_voice_profiles.py, gui_player.py, gui_ui_builder.py) не было обратного
# импорта из fb2_reader_gui, который ломался при запуске программы напрямую
# (python fb2_reader_gui.py) циклической ошибкой импорта.
#
# import gui_shared (модуль целиком, не только from...import) нужен отдельно,
# потому что _log_file_open_failed - изменяемая глобальная переменная внутри
# gui_shared (её меняет gui_shared._write_log_file через global). Если бы мы
# сделали `from gui_shared import _log_file_open_failed`, то получили бы
# независимую копию значения на момент импорта (всегда False) и не увидели
# бы более позднее изменение - поэтому ниже в _install_error_logging нужно
# обращаться именно как gui_shared._log_file_open_failed.
import gui_shared
from gui_shared import (
    LOGS_DIR,
    LOG_FILE_PATH,
    LOG_FILE_MAX_BYTES,
    COSYVOICE_EXPECTED_SERVICE_VERSION,
    COSYVOICE_ENGINE_LABELS,
    COSYVOICE_ENGINE_CODES,
    COSYVOICE_DEFAULT_ENGINE,
    COSYVOICE3_DEFAULT_REST_URL,
    COSYVOICE3_EXPECTED_SERVICE_VERSION,
    _app_dir,
    _write_log_file,
    _log_exception_to_file,
    add_context_menu,
)


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
    "qwen_api_key_var", "qwen_voice_var",
    "qwen_local_url_var", "qwen_local_start_cmd_var", "qwen_local_auto_stop_var",
    "dialogue_var", "attribution_var", "attribution_provider_var",
    "attribution_api_key_var", "attribution_model_var", "attribution_folder_id_var",
]


from gui_service_management import ServiceManagementMixin
from gui_voice_profiles import CosyVoiceVoicesMixin
from gui_player import PlayerMixin
from gui_ui_builder import UIBuilderMixin


class AudiobookApp(ServiceManagementMixin, CosyVoiceVoicesMixin, PlayerMixin, UIBuilderMixin, tk.Tk):
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
        self._qwen_local_proc: subprocess.Popen | None = None
        self._qwen_local_started_by_us = False
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

    def _browse_qwen_local_start_cmd(self):
        path = filedialog.askopenfilename(
            title="Выберите файл запуска локального сервера Qwen3-TTS (.bat/.exe/.cmd)",
            initialdir=self._get_last_dir("qwen_local_start_cmd") or None,
            filetypes=[
                ("Исполняемые файлы", "*.bat *.cmd *.exe"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._remember_last_dir("qwen_local_start_cmd", path)
            self.qwen_local_start_cmd_var.set(path)

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

        qwen_field_state = "normal" if mode == "qwen_tts" else "disabled"
        self.qwen_api_key_entry.configure(state=qwen_field_state)
        qwen_local_field_state = "normal" if mode == "qwen_tts_local" else "disabled"
        self.qwen_local_url_entry.configure(state=qwen_local_field_state)
        self.qwen_local_start_cmd_entry.configure(state=qwen_local_field_state)
        self.qwen_local_start_cmd_btn.configure(state=qwen_local_field_state)
        self.qwen_local_auto_stop_check.configure(state=qwen_local_field_state)
        self.qwen_voice_combo.configure(
            state="readonly" if mode in ("qwen_tts", "qwen_tts_local") else "disabled"
        )
        if mode in ("qwen_tts", "qwen_tts_local"):
            self._refresh_qwen_voice_list(mode)

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
        self._qwen_frame.pack_forget()
        self._intonation_frame.pack_forget()
        if mode == "yandex":
            self._yandex_frame.pack(fill="x", pady=(8, 0))
        if mode in ("qwen_tts", "qwen_tts_local"):
            self._qwen_frame.pack(fill="x", pady=(8, 0))
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
        elif mode in ("qwen_tts", "qwen_tts_local"):
            self._set_qwen_voice_placeholder()
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

    def _set_qwen_voice_placeholder(self):
        self.voice_combo.configure(values=["см. блок «Qwen3-TTS» ниже"], state="disabled")
        self.voice_combo.current(0)

    def _refresh_qwen_voice_list(self, mode: str):
        """Список голосов у облачного (DashScope) и локального (свой
        Gradio-сервер) Qwen3-TTS РАЗНЫЙ — при переключении режима
        перезаполняем self.qwen_voice_combo под нужный список, иначе можно
        было бы выбрать голос, которого у текущего сервера просто нет."""
        voices = QWEN_TTS_LOCAL_VOICES if mode == "qwen_tts_local" else QWEN_TTS_VOICES
        default_voice = QWEN_TTS_LOCAL_DEFAULT_VOICE if mode == "qwen_tts_local" else QWEN_TTS_DEFAULT_VOICE
        new_keys = list(voices.keys())
        if new_keys == getattr(self, "_qwen_voice_keys", None):
            return  # уже тот список — не сбрасываем выбор пользователя зря
        self._qwen_voice_keys = new_keys
        labels = [f"{k} — {v}" for k, v in voices.items()]
        self.qwen_voice_combo.configure(values=labels)
        default_idx = self._qwen_voice_keys.index(default_voice) if default_voice in self._qwen_voice_keys else 0
        self.qwen_voice_var.set(labels[default_idx])

    def _selected_qwen_voice(self) -> str:
        idx = self.qwen_voice_combo.current()
        if idx < 0 or idx >= len(self._qwen_voice_keys):
            return QWEN_TTS_LOCAL_DEFAULT_VOICE if self._current_mode() == "qwen_tts_local" else QWEN_TTS_DEFAULT_VOICE
        return self._qwen_voice_keys[idx]

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
        if mode in ("qwen_tts", "qwen_tts_local"):
            return list(self._qwen_voice_keys)
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
        supported = mode in (
            "silero", "silero_rest", "offline", "yandex", "qwen_tts", "qwen_tts_local",
            "cosyvoice", "piper",
        )

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
        elif mode in ("qwen_tts", "qwen_tts_local"):
            qwen_voices = QWEN_TTS_LOCAL_VOICES if mode == "qwen_tts_local" else QWEN_TTS_VOICES
            for k, v in qwen_voices.items():
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
        elif mode in ("qwen_tts", "qwen_tts_local"):
            main_voice = self._selected_qwen_voice()
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











    def request_stop(self):
        self._stop_requested = True
        self.log("Запрошена остановка после текущей главы…")

    def check_ambiguous_stress(self):
        """Кнопка «Проверить ударения» — сканирует ТЕКСТ выбранной книги (не
        аудио) на предмет слов, у которых ударение зависит от контекста
        (омографы), и открывает готовый текстовый отчёт. Быстрая
        альтернатива тому, чтобы ловить неправильные ударения на слух в
        многочасовой готовой озвучке — см. check_ambiguous_stress.py."""
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Проверка ударений", "Дождитесь окончания текущей озвучки.")
            return
        if not self.book_path:
            messagebox.showwarning("Проверка ударений", "Сначала выберите книгу.")
            return

        try:
            import check_ambiguous_stress
        except Exception as e:
            messagebox.showerror(
                "Проверка ударений",
                f"Не удалось загрузить check_ambiguous_stress.py:\n{e}"
            )
            return

        book_path = self.book_path
        out_path = Path(str(book_path.with_suffix("")) + ".ambiguous_stress_report.txt")

        self.log(f"\n--- Проверка ударений по тексту: {book_path.name} ---")
        self.check_stress_btn.configure(state="disabled")

        def worker():
            try:
                report_text, stats = check_ambiguous_stress.analyze_book(
                    book_path, top=200, max_examples=3,
                    progress_cb=lambda msg: self.after(0, self.log, msg),
                )
                out_path.write_text(report_text, encoding="utf-8")
            except Exception as e:
                self.after(0, self.log, f"Проверка ударений не удалась: {e}")
                self.after(0, messagebox.showerror, "Проверка ударений", str(e))
                self.after(0, lambda: self.check_stress_btn.configure(state="normal"))
                return

            def done():
                self.check_stress_btn.configure(state="normal")
                self.log(
                    f"Отчёт готов: {out_path.name}\n"
                    f"  Раздел 1 (программа решает сама): {stats['auto_unique']} слов, "
                    f"{stats['auto_total']} вхождений\n"
                    f"  Раздел 2 (без автоматики — можно проставить ударения ниже): "
                    f"{stats['plain_unique']} слов, {stats['plain_total']} вхождений"
                )
                self._open_stress_review_dialog(stats, out_path)

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _open_stress_review_dialog(self, stats: dict, report_out_path: Path,
                                    limit: int = 300):
        """Окно со списком спорных слов (раздел 2 отчёта — без автоматики),
        отсортированных по частоте, с полем ввода напротив каждого. Идея
        именно в этом (а не только в чтении отчёта): пользователь САМ
        указывает нужное ударение для конкретных слов книги — программа
        сохраняет их в manual_stress_overrides.json и будет применять при
        каждой следующей озвучке этой (и любой другой) книги, где встретится
        такое же слово. Пустые поля просто пропускаются."""
        plain_words = stats.get("plain_words") or []
        win = tk.Toplevel(self)
        win.title(f"Проверка ударений — {stats.get('book_title', '')}")
        win.geometry("880x640")
        win.transient(self)

        info = ttk.Frame(win, padding=(10, 8))
        info.pack(side="top", fill="x")
        ttk.Label(
            info,
            text=(
                "Щёлкните по ударной гласной букве в слове — вводить ничего не нужно. "
                "Повторный щелчок по той же букве снимает выбор (слово будет пропущено). "
                "Ударения сохраняются ОТДЕЛЬНО ДЛЯ ЭТОЙ КНИГИ (рядом с книгой, файл "
                "«…manual_stress_overrides.json») — то же слово в другой книге можно "
                "разметить иначе. Если у слова вообще не бывает другого варианта ударения "
                "(список составлен по внешнему источнику и не идеален) — жмите «Это не "
                "омограф»: это, наоборот, действует сразу для всех книг, потому что это "
                "не выбор чтения, а факт о самом слове."
            ),
            wraplength=840, justify="left",
        ).pack(anchor="w")
        if len(plain_words) > limit:
            ttk.Label(
                info,
                text=f"Показаны {limit} самых частых слов из {len(plain_words)} — "
                     f"полный список со всеми примерами см. в текстовом отчёте "
                     f"({report_out_path.name}).",
                foreground="#a60", wraplength=840, justify="left",
            ).pack(anchor="w", pady=(4, 0))

        btn_row = ttk.Frame(win, padding=(10, 0, 10, 8))
        btn_row.pack(side="bottom", fill="x")
        status_label = ttk.Label(btn_row, text="")
        status_label.pack(side="left")
        ttk.Button(
            btn_row, text="Открыть полный текстовый отчёт",
            command=lambda: self._open_file_externally(report_out_path),
        ).pack(side="right")
        save_btn = ttk.Button(btn_row, text="Сохранить")
        save_btn.pack(side="right", padx=(0, 8))
        ttk.Button(btn_row, text="Закрыть", command=win.destroy).pack(side="right", padx=(0, 8))

        canvas_holder = ttk.Frame(win)
        canvas_holder.pack(side="top", fill="both", expand=True, padx=10)
        canvas = tk.Canvas(canvas_holder, highlightthickness=0)
        scroll = ttk.Scrollbar(canvas_holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        rows_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=rows_frame, anchor="nw")
        rows_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        win.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>")
                 if e.widget is win else None)

        _RU_VOWELS = set("аеёиоуыэюя")
        letter_font = ("Segoe UI", 12)
        letter_font_selected = ("Segoe UI", 12, "bold", "underline")

        # selections[word] = индекс ударной буквы в word (или None — не выбрано)
        selections: dict[str, "int | None"] = {}
        # letter_labels[word] = [Label, ...] — по одной метке на КАЖДУЮ букву слова,
        # чтобы при выборе можно было сбросить подсветку с предыдущей и включить на новой
        letter_labels: dict[str, list] = {}

        def select_letter(word, idx):
            labels = letter_labels[word]
            prev = selections.get(word)
            if prev == idx:
                # повторный щелчок по той же букве — снимаем выбор
                labels[idx].configure(font=letter_font, foreground="black")
                selections[word] = None
                return
            if prev is not None:
                labels[prev].configure(font=letter_font, foreground="black")
            labels[idx].configure(font=letter_font_selected, foreground="#c00")
            selections[word] = idx

        shown_words = plain_words[:limit]
        for row_i, (word, count, examples) in enumerate(shown_words):
            row = ttk.Frame(rows_frame, padding=(0, 3))
            row.grid(row=row_i, column=0, sticky="ew")
            rows_frame.grid_columnconfigure(0, weight=1)

            head = ttk.Frame(row)
            head.pack(side="top", fill="x")
            ttk.Label(head, text=f"× {count}", width=6, foreground="#555").pack(side="left")

            word_frame = ttk.Frame(head)
            word_frame.pack(side="left", padx=(4, 0))
            selections[word] = None
            labels = []
            letter_labels[word] = labels
            for idx, ch in enumerate(word):
                is_vowel = ch in _RU_VOWELS
                lbl = tk.Label(
                    word_frame, text=ch, font=letter_font,
                    cursor="hand2" if is_vowel else "arrow",
                    padx=0,
                )
                lbl.pack(side="left")
                labels.append(lbl)
                if is_vowel:
                    lbl.bind("<Button-1>", lambda e, w=word, i=idx: select_letter(w, i))

            def mark_not_ambiguous(w=word, rw=row, wf=word_frame):
                try:
                    import fb2_reader
                    fb2_reader.mark_word_not_ambiguous(w)
                except Exception as e:
                    messagebox.showerror("Проверка ударений", f"Не удалось сохранить: {e}")
                    return
                selections.pop(w, None)
                for lbl in letter_labels.get(w, []):
                    lbl.configure(state="disabled", foreground="#aaa", cursor="arrow")
                    lbl.unbind("<Button-1>")
                not_ambig_btn.configure(text="✓ больше не спросим", state="disabled")
                self.log(f'"{w}" отмечено как не омограф — больше не будет появляться '
                         f"в этом списке (not_ambiguous_stress_words_ru.txt).")

            not_ambig_btn = ttk.Button(
                head, text="Это не омограф", command=mark_not_ambiguous,
            )
            not_ambig_btn.pack(side="right", padx=(6, 0))

            if examples:
                _, ctx = examples[0]
                ttk.Label(
                    row, text=ctx, foreground="#666", wraplength=800, justify="left"
                ).pack(side="top", anchor="w", padx=(24, 0))
            ttk.Separator(row, orient="horizontal").pack(side="bottom", fill="x", pady=(4, 0))

        def do_save():
            to_save = {}
            for word, idx in selections.items():
                if idx is None:
                    continue
                to_save[word] = word[:idx] + "+" + word[idx:]
            if not to_save:
                messagebox.showinfo("Проверка ударений", "Ни одна буква не выбрана.")
                return
            book_overrides_path = Path(stats["book_overrides_path"])
            try:
                import fb2_reader
                fb2_reader.save_manual_stress_overrides(to_save, book_overrides_path)
            except Exception as e:
                messagebox.showerror("Проверка ударений", f"Не удалось сохранить: {e}")
                return
            self.log(f"Сохранено вручную заданных ударений для книги "
                      f"«{stats.get('book_title', '')}»: {len(to_save)} "
                      f"({book_overrides_path.name}) — "
                      + ", ".join(f"{w}→{v}" for w, v in sorted(to_save.items())))
            status_label.configure(text=f"Сохранено: {len(to_save)}")
            messagebox.showinfo(
                "Проверка ударений",
                f"Сохранено {len(to_save)} слов(о) в {book_overrides_path.name} — "
                "отдельный файл для ЭТОЙ книги, рядом с ней.\n"
                "Они будут применены при следующей озвучке этой книги.",
            )

        save_btn.configure(command=do_save)

    def open_stress_dictionary_editor(self):
        """Кнопка «Словарь ударений» — полноценное управление всеми вручную
        заданными ударениями: список (со всех источников — основной словарь,
        общие ручные переопределения и, если книга открыта, ещё и её
        собственные), поиск по слову, удаление, плюс форма добавления новой
        записи (используется и для случая, когда слово не входит в отчёт
        «Проверить ударения», но хочется задать ему ударение по привычке —
        см. fb2_reader.add_preferred_stress)."""
        import fb2_reader

        win = tk.Toplevel(self)
        win.title("Словарь ударений")
        win.geometry("760x600")
        win.transient(self)

        # --- поиск + список ---
        search_row = ttk.Frame(win, padding=(10, 8, 10, 4))
        search_row.pack(side="top", fill="x")
        ttk.Label(search_row, text="Поиск:").pack(side="left")
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_row, textvariable=search_var, width=30)
        search_entry.pack(side="left", padx=(6, 0))
        count_label = ttk.Label(search_row, text="", foreground="#555")
        count_label.pack(side="left", padx=(10, 0))

        list_frame = ttk.Frame(win, padding=(10, 0))
        list_frame.pack(side="top", fill="both", expand=True)
        columns = ("word", "stress", "source")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("word", text="Слово")
        tree.heading("stress", text="Ударение")
        tree.heading("source", text="Источник")
        tree.column("word", width=180, anchor="w")
        tree.column("stress", width=180, anchor="w")
        tree.column("source", width=280, anchor="w")
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        book_path = self.book_path

        def _source_label(source_path: Path) -> str:
            if book_path is not None and Path(source_path) == fb2_reader.book_manual_stress_overrides_path(book_path):
                return f"эта книга ({Path(source_path).name})"
            if Path(source_path) == fb2_reader.DEFAULT_STRESS_DICT_PATH:
                return "основной словарь"
            if Path(source_path) == fb2_reader.DEFAULT_MANUAL_STRESS_OVERRIDES_PATH:
                return "общие ручные (все книги)"
            return Path(source_path).name

        all_entries = []  # кэш последней загрузки, чтобы поиск не бил по диску на каждую букву

        def reload_entries():
            nonlocal all_entries
            try:
                all_entries = fb2_reader.list_stress_entries(book_path)
            except Exception as e:
                messagebox.showerror("Словарь ударений", f"Не удалось прочитать словари: {e}")
                all_entries = []
            apply_filter()

        def apply_filter(*_args):
            query = search_var.get().strip().lower()
            tree.delete(*tree.get_children())
            shown = 0
            for e in all_entries:
                if query and query not in e["word"]:
                    continue
                tree.insert("", "end", values=(e["word"], e["stress"], _source_label(e["source"])),
                            tags=(str(e["source"]),))
                shown += 1
            count_label.configure(text=f"{shown} из {len(all_entries)}")

        search_var.trace_add("write", apply_filter)

        # --- удаление выбранной записи ---
        actions_row = ttk.Frame(win, padding=(10, 6))
        actions_row.pack(side="top", fill="x")

        def delete_selected():
            sel = tree.selection()
            if not sel:
                return
            word, stress, _source_display = tree.item(sel[0], "values")
            source_path = Path(tree.item(sel[0], "tags")[0])
            if not messagebox.askyesno(
                "Словарь ударений",
                f'Удалить "{word}" ({stress}) из {source_path.name}?',
            ):
                return
            try:
                fb2_reader.delete_stress_entry(word, source_path)
            except Exception as e:
                messagebox.showerror("Словарь ударений", f"Не удалось удалить: {e}")
                return
            self.log(f'Словарь ударений: удалено "{word}" из {source_path.name}')
            reload_entries()

        ttk.Button(actions_row, text="Удалить выбранное", command=delete_selected).pack(side="left")
        ttk.Button(actions_row, text="Обновить список", command=reload_entries).pack(
            side="left", padx=(6, 0)
        )

        ttk.Separator(win, orient="horizontal").pack(side="top", fill="x", pady=(6, 0))

        # --- добавление новой записи ---
        add_label_row = ttk.Frame(win, padding=(10, 8, 10, 0))
        add_label_row.pack(side="top", fill="x")
        ttk.Label(
            add_label_row,
            text=(
                "Добавить/изменить слово: введите его, нажмите «Показать буквы», "
                "щёлкните по ударной гласной, «Сохранить». Действует сразу для всех книг."
            ),
            wraplength=720, justify="left",
        ).pack(anchor="w")

        entry_row = ttk.Frame(win, padding=(10, 4))
        entry_row.pack(side="top", fill="x")
        word_var = tk.StringVar()
        word_entry = ttk.Entry(entry_row, textvariable=word_var, width=30)
        word_entry.pack(side="left")

        letters_holder = ttk.Frame(win, padding=(10, 10))
        letters_holder.pack(side="top", fill="x")

        status_label = ttk.Label(win, text="", padding=(10, 0))
        status_label.pack(side="top", anchor="w")

        btn_row = ttk.Frame(win, padding=(10, 8))
        btn_row.pack(side="bottom", fill="x")
        save_btn = ttk.Button(btn_row, text="Сохранить", state="disabled")
        save_btn.pack(side="right")
        ttk.Button(btn_row, text="Закрыть", command=win.destroy).pack(side="right", padx=(0, 8))

        _RU_VOWELS = set("аеёиоуыэюя")
        letter_font = ("Segoe UI", 14)
        letter_font_selected = ("Segoe UI", 14, "bold", "underline")

        state = {"word": "", "labels": [], "selected": None}

        def select_letter(idx):
            labels = state["labels"]
            prev = state["selected"]
            if prev == idx:
                labels[idx].configure(font=letter_font, foreground="black")
                state["selected"] = None
                save_btn.configure(state="disabled")
                return
            if prev is not None:
                labels[prev].configure(font=letter_font, foreground="black")
            labels[idx].configure(font=letter_font_selected, foreground="#c00")
            state["selected"] = idx
            save_btn.configure(state="normal")

        def show_letters():
            word = word_var.get().strip().lower()
            for w in letters_holder.winfo_children():
                w.destroy()
            state["word"] = word
            state["labels"] = []
            state["selected"] = None
            save_btn.configure(state="disabled")
            status_label.configure(text="")
            if not word:
                return
            _RU_LETTERS = set("абвгдежзийклмнопрстуфхцчшщъыьэюяё-")
            if not all(ch in _RU_LETTERS for ch in word):
                status_label.configure(
                    text="Похоже, это не одно русское слово — проверьте ввод.",
                    foreground="#a60",
                )
            for idx, ch in enumerate(word):
                is_vowel = ch in _RU_VOWELS
                lbl = tk.Label(
                    letters_holder, text=ch, font=letter_font,
                    cursor="hand2" if is_vowel else "arrow", padx=0,
                )
                lbl.pack(side="left")
                state["labels"].append(lbl)
                if is_vowel:
                    lbl.bind("<Button-1>", lambda e, i=idx: select_letter(i))

        word_entry.bind("<Return>", lambda e: show_letters())
        ttk.Button(entry_row, text="Показать буквы", command=show_letters).pack(
            side="left", padx=(6, 0)
        )

        def on_tree_select(_event):
            sel = tree.selection()
            if not sel:
                return
            word, _stress, _source_display = tree.item(sel[0], "values")
            word_var.set(word)
            show_letters()

        tree.bind("<<TreeviewSelect>>", on_tree_select)

        def do_save():
            idx = state["selected"]
            word = state["word"]
            if idx is None or not word:
                return
            replacement = word[:idx] + "+" + word[idx:]
            try:
                dest = fb2_reader.add_preferred_stress(word, replacement)
            except Exception as e:
                messagebox.showerror("Словарь ударений", f"Не удалось сохранить: {e}")
                return
            self.log(f'Словарь ударений: "{word}" → "{replacement}" (сохранено в {dest.name})')
            status_label.configure(text=f"Сохранено в {dest.name}.", foreground="#080")
            word_var.set("")
            for w in letters_holder.winfo_children():
                w.destroy()
            save_btn.configure(state="disabled")
            word_entry.focus_set()
            reload_entries()

        save_btn.configure(command=do_save)

        reload_entries()
        word_entry.focus_set()

    def _open_file_externally(self, path: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Не удалось открыть файл", f"{path}\n\n{e}")

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
                        book_stress_overrides_path=(
                            book_manual_stress_overrides_path(self.book_path)
                            if self.book_path else None
                        ),
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
                        book_stress_overrides_path=(
                            book_manual_stress_overrides_path(self.book_path)
                            if self.book_path else None
                        ),
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
                elif mode == "qwen_tts":
                    run_qwen_tts(
                        self.chapters,
                        outdir,
                        start,
                        play,
                        self.qwen_api_key_var.get().strip(),
                        voice=self._selected_qwen_voice(),
                        on_progress=self._on_synthesis_progress,
                        chapter_indices=chapter_indices,
                        char_ranges=char_ranges,
                        dialogue_voices=dialogue_voices,
                        attribution=attribution,
                        should_stop=lambda: self._stop_requested,
                        play_fn=self._embedded_play,
                    )
                elif mode == "qwen_tts_local":
                    q_local_url = self.qwen_local_url_var.get().strip() or QWEN_TTS_LOCAL_DEFAULT_URL
                    self.after(0, lambda: self.log(
                        "Проверяю локальный сервер Qwen3-TTS (при необходимости запущу "
                        "автоматически, без отдельной консоли)…"
                    ))
                    if not self._ensure_qwen_local_service_running(
                        q_local_url, self.qwen_local_start_cmd_var.get().strip()
                    ):
                        raise RuntimeError(
                            "Не удалось автоматически запустить локальный сервер Qwen3-TTS. "
                            "Подробности — в журнале выше и в logs\\fb2_reader_gui.log. "
                            "Проверьте команду автозапуска в блоке «Qwen3-TTS» справа."
                        )
                    try:
                        run_qwen_tts_local(
                            self.chapters,
                            outdir,
                            start,
                            play,
                            q_local_url,
                            voice=self._selected_qwen_voice(),
                            on_progress=self._on_synthesis_progress,
                            chapter_indices=chapter_indices,
                            char_ranges=char_ranges,
                            dialogue_voices=dialogue_voices,
                            attribution=attribution,
                            should_stop=lambda: self._stop_requested,
                            play_fn=self._embedded_play,
                        )
                    finally:
                        if self.qwen_local_auto_stop_var.get():
                            self._stop_qwen_local_service()
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

    if gui_shared._log_file_open_failed:
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
