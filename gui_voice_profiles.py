import threading
from pathlib import Path
from tkinter import filedialog, messagebox

from fb2_reader import cosyvoice_list_voices, COSYVOICE_DEFAULT_REST_URL

from gui_shared import (
    LOG_FILE_PATH,
    _log_exception_to_file,
    _write_log_file,
)


class CosyVoiceVoicesMixin:
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
