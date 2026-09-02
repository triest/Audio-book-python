from __future__ import annotations

import threading
import time
from pathlib import Path

from fb2_reader import play_file


class PlayerMixin:
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
