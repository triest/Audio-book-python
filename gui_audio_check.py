"""«Проверка звука готовых глав» — окно, строка статуса ASR на вкладке «Озвучка» и (в `asr_autorepair`) автопочинка при синтезе.

Окно распознаёт готовые главы целиком (whisper через `asr_autorepair.AsrWorker`), сверяет с текстом и показывает места, где Silero
гудит («ммм» вместо слова), растягивает слово, проглатывает кусок или делает необъяснимо долгую паузу: глава / время / тип / что услышано.
«▶ Послушать место» вырезает клип вокруг места; «Пересинтезировать главы с дефектами» помечает главы устаревшими и запускает обычную озвучку
с включённой автопочинкой (она ловит гулы на каждом куске и лечит их). ASR не проверяет ударения — только что текст произнесён.
Логика без Tk (`chapter_wav`, `build_jobs`, `format_time`, `rows_to_csv`) тестируется в `tools/test_accentuation.py`."""
import csv
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import asr_autorepair as ar

KIND_TEXT = {
    "hum": "гул «ммм»",
    "gap": "лишняя пауза/звук",
    "stretch": "растянутое слово",
    "missing": "проглочен текст",
}


def format_time(t: float) -> str:
    t = max(int(t), 0)
    return f"{t // 60}:{t % 60:02d}"


def chapter_wav(outdir, idx: int, title: str) -> Path:
    """File of chapter `idx` (1-based list index, as in the synthesis): NNN_<title>.wav."""
    from fb2_reader import sanitize_filename
    return Path(outdir) / f"{idx:03d}_{sanitize_filename(title)}.wav"


def build_jobs(chapters, outdir, indices=None) -> list:
    """[(idx, title, text, wav_path)] for the chapters (1-based `indices`, None = all) that already have a finished wav."""
    out = []
    for idx in (indices or range(1, len(chapters) + 1)):
        if 1 <= idx <= len(chapters):
            title, text = chapters[idx - 1]
            wav = chapter_wav(outdir, idx, title)
            if wav.exists():
                out.append((idx, title, text, wav))
    return out


def rows_to_csv(rows, path) -> None:
    """rows: [(idx, title, wav, defect)] -> CSV (utf-8 with BOM for Excel)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["глава", "название", "время", "тип", "что услышано"])
        for idx, title, _wav, d in rows:
            w.writerow([idx, title, format_time(d.get("t", 0.0)), KIND_TEXT.get(d["kind"], d["kind"]), d["text"]])


class AudioCheckMixin:

    # ---- strip on the «Озвучка» tab ----
    def _build_audio_check_row(self, parent, row: int):
        self.asr_status_var = tk.StringVar(value="")
        self.asr_status_label = ttk.Label(parent, textvariable=self.asr_status_var, foreground="gray")
        self.asr_status_label.grid(row=row, column=0, sticky="w", pady=(0, 3))
        ttk.Button(parent, text="Проверить звук готовых глав…", command=self._ac_open).grid(
            row=row, column=1, sticky="e", pady=(0, 3))
        try:
            self.asr_repair_var.trace_add("write", lambda *_: self._asr_refresh_status())
        except Exception:
            pass
        self._asr_refresh_status()

    def _asr_refresh_status(self):
        state, text = ar.asr_status()
        if not self.asr_repair_var.get():
            text = "Автопочинка выключена. " + text
            state = "off"
        self.asr_status_var.set(text)
        try:
            self.asr_status_label.configure(foreground={"ok": "#2e7d32", "degraded": "#b26a00"}.get(state, "gray"))
        except Exception:
            pass

    # ---- window ----
    def _ac_open(self):
        win = getattr(self, "_ac_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        self._ac_win = win = tk.Toplevel(self)
        win.title("Проверка звука готовых глав")
        win.geometry("980x560")
        top = ttk.Frame(win)
        top.pack(fill="x", padx=10, pady=(10, 4))
        self._ac_scan_btn = ttk.Button(top, text="Проверить", command=self._ac_scan_toggle)
        self._ac_scan_btn.pack(side="left")
        self._ac_progress = tk.StringVar(
            value="Проверяются выбранные главы в списке (если не выбрано — все озвученные в папке вывода).")
        ttk.Label(top, textvariable=self._ac_progress).pack(side="left", padx=10)
        prog = ttk.Frame(win)
        prog.pack(fill="x", padx=10, pady=(0, 4))
        self._ac_bar = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=1, value=0)
        self._ac_bar.pack(side="left", fill="x", expand=True)
        self._ac_stats = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self._ac_stats, width=52, anchor="e").pack(side="left", padx=(10, 0))
        self._ac_t0 = 0.0
        self._ac_total = 0
        self._ac_n_done = 0

        mid = ttk.Frame(win)
        mid.pack(fill="both", expand=True, padx=10, pady=4)
        cols = ("chapter", "time", "kind", "text")
        self._ac_tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="extended")
        for c, title, width in (("chapter", "Глава", 130), ("time", "Время", 60), ("kind", "Что нашли", 150),
                                ("text", "Услышано / контекст", 600)):
            self._ac_tree.heading(c, text=title)
            self._ac_tree.column(c, width=width, anchor="w")
        sb = ttk.Scrollbar(mid, orient="vertical", command=self._ac_tree.yview)
        self._ac_tree.configure(yscrollcommand=sb.set)
        self._ac_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bot = ttk.Frame(win)
        bot.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(bot, text="▶ Послушать место", command=self._ac_play).pack(side="left")
        ttk.Button(bot, text="Пересинтезировать главы с дефектами (с автопочинкой)",
                   command=self._ac_resynth).pack(side="left", padx=8)
        ttk.Button(bot, text="Сохранить список (CSV)…", command=self._ac_export).pack(side="left")
        self._ac_rows: dict = {}      # tree iid -> (idx, title, wav, defect)
        self._ac_q: "queue.Queue" = queue.Queue()
        self._ac_cancel = threading.Event()
        self._ac_failed = False
        self._ac_thread = None
        self._ac_tree.bind("<Double-1>", lambda _e: self._ac_play())

    def _ac_scanning(self) -> bool:
        return bool(self._ac_thread and self._ac_thread.is_alive())

    def _ac_scan_toggle(self):
        if self._ac_scanning():
            self._ac_cancel.set()
            self._ac_progress.set("Останавливаю после текущей главы…")
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Проверка звука", "Дождитесь окончания озвучки: видеокарта и распознавание заняты.")
            return
        if not getattr(self, "chapters", None):
            messagebox.showwarning("Проверка звука", "Сначала загрузите книгу.")
            return
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        sel = self.chapters_list.curselection()
        jobs = build_jobs(self.chapters, outdir, sorted(i + 1 for i in sel) if sel else None)
        if not jobs:
            messagebox.showinfo("Проверка звука", "В папке вывода нет готовых файлов выбранных глав.")
            return
        state, text = ar.asr_status()
        if state == "off":
            messagebox.showwarning("Проверка звука", text)
            return
        self._ac_tree.delete(*self._ac_tree.get_children())
        self._ac_rows.clear()
        self._ac_cancel.clear()
        self._ac_failed = False
        self._ac_scan_btn.configure(text="Остановить")
        self._ac_progress.set(f"Загружаю модель распознавания… (глав: {len(jobs)})")
        self._ac_t0 = time.monotonic()
        self._ac_total = len(jobs)
        self._ac_n_done = 0
        self._ac_bar.configure(mode="indeterminate")
        self._ac_bar.start(80)
        self._ac_update_stats()
        self._ac_thread = threading.Thread(target=self._ac_scan_thread, args=(jobs,), daemon=True)
        self._ac_thread.start()
        self.after(250, self._ac_poll)

    def _ac_scan_thread(self, jobs):
        q = self._ac_q
        worker = ar.AsrWorker(log=lambda m: q.put(("log", m)), model=ar.CHAPTER_SCAN_MODEL)
        try:
            if not worker.start():
                q.put(("error", "Не удалось запустить распознавание (подробности в журнале)."))
                return
            for n, (idx, title, text, wav) in enumerate(jobs, 1):
                if self._ac_cancel.is_set():
                    break
                q.put(("progress", f"Распознаю главу {idx} «{title}» ({n}/{len(jobs)})…"))
                try:
                    defects = ar.scan_chapter(worker, wav, text)
                except Exception as e:  # noqa: BLE001
                    q.put(("log", f"Проверка звука: глава {idx}: {e}"))
                    q.put(("rows", idx, title, wav, []))
                    continue
                q.put(("rows", idx, title, wav, defects))
        finally:
            worker.close()
            q.put(("done",))

    def _ac_update_stats(self):
        """Counters under the bar: chapters done, places found, elapsed and estimated remaining time."""
        if not self._ac_scanning() and self._ac_n_done == 0:
            return
        el = time.monotonic() - self._ac_t0
        n, total = self._ac_n_done, self._ac_total
        found = len(self._ac_rows)
        eta = f", осталось ~{format_time(el / n * (total - n))}" if n and n < total else ""
        self._ac_stats.set(f"глав {n}/{total} · мест {found} · прошло {format_time(el)}{eta}")

    def _ac_poll(self):
        """Main thread only: the scan thread never touches Tk, it posts to the queue."""
        try:
            while True:
                msg = self._ac_q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self._ac_progress.set(msg[1])
                    if str(self._ac_bar.cget("mode")) == "indeterminate":
                        self._ac_bar.stop()
                        self._ac_bar.configure(mode="determinate", maximum=max(self._ac_total, 1), value=self._ac_n_done)
                elif kind == "log":
                    self.log(msg[1])
                elif kind == "error":
                    self._ac_failed = True
                    self._ac_progress.set(msg[1])
                    self.log("Проверка звука: " + msg[1])
                elif kind == "rows":
                    _, idx, title, wav, defects = msg
                    for d in defects:
                        iid = self._ac_tree.insert("", "end", values=(
                            f"{idx}. {title}", format_time(d.get("t", 0.0)), KIND_TEXT.get(d["kind"], d["kind"]), d["text"]))
                        self._ac_rows[iid] = (idx, title, wav, d)
                    self._ac_n_done += 1
                    self._ac_bar.configure(value=self._ac_n_done)
                elif kind == "done":
                    self._ac_bar.stop()
                    self._ac_bar.configure(mode="determinate", maximum=max(self._ac_total, 1), value=self._ac_n_done)
                    self._ac_update_stats()
                    self._ac_scan_btn.configure(text="Проверить")
                    if self._ac_failed:
                        return
                    n = len(self._ac_rows)
                    chs = len({r[0] for r in self._ac_rows.values()})
                    self._ac_progress.set(
                        f"Готово: найдено мест — {n}, глав с дефектами — {chs}." if n else "Готово: дефектов не найдено.")
                    self._ac_scan_btn.configure(text="Проверить")
                    self.log(f"Проверка звука готовых глав: мест {n}, глав с дефектами {chs}.")
                    return
        except queue.Empty:
            pass
        self._ac_update_stats()
        if self._ac_scanning():
            self.after(250, self._ac_poll)

    def _ac_selected(self):
        sel = self._ac_tree.selection()
        return [self._ac_rows[i] for i in sel if i in self._ac_rows]

    def _ac_play(self):
        rows = self._ac_selected()
        if not rows:
            messagebox.showinfo("Послушать", "Выберите место в списке.")
            return
        idx, title, wav, d = rows[0]
        try:
            clip = ar.cut_clip(wav, d.get("t", 0.0))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Послушать", f"Не удалось вырезать клип: {e}")
            return
        from fb2_reader import play_file
        threading.Thread(target=lambda: play_file(Path(clip)), daemon=True).start()

    def _ac_resynth(self):
        if self._ac_scanning():
            messagebox.showinfo("Пересинтез", "Дождитесь окончания проверки.")
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Пересинтез", "Дождитесь окончания текущей озвучки.")
            return
        rows = self._ac_selected() or list(self._ac_rows.values())
        indices = sorted({r[0] for r in rows})
        if not indices:
            messagebox.showinfo("Пересинтез", "Нет глав с дефектами.")
            return
        if not messagebox.askyesno(
                "Пересинтез",
                f"Пересинтезировать глав: {len(indices)} ({', '.join(map(str, indices[:12]))}{'…' if len(indices) > 12 else ''})?\n"
                "Готовые файлы этих глав будут заменены; автопочинка включится."):
            return
        from gui_stress_editor import stale_meta_files
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        for m in stale_meta_files(outdir, self.chapters, indices):
            try:
                m.unlink()
            except OSError:
                pass
        self.asr_repair_var.set(True)
        self.chapters_list.selection_clear(0, "end")
        for i in indices:
            self.chapters_list.selection_set(i - 1)
        self._on_chapters_selection_change()
        self.log(f"Проверка звука: пересинтез глав {indices} с автопочинкой.")
        self.notebook.select(0)
        self._ac_resynth_chapters = list(indices)
        self._ac_resynth_t0 = time.monotonic()
        self._ac_resynth_on = True
        self._ac_bar.stop()
        self._ac_bar.configure(mode="determinate", maximum=1000, value=0)
        self._ac_progress.set(f"Пересинтез глав: {', '.join(map(str, indices[:12]))}{'…' if len(indices) > 12 else ''}")
        self._ac_stats.set("подготовка…")
        self.start_synthesis()
        self.after(1000, self._ac_resynth_watch)

    def ac_on_synthesis_progress(self, chapter_pos, chapters_total, fragment_done, fragments_total):
        """Called (main thread) by the synthesis progress callback: mirrors it into the sound-check window during a resynthesis."""
        if not getattr(self, "_ac_resynth_on", False):
            return
        try:
            if not self._ac_win.winfo_exists():
                return
            chapters_total, fragments_total = max(1, chapters_total), max(1, fragments_total)
            fraction = min(1.0, max(0.0, ((chapter_pos - 1) + fragment_done / fragments_total) / chapters_total))
            el = time.monotonic() - self._ac_resynth_t0
            eta = f", осталось ~{format_time(el / fraction * (1 - fraction))}" if fraction > 0.02 else ""
            done = chapter_pos - 1
            self._ac_bar.configure(value=fraction * 1000)
            self._ac_stats.set(f"глава {chapter_pos}/{chapters_total} · фрагмент {fragment_done}/{fragments_total} · "
                               f"{fraction * 100:.0f}% · прошло {format_time(el)}{eta}")
            self._ac_progress.set(f"Пересинтез: глава {chapter_pos} из {chapters_total} "
                                  f"(готово глав: {done}). Прогресс виден и на вкладке «Озвучка».")
        except Exception:  # noqa: BLE001
            pass

    def _ac_resynth_watch(self):
        """Ends the resynthesis display when the synthesis thread has finished."""
        if not getattr(self, "_ac_resynth_on", False):
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            self.after(1000, self._ac_resynth_watch)
            return
        self._ac_resynth_on = False
        try:
            if self._ac_win.winfo_exists():
                el = time.monotonic() - self._ac_resynth_t0
                self._ac_bar.configure(value=1000)
                self._ac_stats.set(f"пересинтез завершён за {format_time(el)}")
                self._ac_progress.set("Пересинтез завершён. Нажмите «Проверить», чтобы убедиться, что дефекты ушли.")
        except Exception:  # noqa: BLE001
            pass

    def _ac_export(self):
        if not self._ac_rows:
            messagebox.showinfo("Сохранить", "Список пуст.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="sound_check.csv")
        if path:
            rows_to_csv(list(self._ac_rows.values()), path)
            self.log(f"Список сохранён: {path}")
