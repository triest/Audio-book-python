"""Вкладка «Проверка ударений» (объединила бывшие «Вопросы по ударению» и «Сомнительные места»): список мест книги, где ударение
сомнительно — имена, термины, нестабильные слова, расхождения цепочки с независимым корпусом, слова из поля «Свои слова». У каждого места: контекст, «как звучит в готовой главе» (клип по кэшу ASR), «как прозвучит по текущим правилам»,
варианты ударения с ▶ и решение: подтвердить / исправить «только здесь» (фраза в файле правок книги) / исправить «везде» (словарь).
Логика — `stress_review` (без Tk). Записи идут в те же файлы, что и вкладка «Словарь ударений».
Ударение в звуке подтверждает только слух: вкладка находит место, даёт послушать и записывает решение."""
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import stress_review as srv


def _parse_chapters(text: str, n_chapters: int):
    """'28-29, 31' -> {28, 29, 31}; empty -> None (all chapters)."""
    text = (text or "").strip()
    if not text:
        return None
    out = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"не понял «{part}» (пример: 28-29, 31)")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        out.update(range(a, b + 1))
    return {i for i in out if 0 <= i < n_chapters}


class StressReviewMixin:

    def _build_stress_review_tab(self, parent):
        self._sv_moments: list = []
        self._sv_rows: dict = {}
        self._sv_cur = None
        self._sv_occ = 0
        self._sv_thread = None
        self._sv_cancel = None
        self._sv_state = srv.load_state()
        self._sv_queue: "queue.Queue" = queue.Queue()
        self._sv_asr_cache: dict = {}
        self.sv_status_var = tk.StringVar(value="Загрузите книгу на вкладке «Озвучка», отметьте, что искать, и нажмите «Найти».")
        self.sv_chapters_var = tk.StringVar(value="")
        self.sv_words_var = tk.StringVar(value="")
        self.sv_corpus_var = tk.BooleanVar(value=True)
        self.sv_names_var = tk.BooleanVar(value=True)
        self.sv_terms_var = tk.BooleanVar(value=True)
        self.sv_unstable_var = tk.BooleanVar(value=False)
        self.sv_hide_done_var = tk.BooleanVar(value=True)
        self.sv_choice_var = tk.StringVar(value="")
        self.sv_custom_var = tk.StringVar(value="")
        self.sv_scope_var = tk.StringVar(value="here")
        self.sv_llm_backend_var = tk.StringVar(value="Встроенная")
        self.sv_llm_model_var = tk.StringVar(value="")

        parent.columnconfigure(0, weight=2, uniform="sv")
        parent.columnconfigure(1, weight=3, uniform="sv")
        parent.rowconfigure(2, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(top, text="Искать:").pack(side="left")
        for text, var in (("имена", self.sv_names_var), ("термины", self.sv_terms_var),
                          ("нестабильные слова", self.sv_unstable_var), ("расхождения с корпусом", self.sv_corpus_var)):
            ttk.Checkbutton(top, text=text, variable=var).pack(side="left", padx=(8, 0))
        self.sv_progress = ttk.Progressbar(top, mode="indeterminate", length=90)
        self.sv_progress.pack(side="right")
        top2 = ttk.Frame(parent)
        top2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 0))
        ttk.Label(top2, text="Главы (пусто = все):").pack(side="left")
        ttk.Entry(top2, textvariable=self.sv_chapters_var, width=12).pack(side="left", padx=(4, 10))
        ttk.Label(top2, text="Свои слова (через запятую):").pack(side="left")
        ttk.Entry(top2, textvariable=self.sv_words_var, width=24).pack(side="left", padx=(4, 10))
        self.sv_scan_btn = ttk.Button(top2, text="Найти", command=self._sv_scan_toggle)
        self.sv_scan_btn.pack(side="left")
        self.check_stress_btn = ttk.Button(top2, text="Показать текст главы с ударениями", command=self.check_ambiguous_stress)
        self.check_stress_btn.pack(side="left", padx=10)
        ttk.Checkbutton(top2, text="скрыть решённые", variable=self.sv_hide_done_var, command=self._sv_fill).pack(side="left")

        left = ttk.Frame(parent)
        left.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.sv_tree = ttk.Treeview(left, columns=("word", "reading", "kind", "count"), show="headings", selectmode="browse")
        for col, text, w, anchor, stretch in (("word", "Слово", 110, "w", True), ("reading", "Сейчас", 110, "w", True),
                                              ("kind", "Тип", 90, "w", False), ("count", "Раз", 44, "center", False)):
            self.sv_tree.heading(col, text=text)
            self.sv_tree.column(col, width=w, anchor=anchor, stretch=stretch)
        self.sv_tree.tag_configure("fixed", foreground="#2e7d32")
        self.sv_tree.tag_configure("ok", foreground="#777777")
        sb = ttk.Scrollbar(left, command=self.sv_tree.yview)
        self.sv_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sv_tree.pack(side="left", fill="both", expand=True)
        self.sv_tree.bind("<<TreeviewSelect>>", self._sv_on_select)

        right = ttk.Frame(parent)
        right.grid(row=2, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.sv_title = ttk.Label(right, text="", font=("TkDefaultFont", 12, "bold"))
        self.sv_title.pack(anchor="w")
        self.sv_sub = ttk.Label(right, text="", foreground="#555", wraplength=540, justify="left")
        self.sv_sub.pack(anchor="w")
        nav = ttk.Frame(right)
        nav.pack(fill="x", pady=(2, 2))
        ttk.Button(nav, text="◀", width=3, command=lambda: self._sv_step(-1)).pack(side="left")
        ttk.Button(nav, text="▶", width=3, command=lambda: self._sv_step(1)).pack(side="left", padx=(2, 8))
        self.sv_occ_label = ttk.Label(nav, text="")
        self.sv_occ_label.pack(side="left")
        self.sv_ctx = tk.Text(right, height=5, wrap="word", state="disabled", relief="solid", borderwidth=1)
        self.sv_ctx.tag_configure("hit", background="#fff2a8")
        self.sv_ctx.pack(fill="x")

        listen = ttk.Frame(right)
        listen.pack(fill="x", pady=(6, 0))
        self.sv_chapter_btn = ttk.Button(listen, text="▶ Как в готовой главе", command=self._sv_play_chapter)
        self.sv_chapter_btn.pack(side="left")
        self.sv_asr_btn = ttk.Button(listen, text="Распознать главу", command=self._sv_run_asr)
        self.sv_asr_btn.pack(side="left", padx=6)
        ttk.Button(listen, text="▶ Как по текущим правилам", command=lambda: self._sv_play_form(None)).pack(side="left")

        ttk.Label(right, text="Как правильно? (▶ — синтез этого предложения голосом из вкладки «Озвучка»)").pack(anchor="w", pady=(8, 2))
        self.sv_cands = ttk.Frame(right)
        self.sv_cands.pack(fill="x")
        custom = ttk.Frame(right)
        custom.pack(fill="x", pady=(4, 0))
        ttk.Label(custom, text="Своё чтение (с «+»):").pack(side="left")
        ttk.Entry(custom, textvariable=self.sv_custom_var, width=22).pack(side="left", padx=6)
        ttk.Button(custom, text="▶", width=3, command=lambda: self._sv_play_form(self.sv_custom_var.get().strip().lower())).pack(side="left")

        scope = ttk.Frame(right)
        scope.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(scope, text="только это место (фраза в файле правок книги)", value="here",
                        variable=self.sv_scope_var).pack(anchor="w")
        ttk.Radiobutton(scope, text="везде, во всех книгах (словарь ударений)", value="all",
                        variable=self.sv_scope_var).pack(anchor="w")
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Подтвердить (оставить как есть)", command=self._sv_confirm).pack(side="left")
        ttk.Button(btns, text="Исправить", command=self._sv_fix).pack(side="left", padx=6)
        ttk.Button(btns, text="Пропустить", command=lambda: self._sv_next()).pack(side="left")
        llm = ttk.LabelFrame(right, text="Локальная LLM (необязательно)", padding=6)
        llm.pack(fill="x", pady=(8, 0))
        import stress_llm as _sl
        ttk.Combobox(llm, textvariable=self.sv_llm_backend_var, values=list(_sl.BACKENDS), state="readonly",
                     width=11).pack(side="left")
        ttk.Label(llm, text="Модель (LM Studio/Ollama):").pack(side="left", padx=(8, 2))
        ttk.Entry(llm, textvariable=self.sv_llm_model_var, width=26).pack(side="left")
        self.sv_llm_btn = ttk.Button(llm, text="Спросить LLM", command=self._sv_ask_llm)
        self.sv_llm_btn.pack(side="left", padx=8)
        ttk.Label(right, textvariable=self.sv_status_var, foreground="#555", wraplength=540,
                  justify="left").pack(side="bottom", anchor="w", pady=(6, 0))
        import gui_tooltips as _tt
        _tt.apply_tips(parent, _tt.REVIEW_TIPS)
        _tt.add_help_button(top, side="right", padx=(0, 8))
        _tt.add_tip(self.sv_tree, _tt.REVIEW_BLOCK_TIPS["tree"])
        _tt.add_tip(self.sv_ctx, _tt.REVIEW_BLOCK_TIPS["ctx"])
        _tt.add_tip(self.sv_cands, _tt.REVIEW_BLOCK_TIPS["cands"])
        _tt.tip_headings(self.sv_tree, _tt.REVIEW_HEADINGS)
        self._sv_poll()

    # ---- main-thread queue ----
    def _sv_poll(self):
        try:
            while True:
                self._sv_queue.get_nowait()()
        except queue.Empty:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.after(80, self._sv_poll)
        except Exception:
            pass

    def _sv_post(self, fn):
        self._sv_queue.put(fn)

    def _sv_running(self) -> bool:
        return bool(self._sv_thread and self._sv_thread.is_alive())

    # ---- search ----
    def _sv_scan_toggle(self):
        if self._sv_running():
            self._sv_cancel.set()
            self.sv_status_var.set("Останавливаю…")
            return
        chapters = list(getattr(self, "chapters", None) or [])
        if not chapters:
            messagebox.showwarning("Проверка ударений", "Сначала загрузите книгу на вкладке «Озвучка».")
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Проверка ударений", "Идёт озвучка: поиск использует те же модели ударений. Дождитесь её окончания.")
            return
        fn = getattr(self, "_se_scan_running", None)
        if fn and fn():
            messagebox.showinfo("Проверка ударений", "Идёт анализ омографов на вкладке «Словарь ударений»: дождитесь его окончания.")
            return
        try:
            only = _parse_chapters(self.sv_chapters_var.get(), len(chapters))
        except ValueError as e:
            messagebox.showwarning("Проверка ударений", str(e))
            return
        words = [w for w in re.split(r"[,;\s]+", self.sv_words_var.get()) if w]
        use_corpus = bool(self.sv_corpus_var.get())
        want_q = {k for k, v in (("name", self.sv_names_var), ("unknown", self.sv_terms_var),
                                 ("inconsistent", self.sv_unstable_var)) if v.get()}
        if not words and not use_corpus and not want_q:
            messagebox.showinfo("Проверка ударений", "Отметьте, что искать, или введите свои слова.")
            return
        skip = {k for k, v in self._sv_state.items() if v.get("status") in ("ok", "fixed")}
        try:
            self._se_flush_caches()
        except Exception:
            pass
        book_over = None
        try:
            book_over = self._se_book_overrides_path()
        except Exception:
            pass
        cancel = threading.Event()
        self._sv_cancel = cancel
        self.sv_scan_btn.configure(text="Остановить поиск")
        self.sv_progress.start(12)
        self.sv_status_var.set("Загружаю модели ударений…")

        def work():
            try:
                prog = lambda m: self._sv_post(lambda m=m: self.sv_status_var.set(m))
                ms = []
                if use_corpus or words:
                    ms += srv.scan_moments(chapters, only_chapters=only, manual_words=words, use_corpus=use_corpus, skip_keys=skip,
                                           book_overrides_path=book_over, cancel=cancel, progress_cb=prog)
                if want_q:
                    import stress_questions as sq
                    idxs = sorted(only) if only is not None else list(range(len(chapters)))
                    qs = sq.scan_chapters([chapters[i] for i in idxs], cancel=cancel, skip_words=sq.load_ok_words(), progress_cb=prog)
                    for q in qs:
                        if q["kind"] not in want_q:
                            continue
                        m = srv.from_question(q)
                        for o in m["occ"]:              # question contexts count from the first scanned chapter: back to book indexes
                            o["chapter"] = idxs[o["chapter"]] if 0 <= o["chapter"] < len(idxs) else o["chapter"]
                        m["chapters"] = sorted({o["chapter"] for o in m["occ"]})
                        if m["occ"]:
                            ms.append(m)
                for n, m in enumerate(ms):
                    m["id"] = n
                outcome = ("ok", ms)
            except InterruptedError:
                outcome = ("cancel", None)
            except Exception as e:
                outcome = ("error", f"{type(e).__name__}: {e}")
            finally:
                try:
                    import fb2_reader
                    fb2_reader.reset_stress_context()
                except Exception:
                    pass
            self._sv_post(lambda: self._sv_scan_done(*outcome))

        self._sv_thread = threading.Thread(target=work, daemon=True)
        self._sv_thread.start()

    def _sv_scan_done(self, kind, payload):
        self.sv_progress.stop()
        self.sv_scan_btn.configure(text="Найти сомнительные места")
        if kind == "cancel":
            self.sv_status_var.set("Поиск остановлен.")
            return
        if kind == "error":
            self.sv_status_var.set(f"Поиск не удался: {payload}")
            return
        self._sv_moments = payload
        self._sv_fill()
        self.sv_status_var.set(f"Найдено мест: {len(payload)}. Выберите слово слева. Расхождение с корпусом и «нестабильное» — повод послушать, не ошибка.")

    # ---- list ----
    def _sv_state_of(self, q):
        return self._sv_state.get(srv.moment_key(q["word"], q["reading"]), {}).get("status", q.get("state", ""))

    def _sv_fill(self):
        keep = self._sv_cur["id"] if self._sv_cur else None
        self.sv_tree.delete(*self.sv_tree.get_children())
        self._sv_rows = {}
        for q in self._sv_moments:
            st = self._sv_state_of(q)
            if self.sv_hide_done_var.get() and st:
                continue
            iid = self.sv_tree.insert("", "end", values=(q["word"], q["reading"], srv.KIND_TITLES.get(q["kind"], q["kind"]), q["count"]), tags=(st,) if st else ())
            self._sv_rows[iid] = q
            if keep == q["id"]:
                self.sv_tree.selection_set(iid)

    def _sv_on_select(self, _e=None):
        sel = self.sv_tree.selection()
        if sel:
            self._sv_show(self._sv_rows[sel[0]])

    def _sv_show(self, q):
        self._sv_cur = q
        self._sv_occ = 0
        self.sv_title.configure(text=f"{q['word']}  →  читается «{q['reading']}»")
        corp = ", ".join(q["corpus"]) or "слова нет в корпусе"
        forms = ", ".join(q["group"][1:6])
        warn = "  ВНИМАНИЕ: слово встречается с разными ударениями (возможно, омограф)." if q["kind"] == "inconsistent" else ""
        self.sv_sub.configure(text=f"{srv.KIND_TITLES.get(q['kind'], q['kind'])}; корпус: {corp}; вхождений: {q['count']}; "
                                   f"главы: {', '.join(map(str, q['chapters'][:12]))}" + (f"; формы: {forms}" if forms else "") + warn)
        for w in self.sv_cands.winfo_children():
            w.destroy()
        self.sv_choice_var.set("")
        self.sv_custom_var.set("")
        for form in q["candidates"]:
            row = ttk.Frame(self.sv_cands)
            row.pack(fill="x", pady=1)
            ttk.Radiobutton(row, text=form + ("   (сейчас)" if form == q["reading"] else ""), value=form,
                            variable=self.sv_choice_var).pack(side="left")
            ttk.Button(row, text="▶", width=3, command=lambda f=form: self._sv_play_form(f)).pack(side="left", padx=6)
        self._sv_show_occ()

    def _sv_step(self, d):
        q = self._sv_cur
        if q:
            self._sv_occ = (self._sv_occ + d) % len(q["occ"])
            self._sv_show_occ()

    def _sv_show_occ(self):
        q = self._sv_cur
        o = q["occ"][self._sv_occ]
        self.sv_occ_label.configure(text=f"вхождение {self._sv_occ + 1}/{len(q['occ'])}  (глава {o['chapter']})")
        self.sv_ctx.configure(state="normal")
        self.sv_ctx.delete("1.0", "end")
        pos, k = 0, 0
        s = o["sentence"]
        for m in srv.TOK.finditer(s):
            self.sv_ctx.insert("end", s[pos:m.start()])
            hit = srv.plain(m.group(0)) == o.get("word", q["word"])
            self.sv_ctx.insert("end", m.group(0), ("hit",) if hit and k == o["idx"] else ())
            if hit:
                k += 1
            pos = m.end()
        self.sv_ctx.insert("end", s[pos:])
        self.sv_ctx.configure(state="disabled")
        cached = self._sv_asr_for(o["chapter"]) is not None
        wav = srv.find_chapter_wav(self._sv_outdir(), o["chapter"])
        self.sv_chapter_btn.configure(state="normal" if (cached and wav) else "disabled")
        self.sv_asr_btn.configure(state="disabled" if (cached or not wav) else "normal")

    # ---- chapter audio ----
    def _sv_outdir(self) -> Path:
        return Path(self.outdir_var.get().strip() or "audiobook_output")

    def _sv_asr_for(self, chapter):
        wav = srv.find_chapter_wav(self._sv_outdir(), chapter)
        cache = self._sv_outdir() / "_asr" / f"ch{chapter}.json"
        if not wav or not cache.exists() or cache.stat().st_mtime < wav.stat().st_mtime:
            return None
        if chapter not in self._sv_asr_cache or self._sv_asr_cache[chapter][0] != cache.stat().st_mtime:
            words = srv.load_asr_cache(self._sv_outdir(), chapter)
            if words is None:
                return None
            self._sv_asr_cache[chapter] = (cache.stat().st_mtime, words)
        return self._sv_asr_cache[chapter][1]

    def _sv_play_chapter(self):
        q = self._sv_cur
        if not q:
            return
        o = q["occ"][self._sv_occ]
        words = self._sv_asr_for(o["chapter"])
        wav = srv.find_chapter_wav(self._sv_outdir(), o["chapter"])
        if words is None or wav is None:
            self.sv_status_var.set("Для этой главы нет готового аудио или распознавания: нажмите «Распознать главу».")
            return
        loc = srv.locate_in_asr(words, o["sentence"], o.get("word", q["word"]), o["idx"])
        if loc is None:
            self.sv_status_var.set("Не нашёл это место в распознанном тексте главы (ASR ошибся в окружающих словах).")
            return
        try:
            data = srv.cut_clip_bytes(wav, loc[0], loc[1])
        except Exception as e:
            self.sv_status_var.set(f"Не удалось вырезать клип: {e}")
            return
        self._sv_play_bytes(data, f"Готовая глава {o['chapter']}, ≈{loc[0]:.0f} с")

    def _sv_run_asr(self):
        q = self._sv_cur
        if not q:
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Проверка ударений", "Идёт озвучка: распознавание использует видеокарту, дождитесь её окончания.")
            return
        import fb2_reader
        root = Path(fb2_reader.__file__).resolve().parent
        py = root / ".venv_cosyvoice3" / "Scripts" / "python.exe"
        script = root / "tools" / "asr_cache_cuda.py"
        if not py.exists() or not script.exists():
            self.sv_status_var.set("Распознавание недоступно: нет .venv_cosyvoice3 или tools/asr_cache_cuda.py (ставит install.bat).")
            return
        o = q["occ"][self._sv_occ]
        ch = o["chapter"]
        self.sv_asr_btn.configure(state="disabled")
        self.sv_progress.start(12)
        self.sv_status_var.set(f"Распознаю главу {ch} (1–3 минуты на видеокарте)…")

        def work():
            try:
                r = subprocess.run([str(py), str(script), str(self._sv_outdir()), str(ch)], cwd=str(root), capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=1800)
                ok = r.returncode == 0
                msg = "Готово." if ok else f"Ошибка распознавания (код {r.returncode}): {(r.stderr or '')[-200:]}"
            except Exception as e:
                ok, msg = False, f"Ошибка распознавания: {e}"

            def done():
                self.sv_progress.stop()
                self.sv_status_var.set(msg)
                if self._sv_cur:
                    self._sv_show_occ()
            self._sv_post(done)

        threading.Thread(target=work, daemon=True).start()

    # ---- synthesis of variants ----
    def _sv_play_form(self, form):
        q = self._sv_cur
        if not q:
            return
        o = q["occ"][self._sv_occ]
        if form:
            if srv.plain(form.replace("+", "")) != q["word"]:
                self.sv_status_var.set("Чтение должно состоять из тех же букв, что и слово.")
                return
            f_occ = srv.form_for(q, o, form)
            if not f_occ:
                self.sv_status_var.set("На падежную форму этого вхождения выбранное ударение не переносится: послушайте другое вхождение.")
                return
            text = srv.render_occurrence(o["sentence"], o.get("word", q["word"]), f_occ, o["idx"])
        else:
            text = o["sentence"]
        url = (self.rest_url_var.get().strip() or "http://localhost:5010").rstrip("/")
        voice = self._sv_voice()
        self.sv_status_var.set(f"Синтезирую: {text}")

        def work():
            try:
                import requests
                resp = requests.get(url + "/getwav", params={
                    "text_to_speech": text, "speaker": voice, "sample_rate": 48000,
                    "strip_mono": "false", "server_accent": "false"}, timeout=120)
                resp.raise_for_status()
                data = resp.content
                self._sv_post(lambda: self._sv_play_bytes(data, "Синтез"))
            except Exception as e:
                self._sv_post(lambda e=e: self.sv_status_var.set(
                    f"Не удалось озвучить ({type(e).__name__}). Запущен ли сервис Silero по адресу {url}?"))

        threading.Thread(target=work, daemon=True).start()

    def _sv_voice(self) -> str:
        try:
            return self._silero_voice_keys[self._silero_voice_labels.index(self.voice_var.get())]
        except Exception:
            return "xenia"

    def _sv_play_bytes(self, data: bytes, what: str):
        try:
            import os
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
            path = os.path.join(tempfile.gettempdir(), "audiobook_stress_review.wav")
            with open(path, "wb") as f:
                f.write(data)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.sv_status_var.set(f"{what}: воспроизвожу…")
        except Exception as e:
            self.sv_status_var.set(f"Не удалось воспроизвести: {e}")

    # ---- decisions ----
    def _sv_save_state(self, q, status, extra=None):
        import datetime
        self._sv_state[srv.moment_key(q["word"], q["reading"])] = {
            "status": status, "date": datetime.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
        try:
            srv.save_state(self._sv_state)
        except Exception as e:
            self.sv_status_var.set(f"Не удалось сохранить состояние: {e}")

    def _sv_ask_llm(self):
        """Подсказка локальной LLM (LM Studio / Ollama): выбирает номер варианта; результат подставляется в форму, запись делает пользователь."""
        q = self._sv_cur
        if not q:
            return
        import stress_llm as sl
        backend = sl.BACKENDS.get(self.sv_llm_backend_var.get(), "lmstudio")
        model = self.sv_llm_model_var.get()
        host = ""
        try:
            host = self.ollama_host_var.get().strip()
        except Exception:
            pass
        self.sv_llm_btn.configure(state="disabled")
        self.sv_progress.start(12)
        self.sv_status_var.set("Спрашиваю локальную LLM… (первый вопрос: запуск и загрузка модели, до минуты)" if backend == "builtin" else "Спрашиваю локальную LLM…")

        def work():
            try:
                r = sl.judge_moment(q, backend, model, host)
                err = None
            except Exception as e:
                r, err = None, f"{type(e).__name__}: {e}"

            def done():
                self.sv_progress.stop()
                self.sv_llm_btn.configure(state="normal")
                if self._sv_cur is not q:
                    return
                if err:
                    self.sv_status_var.set(f"LLM недоступна: {err[:300]} ")
                elif not r["form"]:
                    self.sv_status_var.set("LLM не уверена" + (f": {r['reason']}" if r["reason"] else ".") + " Решайте на слух.")
                else:
                    self.sv_choice_var.set(r["form"])
                    same = " (совпадает с текущим чтением)" if r["form"] == q["reading"] else ""
                    self.sv_status_var.set(f"LLM предлагает «{r['form']}»{same}. {r['reason']} Это подсказка, проверьте на слух.")
            self._sv_post(done)

        threading.Thread(target=work, daemon=True).start()

    def _sv_confirm(self):
        q = self._sv_cur
        if not q:
            return
        if q.get("question"):                       # names / terms / unstable words: never ask about this word again
            import stress_questions as sq
            sq.add_ok_words(q["group"])
        self._sv_save_state(q, "ok")
        self.sv_status_var.set(f"«{q['word']}»: чтение «{q['reading']}» подтверждено, больше не показываю.")
        self._sv_fill()
        self._sv_next(after=q)

    def _sv_choice(self):
        q = self._sv_cur
        form = (self.sv_custom_var.get().strip() or self.sv_choice_var.get()).lower()
        if not form:
            return None, "Выберите вариант или введите своё чтение."
        letters = form.replace("+", "")
        if srv.plain(letters) != q["word"]:
            return None, "Чтение должно состоять из тех же букв, что и слово."
        if form.count("+") > 1 or ("+" not in form and "ё" not in form):
            return None, "Поставьте один «+» перед ударной гласной."
        return form, None

    def _sv_fix(self):
        q = self._sv_cur
        if not q:
            return
        form, err = self._sv_choice()
        if err:
            self.sv_status_var.set(err)
            return
        o = q["occ"][self._sv_occ]
        scope = self.sv_scope_var.get()
        if scope == "here":
            from gui_stress_editor import merge_phrases_into_file, validate_phrase
            path = None
            try:
                path = self._se_book_overrides_path()
            except Exception:
                pass
            if path is None:
                messagebox.showinfo("Проверка ударений", "Для записи «только здесь» загрузите книгу.")
                return
            f_occ = srv.form_for(q, o, form)
            if not f_occ:
                self.sv_status_var.set("На падежную форму этого вхождения выбранное ударение не переносится: исправьте другое вхождение.")
                return
            try:
                key, val = srv.phrase_for_occurrence(o["sentence"], o.get("word", q["word"]), f_occ, o["idx"])
            except ValueError as e:
                self.sv_status_var.set(str(e))
                return
            bad = validate_phrase(key, val)
            if bad:
                self.sv_status_var.set(f"Фраза не принята: {bad}")
                return
            try:
                merge_phrases_into_file(path, {key: val})
            except Exception as e:
                messagebox.showerror("Проверка ударений", f"Не удалось записать {path}:\n{e}")
                return
            what = f"фраза «{key}» → «{val}» ({Path(path).name})"
        else:
            if not messagebox.askyesno(
                    "Проверка ударений",
                    f"Слово «{q['word']}» будет читаться как «{form}» ВЕЗДЕ, во всех главах и во всех книгах.\n"
                    "Если у слова есть другие чтения (омограф), используйте «только это место». Продолжить?"):
                return
            import stress_questions as sq
            forms = sq.answer_forms(q.get("question") or {"group": [q["word"]]}, form) if len(q["group"]) > 1 else {q["word"]: form}
            for pl, fm in forms.items():
                prev = self._se_entries.get(pl)
                self._se_entries[pl] = fm
                sq.log_answer(pl, fm, prev, "stress check tab (user chose after listening)")
            self._se_mark_dirty()
            if not self._se_save():
                return
            self._se_fill_list()
            what = "словарь: " + ", ".join(forms.values())
        try:
            self._se_flush_caches()
        except Exception:
            pass
        self._sv_save_state(q, "fixed", {"to": form, "scope": scope})
        chs = ", ".join(map(str, q["chapters"][:15]))
        self.sv_status_var.set(f"Записано: {what}. Затронуты главы: {chs}. Пересинтез — «Re-run Synthesis» на вкладке «Словарь ударений».")
        try:
            self.log(f"Проверка ударений: {what}")
        except Exception:
            pass
        self._sv_fill()
        self._sv_next(after=q)

    def _sv_next(self, after=None):
        cur = after or self._sv_cur
        kids = self.sv_tree.get_children()
        if not kids:
            return
        order = [self._sv_rows[k] for k in kids]
        if cur in order:
            i = order.index(cur) + 1
        elif cur is not None:
            i = next((n for n, x in enumerate(order) if x["id"] > cur["id"]), len(order))
        else:
            i = 0
        if i < len(kids):
            self.sv_tree.selection_set(kids[i])
            self.sv_tree.see(kids[i])
