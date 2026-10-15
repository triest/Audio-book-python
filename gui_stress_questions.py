"""Вкладка «Вопросы по ударению»: сначала анализируется книга, затем пользователю выдаются имена, термины и
нестабильные слова, в которых нельзя быть уверенным; пользователь слушает варианты и выбирает верный.
Логика поиска вопросов — `stress_questions.scan_chapters` (без Tk, тестируется отдельно). Ответы пишутся в тот же
словарь, что и вкладка «Словарь ударений» (`self._se_entries`), поэтому два экрана не расходятся."""
import json
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

KIND_TITLES = {"name": "имя", "unknown": "термин", "inconsistent": "нестабильное"}


class StressQuestionsMixin:

    def _build_stress_questions_tab(self, parent):
        self._sq_questions: list = []
        self._sq_rows: dict = {}                 # tree iid -> question
        self._sq_current = None
        self._sq_thread = None
        self._sq_cancel = None
        self._sq_queue: "queue.Queue" = queue.Queue()
        self.sq_status_var = tk.StringVar(value="Загрузите книгу на вкладке «Озвучка», затем нажмите «Проанализировать книгу».")
        self.sq_choice_var = tk.StringVar(value="")
        self.sq_names_var = tk.BooleanVar(value=True)
        self.sq_terms_var = tk.BooleanVar(value=True)
        self.sq_unstable_var = tk.BooleanVar(value=False)
        self.sq_hide_done_var = tk.BooleanVar(value=True)

        parent.columnconfigure(0, weight=2, uniform="sq")
        parent.columnconfigure(1, weight=3, uniform="sq")
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        self.sq_scan_btn = ttk.Button(top, text="Проанализировать книгу", command=self._sq_scan_toggle)
        self.sq_scan_btn.pack(side="left")
        ttk.Checkbutton(top, text="имена", variable=self.sq_names_var, command=self._sq_fill).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(top, text="термины", variable=self.sq_terms_var, command=self._sq_fill).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(top, text="нестабильные (возможные омографы)", variable=self.sq_unstable_var,
                        command=self._sq_fill).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(top, text="скрыть отвеченные", variable=self.sq_hide_done_var, command=self._sq_fill).pack(side="left", padx=(8, 0))
        self.sq_progress = ttk.Progressbar(top, mode="indeterminate", length=90)
        self.sq_progress.pack(side="right")

        left = ttk.Frame(parent)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.sq_tree = ttk.Treeview(left, columns=("word", "kind", "count"), show="headings", selectmode="browse")
        for col, text, w, anchor, stretch in (("word", "Слово", 170, "w", True), ("kind", "Тип", 90, "w", False),
                                              ("count", "Раз", 50, "center", False)):
            self.sq_tree.heading(col, text=text)
            self.sq_tree.column(col, width=w, anchor=anchor, stretch=stretch)
        self.sq_tree.tag_configure("done", foreground="#2e7d32")
        self.sq_tree.tag_configure("model", foreground="#777777")
        sb = ttk.Scrollbar(left, command=self.sq_tree.yview)
        self.sq_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sq_tree.pack(side="left", fill="both", expand=True)
        self.sq_tree.bind("<<TreeviewSelect>>", self._sq_on_select)

        right = ttk.Frame(parent)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.sq_title = ttk.Label(right, text="", font=("TkDefaultFont", 12, "bold"))
        self.sq_title.pack(anchor="w")
        self.sq_sub = ttk.Label(right, text="", foreground="#555", wraplength=520, justify="left")
        self.sq_sub.pack(anchor="w", pady=(0, 4))
        self.sq_ctx = tk.Text(right, height=7, wrap="word", state="disabled", relief="solid", borderwidth=1)
        self.sq_ctx.tag_configure("hit", background="#fff2a8")
        self.sq_ctx.pack(fill="x")
        ttk.Label(right, text="Как правильно читается? (▶ — прослушать голосом из вкладки «Озвучка»)").pack(anchor="w", pady=(8, 2))
        self.sq_cands = ttk.Frame(right)
        self.sq_cands.pack(fill="x")
        self.sq_warn = ttk.Label(right, text="", foreground="#a00", wraplength=520, justify="left")
        self.sq_warn.pack(anchor="w", pady=(4, 0))
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(8, 0))
        self.sq_save_btn = ttk.Button(btns, text="Сохранить ответ", command=self._sq_save_answer)
        self.sq_save_btn.pack(side="left")
        ttk.Button(btns, text="Пропустить", command=lambda: self._sq_next()).pack(side="left", padx=6)
        ttk.Label(right, textvariable=self.sq_status_var, foreground="#555", wraplength=520,
                  justify="left").pack(side="bottom", anchor="w", pady=(6, 0))
        self._sq_poll()

    # ---- worker threads post callables; the main thread runs them ----
    def _sq_poll(self):
        try:
            while True:
                self._sq_queue.get_nowait()()
        except queue.Empty:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.after(80, self._sq_poll)
        except Exception:
            pass

    def _sq_post(self, fn):
        self._sq_queue.put(fn)

    # ---- analysis ----
    def _sq_running(self) -> bool:
        return bool(self._sq_thread and self._sq_thread.is_alive())

    def _sq_scan_toggle(self):
        if self._sq_running():
            self._sq_cancel.set()
            self.sq_status_var.set("Останавливаю…")
            return
        chapters = list(getattr(self, "chapters", None) or [])
        if not chapters:
            messagebox.showwarning("Вопросы по ударению", "Сначала загрузите книгу на вкладке «Озвучка».")
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Вопросы по ударению", "Идёт озвучка: анализ использует те же модели ударений. Дождитесь её окончания.")
            return
        if getattr(self, "_se_scan_running", None) and self._se_scan_running():
            messagebox.showinfo("Вопросы по ударению", "Идёт анализ омографов на вкладке «Словарь ударений».")
            return
        try:
            self._se_flush_caches()
        except Exception:
            pass
        cancel = threading.Event()
        self._sq_cancel = cancel
        self.sq_scan_btn.configure(text="Остановить анализ")
        self.sq_progress.start(12)
        self.sq_status_var.set("Загружаю модели ударений…")

        def work():
            try:
                import stress_questions as sq
                qs = sq.scan_chapters(chapters, cancel=cancel, skip_words=sq.load_ok_words(),
                                      progress_cb=lambda m: self._sq_post(lambda m=m: self.sq_status_var.set(m)))
                outcome = ("ok", qs)
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
            self._sq_post(lambda: self._sq_scan_done(*outcome))

        self._sq_thread = threading.Thread(target=work, daemon=True)
        self._sq_thread.start()

    def _sq_scan_done(self, kind, payload):
        self.sq_progress.stop()
        self.sq_scan_btn.configure(text="Проанализировать книгу")
        if kind == "cancel":
            self.sq_status_var.set("Анализ остановлен.")
            return
        if kind == "error":
            self.sq_status_var.set(f"Анализ не удался: {payload}")
            return
        self._sq_questions = payload
        for q in self._sq_questions:
            q["state"] = ""
        self._sq_fill()
        by = {}
        for q in payload:
            by[q["kind"]] = by.get(q["kind"], 0) + 1
        self.sq_status_var.set(
            f"Готово. Имён: {by.get('name', 0)}, терминов: {by.get('unknown', 0)}, нестабильных: {by.get('inconsistent', 0)}. "
            "Выберите слово слева.")

    # ---- list ----
    def _sq_visible(self, q) -> bool:
        if q["kind"] == "name" and not self.sq_names_var.get():
            return False
        if q["kind"] == "unknown" and not self.sq_terms_var.get():
            return False
        if q["kind"] == "inconsistent" and not self.sq_unstable_var.get():
            return False
        return not (self.sq_hide_done_var.get() and q.get("state"))

    def _sq_fill(self):
        keep = self._sq_current["id"] if self._sq_current else None
        self.sq_tree.delete(*self.sq_tree.get_children())
        self._sq_rows = {}
        for q in self._sq_questions:
            if not self._sq_visible(q):
                continue
            tag = {"saved": "done", "model": "model"}.get(q.get("state"), "")
            iid = self.sq_tree.insert("", "end", values=(q["word"], KIND_TITLES.get(q["kind"], q["kind"]), q["count"]),
                                      tags=(tag,) if tag else ())
            self._sq_rows[iid] = q
            if keep == q["id"]:
                self.sq_tree.selection_set(iid)

    def _sq_on_select(self, _e=None):
        sel = self.sq_tree.selection()
        if not sel:
            return
        self._sq_show(self._sq_rows[sel[0]])

    def _sq_show(self, q):
        self._sq_current = q
        self.sq_title.configure(text=q["word"])
        extra = ", ".join(q["group"][1:6])
        self.sq_sub.configure(text=f"{KIND_TITLES.get(q['kind'], q['kind'])}; встречается {q['count']} раз"
                                   + (f"; формы: {extra}" if extra else "")
                                   + f"; сейчас читается как: {q.get('model_form', '')}")
        self.sq_ctx.configure(state="normal")
        self.sq_ctx.delete("1.0", "end")
        for c in q["contexts"]:
            line = f"[гл. {c['chapter']}] "
            self.sq_ctx.insert("end", line)
            self._sq_insert_marked(c["sentence"], q["group"])
            self.sq_ctx.insert("end", chr(10) + chr(10))
        self.sq_ctx.configure(state="disabled")
        for w in self.sq_cands.winfo_children():
            w.destroy()
        self.sq_choice_var.set("")
        for form in q["candidates"]:
            row = ttk.Frame(self.sq_cands)
            row.pack(fill="x", pady=1)
            ttk.Radiobutton(row, text=form, value=form, variable=self.sq_choice_var).pack(side="left")
            ttk.Button(row, text="▶", width=3, command=lambda f=form: self._sq_play(f)).pack(side="left", padx=6)
        ttk.Radiobutton(self.sq_cands, text="модель права — оставить как есть и больше не спрашивать", value="__model__",
                        variable=self.sq_choice_var).pack(anchor="w", pady=(4, 0))
        self.sq_warn.configure(text=(
            "Слово встречается с разными ударениями: возможно, это омограф. Запись в словарь применится ко ВСЕМ вхождениям; "
            "отвечайте, только если слово точно читается одним способом." if q["kind"] == "inconsistent" else ""))

    def _sq_insert_marked(self, sentence, group):
        """inserts the sentence, highlighting the words of the question group (marks '+' are hidden for reading)."""
        import re
        targets = set(group)
        pos = 0
        for m in re.finditer(r"[А-Яа-яЁё+]+", sentence):
            self.sq_ctx.insert("end", sentence[pos:m.start()])
            w = m.group(0)
            plain = w.replace("+", "").lower().replace("ё", "е")
            self.sq_ctx.insert("end", w, ("hit",) if plain in targets else ())
            pos = m.end()
        self.sq_ctx.insert("end", sentence[pos:])

    # ---- listening ----
    def _sq_voice(self) -> str:
        try:
            return self._silero_voice_keys[self._silero_voice_labels.index(self.voice_var.get())]
        except Exception:
            return "xenia"

    def _sq_play(self, form):
        q = self._sq_current
        if not q:
            return
        import stress_questions as sq
        sent = q["contexts"][0]["sentence"] if q["contexts"] else q["word"]
        text = sq.render_sentence(sent, q["word"], form)
        url = (self.rest_url_var.get().strip() or "http://localhost:5010").rstrip("/")
        voice = self._sq_voice()
        self.sq_status_var.set(f"Синтезирую: {text}")

        def work():
            try:
                import requests
                resp = requests.get(url + "/getwav", params={
                    "text_to_speech": text, "speaker": voice, "sample_rate": 48000,
                    "strip_mono": "false", "server_accent": "false"}, timeout=120)
                resp.raise_for_status()
                data = resp.content
                self._sq_post(lambda: self._sq_play_bytes(data))
            except Exception as e:
                self._sq_post(lambda e=e: self.sq_status_var.set(
                    f"Не удалось озвучить ({type(e).__name__}). Запущен ли сервис Silero по адресу {url}? "
                    "Он поднимается при запуске озвучки или в настройках вкладки «Озвучка»."))

        threading.Thread(target=work, daemon=True).start()

    def _sq_play_bytes(self, data: bytes):
        try:
            import os
            import tempfile
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
            path = os.path.join(tempfile.gettempdir(), "audiobook_stress_question.wav")   # SND_ASYNC cannot play from memory
            with open(path, "wb") as f:
                f.write(data)
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.sq_status_var.set("Воспроизвожу…")
        except Exception as e:
            self.sq_status_var.set(f"Не удалось воспроизвести: {e}")

    # ---- answers ----
    def _sq_save_answer(self):
        q = self._sq_current
        if not q:
            return
        choice = self.sq_choice_var.get()
        if not choice:
            self.sq_status_var.set("Выберите вариант или нажмите «Пропустить».")
            return
        import stress_questions as sq
        if choice == "__model__":
            sq.add_ok_words(q["group"])
            q["state"] = "model"
            self.sq_status_var.set(f"«{q['word']}»: оставлено как есть, больше не спрашиваю.")
        else:
            if q["kind"] == "inconsistent" and not messagebox.askyesno(
                    "Вопросы по ударению",
                    "Слово встречается с разными ударениями. Запись применится ко всем вхождениям во всех книгах. Продолжить?"):
                return
            forms = sq.answer_forms(q, choice)
            for pl, form in forms.items():
                prev = self._se_entries.get(pl)
                self._se_entries[pl] = form
                sq.log_answer(pl, form, prev, "stress_questions tab (user chose after listening)")
            self._se_mark_dirty()
            if not self._se_save():
                return
            self._se_fill_list()
            try:
                self._se_flush_caches()
            except Exception:
                pass
            q["state"] = "saved"
            self.sq_status_var.set(f"Записано в словарь: {', '.join(forms.values())}. Главы со словом озвучатся заново при следующем запуске.")
        self._sq_fill()
        self._sq_next(after=q)

    def _sq_next(self, after=None):
        cur = after or self._sq_current
        kids = self.sq_tree.get_children()
        if not kids:
            return
        order = [self._sq_rows[k] for k in kids]
        if cur in order:
            i = order.index(cur) + 1
        elif cur is not None:
            ids = [q["id"] for q in order]
            i = next((n for n, x in enumerate(ids) if x > cur["id"]), len(order))
        else:
            i = 0
        if i < len(kids):
            self.sq_tree.selection_set(kids[i])
            self.sq_tree.see(kids[i])
