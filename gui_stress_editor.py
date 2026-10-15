"""Stress Dictionary Editor: the «Словарь ударений» tab of the main window.

What it does
  * shows the active book chapter and highlights every occurrence of the searched word;
  * edits forced_stress_words.json visually: the word is drawn letter by letter, the vowels are
    clickable, a click puts (or moves) the "+" right before that vowel, a second click removes it;
  * homograph alert: for a word whose stress depends on context (все, земли, сети ...) it lists the
    different readings the pipeline gives to the occurrences found in the chapter;
  * «Анализ омографов книги»: a worker thread scans the loaded book with the default stress pipeline
    (fb2_reader.analyze_chapters_homographs) and fills the «Омографы книги» list with phrase keys
    ("мы все побежали" -> "мы вс+е побежали"); a click on a phrase opens it in the visual editor (every
    word of the phrase has clickable vowels), «Применить правку» fixes a neural slip, «Сохранить для книги»
    merges the list into <book>.manual_stress_overrides.json (or «Открыть JSON…» loads any phrase file);
  * «Сохранить словарь» writes forced_stress_words.json immediately (atomic), «Re-run Synthesis»
    saves, flushes every in-memory stress cache and re-synthesizes the selected chapters.

Layout of the file: pure helpers first (no Tk, unit-tested, work the same frozen or not), then
StressEditorMixin, which AudiobookApp mixes in. Paths come from fb2_reader.DEFAULT_FORCED_STRESS_PATH
(app_paths.user_file), so the editor edits the file next to the .exe when frozen.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
from pathlib import Path

try:                                # the pure helpers below stay importable (and testable) without Tk
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:                 # pragma: no cover - headless Python without tkinter
    tk = filedialog = messagebox = ttk = None

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
_WORD_AT_RE = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def plain_word(value: str) -> str:
    """Word without the "+" mark."""
    return (value or "").replace("+", "")


def vowel_positions(word: str) -> "list[int]":
    """Indices (in the word WITHOUT "+") of its vowels."""
    return [i for i, ch in enumerate(plain_word(word)) if ch in VOWELS]


def stress_position(value: str) -> "int | None":
    """Index (in the plain word) of the vowel that carries the "+", or None."""
    pos = 0
    for ch in value or "":
        if ch == "+":
            return pos
        pos += 1
    return None


def place_stress(value: str, vowel_index: int) -> str:
    """The word with its "+" put right before the plain letter `vowel_index` (the old mark is
    moved). Clicking the vowel that already carries the mark removes it. A non-vowel index
    changes nothing."""
    plain = plain_word(value)
    if not (0 <= vowel_index < len(plain)) or plain[vowel_index] not in VOWELS:
        return value
    if stress_position(value) == vowel_index:
        return plain
    return plain[:vowel_index] + "+" + plain[vowel_index:]


def stress_positions(value: str) -> "list[int]":
    """Plain-text indices of every "+" in `value` (a word or a whole phrase)."""
    out, pos = [], 0
    for ch in value or "":
        if ch == "+":
            out.append(pos)
        else:
            pos += 1
    return out


def place_stress_in_phrase(value: str, plain_index: int) -> str:
    """place_stress for a phrase: `plain_index` counts letters AND spaces of the phrase without marks;
    only the word that owns the clicked vowel changes (its own "+" moves / is removed)."""
    words = (value or "").split(" ")
    start = 0
    for i, w in enumerate(words):
        n = len(plain_word(w))
        if start <= plain_index < start + n:
            words[i] = place_stress(w, plain_index - start)
            break
        start += n + 1
    return " ".join(words)


def validate_phrase(key: str, value: str) -> "str | None":
    """None when a phrase entry is usable (same words as the key, only "+" differs, at most one mark per
    word, at least one mark overall), otherwise a short Russian error text."""
    kw, vw = (key or "").split(), (value or "").split()
    if len(kw) < 2:
        return "Фраза должна состоять минимум из двух слов."
    if len(kw) != len(vw):
        return "В записи должно быть столько же слов, сколько в ключе."
    for a, b in zip(kw, vw):
        if plain_word(b).lower().replace("ё", "е") != a.lower().replace("ё", "е"):
            return "Буквы записи должны совпадать с ключом: меняется только «+»."
        if b.count("+") > 1:
            return "В одном слове может быть только одно ударение."
    if "+" not in value:
        return "Поставьте хотя бы одно ударение: нажмите на ударную гласную."
    return None


def merge_phrases_into_file(path: "Path | str", phrases: dict) -> int:
    """Adds/overwrites the phrase entries {key: value} in a JSON dictionary file (atomic, other entries and
    "_" comment keys untouched). Returns the number of phrases written."""
    entries, extras = load_entries(path)
    entries.update({k: v for k, v in phrases.items()})
    save_entries(path, entries, extras)
    return len(phrases)


def validate_entry(key: str, value: str) -> "str | None":
    """None when the entry is usable, otherwise a short Russian error text."""
    k = (key or "").strip().lower()
    if not k or not _LETTER_RE.search(k):
        return "Введите слово."
    if value.count("+") != 1:
        return "Поставьте ровно одно ударение: нажмите на ударную гласную."
    if plain_word(value).lower().replace("ё", "е") != k.replace("ё", "е"):
        return "Запись не совпадает со словом: буквы должны быть те же, меняется только «+»."
    return None


def search_pattern(query: str, whole_word: bool = True) -> "re.Pattern | None":
    """Regex for the search bar: case-insensitive, е == ё, an optional "+" anywhere inside the word
    (so it also finds the word in the text with stress marks)."""
    q = (query or "").replace("+", "").strip()
    if not q:
        return None
    words = []
    for w in q.split():                      # a phrase: any whitespace between its words
        parts = []
        for ch in w:
            low = ch.lower()
            parts.append("[её]" if low in "её" else re.escape(ch))
        words.append(r"\+?".join(parts))
    body = r"\s+\+?".join(words)
    if whole_word:
        return re.compile(r"(?<![А-Яа-яЁё])\+?" + body + r"(?![А-Яа-яЁё])(?!-[А-Яа-яЁё])", re.I)
    return re.compile(r"\+?" + body, re.I)


def load_entries(path: "Path | str") -> "tuple[dict, dict]":
    """(entries, extras) of forced_stress_words.json. `extras` are the keys starting with "_"
    (comments), kept untouched on save. Bad/missing file -> empty."""
    p = Path(path)
    entries, extras = {}, {}
    try:
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8") or "{}")
            for k, v in raw.items():
                (extras if str(k).startswith("_") else entries)[str(k)] = v if str(k).startswith("_") else str(v)
    except Exception:
        pass
    return entries, extras


def save_entries(path: "Path | str", entries: dict, extras: "dict | None" = None) -> None:
    """Atomic write (tmp file + os.replace): a crash can never leave a half-written dictionary."""
    p = Path(path)
    data = dict(extras or {})
    data.update({k: entries[k] for k in entries})
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def group_variants(variants: "list[dict]") -> "list[dict]":
    """[{"form","count","example","positions"}] sorted by frequency, from stress_word_variants()."""
    groups: dict = {}
    for v in variants:
        g = groups.setdefault(v["form"], {"form": v["form"], "count": 0, "example": v["context"],
                                           "positions": []})
        g["count"] += 1
        g["positions"].append(v["pos"])
    return sorted(groups.values(), key=lambda g: (-g["count"], g["form"]))


def stale_meta_files(outdir: "Path | str", chapters, indices) -> "list[Path]":
    """Existing «.meta.json» fingerprints of the given chapters (1-based `indices`). Deleting them
    makes the next run synthesize those chapters again even if nothing else changed."""
    from fb2_reader import sanitize_filename
    out = []
    for idx in indices:
        if 1 <= idx <= len(chapters):
            title = chapters[idx - 1][0]
            m = Path(outdir) / f"{idx:03d}_{sanitize_filename(title)}.wav.meta.json"
            if m.exists():
                out.append(m)
    return out


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class StressEditorMixin:
    """Adds the «Словарь ударений» tab to AudiobookApp (see UIBuilderMixin._build_ui)."""

    # colours (light UI)
    _SE_HIT = "#fff2a8"
    _SE_CUR = "#ffb347"
    _SE_MARK = "#ffe08a"

    def _build_stress_editor_tab(self, parent):
        import fb2_reader
        self._se_path = Path(fb2_reader.ensure_forced_stress_file())
        self._se_entries, self._se_extras = load_entries(self._se_path)
        self._se_dirty = False
        self._se_chapter_idx = None
        self._se_book_key = None
        self._se_raw_text = ""
        self._se_marked_text = None
        self._se_hits: list = []
        self._se_hit_i = -1
        self._se_value = ""
        self._se_job = 0
        self._se_groups: list = []
        self._se_queue: "queue.Queue" = queue.Queue()
        self._se_phr: dict = {}                  # phrase key -> {"value","orig","count","chapter"}
        self._se_phr_dirty = False
        self._se_phrase_key = None               # phrase currently loaded in the visual editor
        self._se_scan_thread = None
        self._se_scan_cancel = None

        self.se_search_var = tk.StringVar()
        self.se_whole_var = tk.BooleanVar(value=True)
        self.se_view_var = tk.StringVar(value="text")
        self.se_word_var = tk.StringVar()
        self.se_filter_var = tk.StringVar()
        self.se_count_var = tk.StringVar(value="")
        self.se_status_var = tk.StringVar(value="")
        self.se_value_var = tk.StringVar(value="")
        self.se_alert_var = tk.StringVar(value="")
        self.se_chapter_var = tk.StringVar()
        self.se_phr_filter_var = tk.StringVar()
        self.se_phr_multi_var = tk.BooleanVar(value=True)
        self.se_scan_var = tk.StringVar(value="")
        self.se_phr_status_var = tk.StringVar(value="")

        # two grid columns (a ttk.PanedWindow can collapse the left pane at first layout)
        parent.columnconfigure(0, weight=3, uniform="se")
        parent.columnconfigure(1, weight=2, uniform="se")
        parent.rowconfigure(0, weight=1)
        left = ttk.Frame(parent)
        right = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        # ---- left: chapter text + search ----
        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Label(top, text="Глава:").pack(side="left")
        self.se_chapter_combo = ttk.Combobox(top, textvariable=self.se_chapter_var, state="readonly", width=44)
        self.se_chapter_combo.pack(side="left", padx=6, fill="x", expand=True)
        self.se_chapter_combo.bind("<<ComboboxSelected>>", lambda e: self._se_show_chapter(self._se_combo_index()))
        vrow2 = ttk.Frame(left)
        vrow2.pack(fill="x", pady=(4, 0))
        ttk.Label(vrow2, text="Показывать:").pack(side="left")
        ttk.Radiobutton(vrow2, text="исходный текст", value="text", variable=self.se_view_var,
                        command=self._se_change_view).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(vrow2, text="с ударениями (предпросмотр)", value="marked", variable=self.se_view_var,
                        command=self._se_change_view).pack(side="left", padx=(6, 0))

        srow = ttk.Frame(left)
        srow.pack(fill="x", pady=(6, 4))
        ttk.Label(srow, text="Поиск слова:").pack(side="left")
        ent = ttk.Entry(srow, textvariable=self.se_search_var, width=24)
        ent.pack(side="left", padx=6)
        ent.bind("<Return>", lambda e: self._se_step(+1))
        ent.bind("<Shift-Return>", lambda e: self._se_step(-1))
        self.se_search_var.trace_add("write", lambda *_: self._se_search())
        ttk.Button(srow, text="◀", width=3, command=lambda: self._se_step(-1)).pack(side="left")
        ttk.Button(srow, text="▶", width=3, command=lambda: self._se_step(+1)).pack(side="left", padx=(2, 8))
        ttk.Checkbutton(srow, text="слово целиком", variable=self.se_whole_var,
                        command=self._se_search).pack(side="left")
        ttk.Label(srow, textvariable=self.se_count_var, foreground="#555").pack(side="left", padx=10)
        ttk.Button(srow, text="Править", width=9, command=self._se_edit_searched).pack(side="right")

        tframe = ttk.Frame(left)
        tframe.pack(fill="both", expand=True)
        self.se_text = tk.Text(tframe, wrap="word", font=("Consolas", 11), undo=False, width=50, height=24)
        sb = ttk.Scrollbar(tframe, command=self.se_text.yview)
        self.se_text.configure(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.se_text.pack(side="left", fill="both", expand=True)
        self.se_text.tag_configure("mark", background=self._SE_MARK, foreground="#7a1f00")
        self.se_text.tag_configure("hit", background=self._SE_HIT)
        self.se_text.tag_configure("cur", background=self._SE_CUR)
        self.se_text.tag_raise("hit")
        self.se_text.tag_raise("cur")
        self.se_text.bind("<Double-Button-1>", self._se_on_text_double_click)
        ttk.Label(left, text="Двойной щелчок по слову в тексте загружает его в редактор справа.",
                  foreground="#777").pack(anchor="w", pady=(4, 0))

        # ---- right: visual editor ----
        box = ttk.LabelFrame(right, text="Редактор ударения", padding=8)
        box.pack(fill="x")
        wrow = ttk.Frame(box)
        wrow.pack(fill="x")
        ttk.Label(wrow, text="Слово:").pack(side="left")
        wentry = ttk.Entry(wrow, textvariable=self.se_word_var, width=26)
        wentry.pack(side="left", padx=6)
        wentry.bind("<Return>", lambda e: self._se_load_word(self.se_word_var.get()))
        ttk.Button(wrow, text="Загрузить", command=lambda: self._se_load_word(self.se_word_var.get())).pack(side="left")

        self.se_glyphs = tk.Text(box, height=1, font=("Consolas", 28), wrap="none", cursor="arrow",
                                 state="disabled", relief="flat", background=self._se_bg())
        self.se_glyphs.pack(fill="x", pady=(8, 2))
        self.se_glyphs.tag_configure("cons", foreground="#8a8a8a")
        self.se_glyphs.tag_configure("vowel", foreground="#0b4f9e", background="#dce9f8")
        self.se_glyphs.tag_configure("stressed", foreground="#9b0000", background="#f7c6c6")
        self.se_glyphs.tag_configure("plus", foreground="#d00000", font=("Consolas", 28, "bold"))
        ttk.Label(box, text="Нажмите на гласную — «+» встанет перед ней. Повторный щелчок снимает ударение.",
                  foreground="#666", wraplength=380, justify="left").pack(anchor="w")
        vrow = ttk.Frame(box)
        vrow.pack(fill="x", pady=(6, 0))
        ttk.Label(vrow, text="Запись:").pack(side="left")
        ttk.Label(vrow, textvariable=self.se_value_var, font=("Consolas", 12, "bold")).pack(side="left", padx=6)
        brow = ttk.Frame(box)
        brow.pack(fill="x", pady=(6, 0))
        ttk.Button(brow, text="Добавить / обновить", command=self._se_apply_entry).pack(side="left")
        ttk.Button(brow, text="Удалить из словаря", command=self._se_delete_entry).pack(side="left", padx=6)
        brow2 = ttk.Frame(box)
        brow2.pack(fill="x", pady=(6, 0))
        self.se_scan_btn = ttk.Button(brow2, text="Анализ омографов книги", command=self._se_scan_toggle)
        self.se_scan_btn.pack(side="left")

        # homograph alert
        self.se_alert_frame = ttk.LabelFrame(right, text="Омографы в главе", padding=8)
        self.se_alert_frame.pack(fill="x", pady=(8, 0))
        self.se_alert_label = tk.Label(self.se_alert_frame, textvariable=self.se_alert_var, justify="left",
                                       anchor="w", wraplength=380, fg="#444")
        self.se_alert_label.pack(fill="x")
        cols = ("form", "count", "example")
        self.se_variants = ttk.Treeview(self.se_alert_frame, columns=cols, show="headings", height=2,
                                        selectmode="browse")
        self.se_variants.heading("form", text="Чтение")
        self.se_variants.heading("count", text="Раз")
        self.se_variants.heading("example", text="Пример")
        self.se_variants.column("form", width=100, anchor="w", stretch=False)
        self.se_variants.column("count", width=40, anchor="center", stretch=False)
        self.se_variants.column("example", width=260, anchor="w")
        self.se_variants.pack(fill="x", pady=(4, 0))
        self.se_variants.bind("<<TreeviewSelect>>", self._se_on_variant_select)

        # lower right: two pages - the dictionary file and the book's homograph phrases
        self.se_right_nb = ttk.Notebook(right)
        self.se_right_nb.pack(fill="both", expand=True, pady=(8, 0))
        dbox = ttk.Frame(self.se_right_nb, padding=8)
        self.se_right_nb.add(dbox, text="Словарь: forced_stress_words.json")
        hbox = ttk.Frame(self.se_right_nb, padding=8)
        self.se_right_nb.add(hbox, text="Омографы книги")
        frow = ttk.Frame(dbox)
        frow.pack(fill="x")
        ttk.Label(frow, text="Фильтр:").pack(side="left")
        fe = ttk.Entry(frow, textvariable=self.se_filter_var, width=20)
        fe.pack(side="left", padx=6)
        self.se_filter_var.trace_add("write", lambda *_: self._se_fill_list())
        ttk.Label(dbox, textvariable=self.se_status_var, foreground="#555", wraplength=430,
                  justify="left").pack(side="bottom", anchor="w", pady=(4, 0))

        arow = ttk.Frame(dbox)
        arow.pack(side="bottom", fill="x")
        self.se_save_btn = ttk.Button(arow, text="Сохранить словарь", command=self._se_save)
        self.se_save_btn.pack(side="left")
        ttk.Button(arow, text="С диска", command=self._se_reload).pack(side="left", padx=6)
        ttk.Button(arow, text="Предпросмотр", command=self._se_refresh_preview).pack(side="left")
        arow2 = ttk.Frame(dbox)
        arow2.pack(side="bottom", fill="x", pady=(6, 0))
        self.se_rerun_btn = ttk.Button(arow2, text="Re-run Synthesis (выбранные главы)",
                                       command=self._se_rerun_synthesis)
        self.se_rerun_btn.pack(side="left")
        lframe = ttk.Frame(dbox)
        lframe.pack(fill="both", expand=True, pady=(4, 4))
        self.se_list = ttk.Treeview(lframe, columns=("word", "stress"), show="headings", height=5,
                                    selectmode="browse")
        self.se_list.heading("word", text="Слово")
        self.se_list.heading("stress", text="Ударение")
        self.se_list.column("word", width=170, anchor="w", stretch=False)
        self.se_list.column("stress", width=200, anchor="w")
        lsb = ttk.Scrollbar(lframe, command=self.se_list.yview)
        self.se_list.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self.se_list.pack(side="left", fill="both", expand=True)
        self.se_list.bind("<<TreeviewSelect>>", self._se_on_list_select)
        self._build_homograph_page(hbox)
        self._se_fill_list()
        self._se_render_word("")
        self._se_poll()
        self.notebook.bind("<<NotebookTabChanged>>", self._se_on_tab_changed, add="+")

    def _build_homograph_page(self, hbox):
        top = ttk.Frame(hbox)
        top.pack(fill="x")
        ttk.Checkbutton(top, text="только слова с ≥2 чтениями в книге", variable=self.se_phr_multi_var).pack(side="left")
        self.se_progress = ttk.Progressbar(top, mode="indeterminate", length=90)
        self.se_progress.pack(side="right")
        ttk.Label(hbox, textvariable=self.se_scan_var, foreground="#555", wraplength=430,
                  justify="left").pack(fill="x", pady=(2, 0))
        frow = ttk.Frame(hbox)
        frow.pack(fill="x", pady=(4, 0))
        ttk.Label(frow, text="Фильтр:").pack(side="left")
        ttk.Entry(frow, textvariable=self.se_phr_filter_var, width=24).pack(side="left", padx=6)
        self.se_phr_filter_var.trace_add("write", lambda *_: self._se_fill_phrases())

        bot = ttk.Frame(hbox)
        bot.pack(side="bottom", fill="x")
        ttk.Label(bot, textvariable=self.se_phr_status_var, foreground="#555", wraplength=430,
                  justify="left").pack(side="bottom", anchor="w", pady=(4, 0))
        b2 = ttk.Frame(bot)
        b2.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(b2, text="Открыть JSON…", command=self._se_phr_open_json).pack(side="left")
        ttk.Button(b2, text="Из файла книги", command=self._se_phr_from_book).pack(side="left", padx=6)
        self.se_phr_save_btn = ttk.Button(b2, text="Сохранить для книги", command=self._se_phr_save_book)
        self.se_phr_save_btn.pack(side="left")
        b1 = ttk.Frame(bot)
        b1.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(b1, text="Применить правку", command=self._se_phr_apply).pack(side="left")
        ttk.Button(b1, text="Вернуть как было", command=self._se_phr_revert).pack(side="left", padx=6)
        ttk.Button(b1, text="Удалить фразу", command=self._se_phr_delete).pack(side="left")

        lframe = ttk.Frame(hbox)
        lframe.pack(fill="both", expand=True, pady=(4, 0))
        self.se_phr_list = ttk.Treeview(lframe, columns=("key", "value", "count", "chapter"), show="headings",
                                        height=4, selectmode="browse")
        for col, text, w, anchor, stretch in (("key", "Фраза", 150, "w", False), ("value", "С ударением", 170, "w", True),
                                              ("count", "Раз", 36, "center", False), ("chapter", "Гл.", 36, "center", False)):
            self.se_phr_list.heading(col, text=text)
            self.se_phr_list.column(col, width=w, anchor=anchor, stretch=stretch)
        self.se_phr_list.tag_configure("edited", background="#dff3df")
        psb = ttk.Scrollbar(lframe, command=self.se_phr_list.yview)
        self.se_phr_list.configure(yscrollcommand=psb.set)
        psb.pack(side="right", fill="y")
        self.se_phr_list.pack(side="left", fill="both", expand=True)
        self.se_phr_list.bind("<<TreeviewSelect>>", self._se_on_phrase_select)

    # ---- worker threads never touch Tk: they post callables, the main thread runs them ----
    def _se_post(self, fn) -> None:
        self._se_queue.put(fn)

    def _se_poll(self) -> None:
        try:
            while True:
                self._se_queue.get_nowait()()
        except queue.Empty:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.after(80, self._se_poll)
        except Exception:
            pass

    # ---- small utilities ----
    def _se_bg(self) -> str:
        try:
            return ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0"
        except Exception:
            return "#f0f0f0"

    def _se_set_status(self, msg: str) -> None:
        self.se_status_var.set(msg)

    def _se_mark_dirty(self, dirty: bool = True) -> None:
        self._se_dirty = dirty
        self.se_save_btn.configure(text="Сохранить словарь *" if dirty else "Сохранить словарь")

    # ---- tab / chapters ----
    def _se_on_tab_changed(self, _event=None):
        try:
            tab = self.notebook.tab(self.notebook.select(), "text")
        except Exception:
            return
        if tab != "Словарь ударений":
            return
        key = (str(getattr(self, "book_path", "")), len(getattr(self, "chapters", []) or []))
        if key != self._se_book_key:
            self._se_book_key = key
            self._se_fill_chapters()
        else:
            sel = self.chapters_list.curselection()
            if sel and sel[0] != self._se_chapter_idx:
                self._se_show_chapter(sel[0])

    def _se_fill_chapters(self):
        chapters = getattr(self, "chapters", None) or []
        self.se_chapter_combo["values"] = [f"{i}. {t}" for i, (t, _x) in enumerate(chapters, 1)]
        if not chapters:
            self._se_chapter_idx = None
            self._se_set_text("")
            self.se_chapter_var.set("")
            self._se_set_status("Сначала загрузите книгу на вкладке «Озвучка».")
            return
        sel = self.chapters_list.curselection()
        self._se_show_chapter(sel[0] if sel else 0)

    def _se_combo_index(self) -> int:
        try:
            return int(self.se_chapter_combo.current())
        except Exception:
            return 0

    def _se_show_chapter(self, idx: int):
        chapters = getattr(self, "chapters", None) or []
        if not chapters or not (0 <= idx < len(chapters)):
            return
        self._se_chapter_idx = idx
        self.se_chapter_combo.current(idx)
        self._se_raw_text = chapters[idx][1]
        self._se_marked_text = None
        if self.se_view_var.get() == "marked":
            self.se_view_var.set("text")
        self._se_set_text(self._se_raw_text)
        self._se_search()
        if self.se_word_var.get().strip():
            self._se_analyze(self.se_word_var.get())

    def _se_set_text(self, text: str):
        self.se_text.configure(state="normal")
        self.se_text.delete("1.0", "end")
        self.se_text.insert("1.0", text)
        if self.se_view_var.get() == "marked":
            for m in re.finditer(r"\+[А-Яа-яЁё]", text):
                self.se_text.tag_add("mark", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        self.se_text.configure(state="disabled")

    def _se_change_view(self):
        if self.se_view_var.get() == "marked":
            if self._se_marked_text is None:
                self._se_refresh_preview()
                return
            self._se_set_text(self._se_marked_text)
        else:
            self._se_set_text(self._se_raw_text)
        self._se_search()

    # ---- search ----
    def _se_search(self):
        for tag in ("hit", "cur"):
            self.se_text.tag_remove(tag, "1.0", "end")
        self._se_hits, self._se_hit_i = [], -1
        pat = search_pattern(self.se_search_var.get(), self.se_whole_var.get())
        if pat is None:
            self.se_count_var.set("")
            return
        text = self.se_text.get("1.0", "end-1c")
        for m in pat.finditer(text):
            self._se_hits.append((m.start(), m.end()))
            self.se_text.tag_add("hit", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        n = len(self._se_hits)
        self.se_count_var.set(f"найдено: {n}" if n else "не найдено")
        if n:
            self._se_step(+1)

    def _se_step(self, direction: int):
        if not self._se_hits:
            return
        self.se_text.tag_remove("cur", "1.0", "end")
        self._se_hit_i = (self._se_hit_i + direction) % len(self._se_hits)
        s, e = self._se_hits[self._se_hit_i]
        self.se_text.tag_add("cur", f"1.0+{s}c", f"1.0+{e}c")
        self.se_text.see(f"1.0+{s}c")
        self.se_count_var.set(f"{self._se_hit_i + 1} из {len(self._se_hits)}")

    def _se_edit_searched(self):
        self._se_load_word(self.se_search_var.get())

    def _se_on_text_double_click(self, event):
        try:
            idx = self.se_text.index(f"@{event.x},{event.y}")
            line, col = idx.split(".")
            text = self.se_text.get(f"{line}.0", f"{line}.end")
            for m in _WORD_AT_RE.finditer(text):
                if m.start() <= int(col) <= m.end():
                    self._se_load_word(m.group(0))
                    break
        except Exception:
            pass
        return "break"

    # ---- visual word ----
    def _se_load_word(self, raw: str):
        word = (raw or "").strip()
        if not word:
            return
        key = " ".join(plain_word(word).lower().split())
        self._se_phrase_key = None
        self.se_word_var.set(key)
        if key in self._se_entries:
            value = self._se_entries[key]
        elif "+" in word:
            value = word.lower()
        else:
            value = key
        self._se_render_word(value)
        self.se_search_var.set(key)
        self._se_analyze(key)

    def _se_render_word(self, value: str):
        self._se_value = value
        t = self.se_glyphs
        t.configure(state="normal")
        t.delete("1.0", "end")
        for tag in [n for n in t.tag_names() if re.fullmatch(r"v\d+", n)]:
            t.tag_delete(tag)
        plain = plain_word(value)
        stress = set(stress_positions(value))
        phrase = " " in plain.strip()
        size = 17 if phrase else 28
        t.configure(font=("Consolas", size), height=2 if phrase else 1, wrap="word" if phrase else "none")
        t.tag_configure("plus", foreground="#d00000", font=("Consolas", size, "bold"))
        for i, ch in enumerate(plain):
            if i in stress:
                t.insert("end", "+", ("plus",))
            if ch in VOWELS:
                tag = f"v{i}"
                t.insert("end", ch, ("stressed" if i in stress else "vowel", tag))
                t.tag_bind(tag, "<Button-1>", lambda e, i=i: self._se_click_vowel(i))
                t.tag_bind(tag, "<Enter>", lambda e: t.configure(cursor="hand2"))
                t.tag_bind(tag, "<Leave>", lambda e: t.configure(cursor="arrow"))
            else:
                t.insert("end", ch, ("cons",))
        t.configure(state="disabled")
        self.se_value_var.set(value if "+" in value else (f"{value}  (ударение не задано)" if value else ""))

    def _se_click_vowel(self, i: int):
        self._se_render_word(place_stress_in_phrase(self._se_value, i))
        return "break"

    # ---- dictionary ----
    def _se_fill_list(self):
        flt = self.se_filter_var.get().strip().lower()
        self.se_list.delete(*self.se_list.get_children())
        for k in sorted(self._se_entries):
            if flt and flt not in k:
                continue
            self.se_list.insert("", "end", iid=k, values=(k, self._se_entries[k]))

    def _se_on_list_select(self, _e=None):
        sel = self.se_list.selection()
        if sel:
            self._se_load_word(sel[0])

    def _se_apply_entry(self):
        key = " ".join(plain_word(self.se_word_var.get()).lower().split())
        err = validate_phrase(key, self._se_value) if " " in key else validate_entry(key, self._se_value)
        if err:
            messagebox.showwarning("Словарь ударений", err)
            return
        if " " not in key and self._se_is_homograph(key):
            if not messagebox.askyesno(
                    "Омограф",
                    f"«{key}» читается по-разному в зависимости от контекста.\n\n"
                    "Запись в словаре применится ко ВСЕМ вхождениям слова в любой книге.\n"
                    "Всё равно добавить?"):
                return
        self._se_entries[key] = self._se_value
        self._se_mark_dirty()
        self._se_fill_list()
        self._se_set_status(f"«{key}» → {self._se_value}. Нажмите «Сохранить словарь».")

    def _se_delete_entry(self):
        key = plain_word(self.se_word_var.get()).strip().lower()
        if key in self._se_entries:
            del self._se_entries[key]
            self._se_mark_dirty()
            self._se_fill_list()
            self._se_set_status(f"«{key}» удалено из словаря (пока не сохранено).")
        else:
            self._se_set_status("Этого слова нет в словаре.")

    def _se_save(self) -> bool:
        try:
            save_entries(self._se_path, self._se_entries, self._se_extras)
        except Exception as e:
            messagebox.showerror("Словарь ударений", f"Не удалось сохранить {self._se_path}:\n{e}")
            return False
        self._se_mark_dirty(False)
        self._se_set_status(f"Сохранено: {self._se_path} ({len(self._se_entries)} слов).")
        try:
            self.log(f"Словарь ударений сохранён: {self._se_path} ({len(self._se_entries)} слов).")
        except Exception:
            pass
        return True

    def _se_reload(self):
        if self._se_dirty and not messagebox.askyesno(
                "Словарь ударений", "Есть несохранённые изменения. Перечитать файл и потерять их?"):
            return
        self._se_entries, self._se_extras = load_entries(self._se_path)
        self._se_mark_dirty(False)
        self._se_fill_list()
        self._se_set_status("Словарь перечитан с диска.")

    # ---- homograph alert ----
    def _se_is_homograph(self, key: str) -> bool:
        try:
            import fb2_reader
            import grammar_gate
            return key in fb2_reader._load_ambiguous_stress_words() or key in grammar_gate.PARADIGM_TABLE
        except Exception:
            return False

    def _se_analyze(self, word: str):
        key = plain_word(word).strip().lower()
        self.se_variants.delete(*self.se_variants.get_children())
        self._se_groups = []
        if " " in key:
            self.se_variants.pack_forget()            # no per-reading table for a phrase: more room for the list
            self.se_alert_var.set("Фраза: кликайте по гласным, затем «Применить правку» (вкладка «Омографы книги»).")
            self.se_alert_label.configure(fg="#444")
            return
        if not self.se_variants.winfo_manager():
            self.se_variants.pack(fill="x", pady=(4, 0))
        if not key or not self._se_raw_text:
            self.se_alert_var.set("")
            return
        self._se_job += 1
        job = self._se_job
        known = self._se_is_homograph(key)
        self.se_alert_var.set("Анализирую вхождения в главе…")
        self.se_alert_label.configure(fg="#444")
        text = self._se_raw_text

        def work():
            try:
                import fb2_reader
                variants = fb2_reader.stress_word_variants(text, key)
            except Exception as e:
                variants = None
                err = f"{type(e).__name__}: {e}"
            self._se_post(lambda: self._se_show_variants(job, key, known, variants,
                                                         None if variants is not None else err))

        threading.Thread(target=work, daemon=True).start()

    def _se_show_variants(self, job, key, known, variants, err):
        if job != self._se_job:
            return
        if variants is None:
            self.se_alert_var.set(f"Не удалось проанализировать: {err}")
            return
        groups = group_variants(variants)
        self._se_groups = groups
        for gi, g in enumerate(groups):
            self.se_variants.insert("", "end", iid=str(gi), values=(g["form"], g["count"], g["example"]))
        total = sum(g["count"] for g in groups)
        if not groups:
            self.se_alert_var.set(f"«{key}»: в этой главе не найдено (или silero-stress недоступен).")
            self.se_alert_label.configure(fg="#444")
        elif len(groups) >= 2:
            forms = ", ".join(g["form"] for g in groups)
            self.se_alert_var.set(
                f"⚠ «{key}» — омограф: в этой главе {total} вхождений читаются по-разному ({forms}). "
                "Запись в словаре зафиксирует одно чтение ВЕЗДЕ. Выберите строку, чтобы увидеть места в тексте.")
            self.se_alert_label.configure(fg="#b00020")
        elif known:
            self.se_alert_var.set(
                f"ℹ «{key}» — известный омограф, но в этой главе все {total} вхождений читаются как "
                f"{groups[0]['form']}. В других главах чтение может быть другим.")
            self.se_alert_label.configure(fg="#8a5a00")
        else:
            self.se_alert_var.set(f"«{key}»: {total} вхождений, везде читается как {groups[0]['form']}.")
            self.se_alert_label.configure(fg="#2a6f2a")

    def _se_on_variant_select(self, _e=None):
        sel = self.se_variants.selection()
        if not sel:
            return
        g = self._se_groups[int(sel[0])]
        if self.se_view_var.get() != "text":
            self.se_view_var.set("text")
            self._se_set_text(self._se_raw_text)
        for tag in ("hit", "cur"):
            self.se_text.tag_remove(tag, "1.0", "end")
        word_len = len(plain_word(g["form"]))
        for pos in g["positions"]:
            self.se_text.tag_add("hit", f"1.0+{pos}c", f"1.0+{pos + word_len}c")
        if g["positions"]:
            self.se_text.tag_add("cur", f"1.0+{g['positions'][0]}c", f"1.0+{g['positions'][0] + word_len}c")
            self.se_text.see(f"1.0+{g['positions'][0]}c")
        self.se_count_var.set(f"{g['form']}: {g['count']}")

    # ---- preview / re-run ----
    def _se_current_text(self) -> str:
        return self._se_raw_text

    def _se_refresh_preview(self):
        """Dry-run of the whole chain on the shown chapter with the CURRENT (even unsaved) entries."""
        if not self._se_raw_text:
            self._se_set_status("Нет открытой главы.")
            return
        if self._se_dirty and not self._se_save():
            return
        self._se_flush_caches()
        self._se_set_status("Считаю ударения главы…")
        text = self._se_raw_text
        strip_mono = self.strip_mono_var.get()
        try:
            from fb2_reader import book_manual_stress_overrides_path, stress_dry_run
            book_overrides = book_manual_stress_overrides_path(self.book_path) if self.book_path else None
        except Exception as e:
            self._se_set_status(f"Ошибка: {e}")
            return

        def work():
            try:
                marked, info = stress_dry_run(text, strip_monosyllables=strip_mono,
                                              book_overrides_path=book_overrides)
            except Exception as e:
                msg = f"Предпросмотр не удался: {type(e).__name__}: {e}"
                self._se_post(lambda: self._se_set_status(msg))
                return

            def done():
                self._se_marked_text = marked
                self.se_view_var.set("marked")
                self._se_set_text(marked)
                self._se_search()
                self._se_set_status(f"Готово за {info['seconds']:.1f} с: меток «+» — {info['marks']}, "
                                    f"слов в словаре — {info['dictionary_words']}.")
            self._se_post(done)

        threading.Thread(target=work, daemon=True).start()

    def _se_flush_caches(self):
        try:
            import fb2_reader
            fb2_reader.reload_stress_models()
        except Exception:
            pass

    def _se_rerun_synthesis(self):
        """Save -> flush every in-memory stress cache -> mark the selected chapters stale -> synthesize."""
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Re-run Synthesis", "Дождитесь окончания текущей озвучки.")
            return
        if self._se_scan_running():
            messagebox.showinfo("Re-run Synthesis", "Дождитесь окончания анализа омографов.")
            return
        if not getattr(self, "chapters", None):
            messagebox.showwarning("Re-run Synthesis", "Сначала загрузите книгу.")
            return
        if self._se_dirty and not self._se_save():
            return
        self._se_flush_caches()
        sel = self.chapters_list.curselection()
        if not sel and self._se_chapter_idx is not None:
            self.chapters_list.selection_set(self._se_chapter_idx)
            self._on_chapters_selection_change()
            sel = self.chapters_list.curselection()
        indices = sorted(i + 1 for i in sel)
        outdir = Path(self.outdir_var.get().strip() or "audiobook_output")
        stale = stale_meta_files(outdir, self.chapters, indices)
        removed = 0
        for m in stale:
            try:
                m.unlink()
                removed += 1
            except OSError:
                pass
        self._se_set_status(f"Кэши сброшены, помечено к пересинтезу глав: {removed}. Запускаю озвучку…")
        self.log(f"Re-run Synthesis: словарь сохранён, кэши ударений сброшены, глав к пересинтезу: "
                 f"{removed or len(indices)}.")
        self.notebook.select(0)
        self.start_synthesis()

    # ---- «Анализ омографов книги» (worker thread -> list of phrase keys) ----
    def _se_scan_running(self) -> bool:
        return bool(self._se_scan_thread and self._se_scan_thread.is_alive())

    def _se_scan_toggle(self):
        if self._se_scan_running():
            self._se_scan_cancel.set()
            self.se_scan_var.set("Останавливаю…")
            return
        chapters = list(getattr(self, "chapters", None) or [])
        if not chapters:
            messagebox.showwarning("Анализ омографов", "Сначала загрузите книгу на вкладке «Озвучка».")
            return
        if getattr(self, "_worker", None) and self._worker.is_alive():
            messagebox.showinfo("Анализ омографов",
                                "Идёт озвучка: анализ использует те же модели ударений. Дождитесь её окончания.")
            return
        if self._se_phr_dirty and not messagebox.askyesno(
                "Анализ омографов", "В списке есть несохранённые правки фраз. Заменить список результатом анализа?"):
            return
        self._se_flush_caches()
        only_multi = self.se_phr_multi_var.get()
        cancel = threading.Event()
        self._se_scan_cancel = cancel
        self.se_scan_btn.configure(text="Остановить анализ")
        self.se_right_nb.select(1)
        self.se_progress.start(12)
        self.se_scan_var.set("Загружаю модели ударений…")
        self.se_phr_status_var.set("")

        def work():
            try:
                import fb2_reader
                entries, stats = fb2_reader.analyze_chapters_homographs(
                    chapters, only_multi=only_multi, cancel=cancel,
                    progress_cb=lambda m: self._se_post(lambda m=m: self.se_scan_var.set(m.strip())))
                outcome = ("ok", stats)
            except InterruptedError:
                outcome = ("cancel", None)
            except Exception as e:
                outcome = ("error", f"{type(e).__name__}: {e}")
            finally:
                try:
                    import fb2_reader as _fr
                    _fr.reset_stress_context()
                except Exception:
                    pass
            self._se_post(lambda: self._se_scan_done(*outcome))

        self._se_scan_thread = threading.Thread(target=work, daemon=True)
        self._se_scan_thread.start()

    def _se_scan_done(self, kind, payload):
        self.se_progress.stop()
        self.se_scan_btn.configure(text="Анализ омографов книги")
        if kind == "cancel":
            self.se_scan_var.set("Анализ остановлен.")
            return
        if kind == "error":
            self.se_scan_var.set(f"Анализ не удался: {payload}")
            return
        self._se_phr = {it["key"]: {"value": it["value"], "orig": it["value"], "count": it["count"],
                                    "chapter": it["chapter"]} for it in payload["items"]}
        self._se_phr_dirty = False
        self._se_fill_phrases()
        unresolved = payload["unresolved_occurrences"]
        self.se_scan_var.set(
            f"Готово: слов {payload['words_total']}, вхождений омографов {payload['homograph_occurrences']} "
            f"({payload['homograph_words_found']} слов, ≥2 чтений: {payload['homograph_words_multi']}). "
            f"Фраз в списке: {len(self._se_phr)}."
            + (f" Без однозначного контекста: {unresolved}." if unresolved else ""))
        self.se_phr_status_var.set("Выберите фразу, исправьте ударение, «Применить правку», «Сохранить для книги».")

    # ---- phrase list ----
    def _se_fill_phrases(self):
        flt = " ".join(self.se_phr_filter_var.get().lower().split())
        sel = self.se_phr_list.selection()
        self.se_phr_list.delete(*self.se_phr_list.get_children())
        for k in sorted(self._se_phr):
            if flt and flt not in k:
                continue
            it = self._se_phr[k]
            self.se_phr_list.insert("", "end", iid=k, tags=("edited",) if it["value"] != it["orig"] else (),
                                    values=(k, it["value"], it["count"] or "", it["chapter"] or ""))
        if sel and self.se_phr_list.exists(sel[0]):
            self.se_phr_list.selection_set(sel[0])

    def _se_phr_mark_dirty(self, dirty=True):
        self._se_phr_dirty = dirty
        self.se_phr_save_btn.configure(text="Сохранить для книги *" if dirty else "Сохранить для книги")

    def _se_locate_chapter(self, key: str) -> int:
        """1-based chapter that holds the phrase (the analysis remembers it; a loaded file does not)."""
        it = self._se_phr.get(key)
        if it and it.get("chapter"):
            return int(it["chapter"])
        pat = search_pattern(key, True)
        for i, (_t, text) in enumerate(getattr(self, "chapters", None) or [], 1):
            if pat and pat.search(text):
                if it:
                    it["chapter"] = i
                return i
        return 0

    def _se_on_phrase_select(self, _e=None):
        sel = self.se_phr_list.selection()
        if not sel or sel[0] not in self._se_phr:
            return
        key = sel[0]
        ch = self._se_locate_chapter(key)
        if ch and ch - 1 != self._se_chapter_idx and getattr(self, "chapters", None):
            self._se_show_chapter(ch - 1)
        self._se_phrase_key = key
        self.se_word_var.set(key)
        self._se_render_word(self._se_phr[key]["value"])
        self.se_search_var.set(key)
        self._se_analyze(key)
        self._se_fill_phrases_row(key)

    def _se_fill_phrases_row(self, key):
        it = self._se_phr.get(key)
        if it and self.se_phr_list.exists(key):
            self.se_phr_list.item(key, values=(key, it["value"], it["count"] or "", it["chapter"] or ""),
                                  tags=("edited",) if it["value"] != it["orig"] else ())

    def _se_phr_apply(self):
        key = self._se_phrase_key
        if key is None or key not in self._se_phr:
            messagebox.showinfo("Омографы книги", "Сначала выберите фразу в списке.")
            return
        err = validate_phrase(key, self._se_value)
        if err:
            messagebox.showwarning("Омографы книги", err)
            return
        self._se_phr[key]["value"] = self._se_value
        self._se_fill_phrases_row(key)
        self._se_phr_mark_dirty()
        self.se_phr_status_var.set(f"«{key}» → {self._se_value}. Нажмите «Сохранить для книги».")

    def _se_phr_revert(self):
        key = self._se_phrase_key
        if key in self._se_phr:
            self._se_phr[key]["value"] = self._se_phr[key]["orig"]
            self._se_render_word(self._se_phr[key]["value"])
            self._se_fill_phrases_row(key)
            self._se_phr_mark_dirty(any(v["value"] != v["orig"] for v in self._se_phr.values()))

    def _se_phr_delete(self):
        key = self._se_phrase_key
        if key in self._se_phr:
            del self._se_phr[key]
            self._se_phrase_key = None
            self._se_fill_phrases()
            self._se_phr_mark_dirty()
            self.se_phr_status_var.set(f"«{key}» удалена из списка (пока не сохранено).")

    def _se_phr_load_entries(self, entries: dict, source: str):
        good = {k: v for k, v in entries.items() if " " in k.strip() and not validate_phrase(k, v)}
        self._se_phr = {" ".join(k.lower().split()): {"value": v, "orig": v, "count": 0, "chapter": 0}
                        for k, v in good.items()}
        self._se_phrase_key = None
        self._se_fill_phrases()
        self._se_phr_mark_dirty(False)
        skipped = len(entries) - len(good)
        self.se_right_nb.select(1)
        self.se_phr_status_var.set(
            f"Загружено фраз: {len(good)} из {source}" + (f" (пропущено не-фраз или с ошибкой: {skipped})" if skipped else "")
            + ".")

    def _se_phr_open_json(self):
        if self._se_phr_dirty and not messagebox.askyesno(
                "Омографы книги", "В списке есть несохранённые правки. Заменить его содержимым файла?"):
            return
        path = filedialog.askopenfilename(title="JSON с фразами ударений",
                                          filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")])
        if path:
            entries, _extras = load_entries(path)
            self._se_phr_load_entries(entries, Path(path).name)

    def _se_book_overrides_path(self):
        if not getattr(self, "book_path", None):
            return None
        from fb2_reader import book_manual_stress_overrides_path
        return book_manual_stress_overrides_path(self.book_path)

    def _se_phr_from_book(self):
        p = self._se_book_overrides_path()
        if p is None:
            messagebox.showinfo("Омографы книги", "Сначала загрузите книгу.")
            return
        if not p.exists():
            self.se_phr_status_var.set(f"Файла правок книги ещё нет: {p}")
            return
        if self._se_phr_dirty and not messagebox.askyesno(
                "Омографы книги", "В списке есть несохранённые правки. Заменить его содержимым файла?"):
            return
        entries, _extras = load_entries(p)
        self._se_phr_load_entries(entries, p.name)

    def _se_phr_save_book(self):
        p = self._se_book_overrides_path()
        if p is None:
            messagebox.showinfo("Омографы книги", "Сначала загрузите книгу.")
            return
        if not self._se_phr:
            self.se_phr_status_var.set("Список пуст - сохранять нечего.")
            return
        phrases = {k: v["value"] for k, v in self._se_phr.items() if not validate_phrase(k, v["value"])}
        try:
            n = merge_phrases_into_file(p, phrases)
        except Exception as e:
            messagebox.showerror("Омографы книги", f"Не удалось сохранить {p}:\n{e}")
            return
        for v in self._se_phr.values():
            v["orig"] = v["value"]
        self._se_fill_phrases()
        self._se_phr_mark_dirty(False)
        self._se_flush_caches()
        msg = (f"Сохранено фраз: {n} -> {p}. Они действуют при следующей озвучке этой книги "
               "(Re-run Synthesis пересинтезирует затронутые главы).")
        self.se_phr_status_var.set(msg)
        try:
            self.log(msg)
        except Exception:
            pass
