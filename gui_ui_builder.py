import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from fb2_reader import (
    ATTRIBUTION_PROVIDERS,
    DEFAULT_ATTRIBUTION_MODEL,
    DEFAULT_ATTRIBUTION_PROVIDER,
    TTS_MODES,
    YANDEX_VOICES,
    COSYVOICE_DEFAULT_REST_URL,
    QWEN_TTS_VOICES,
    QWEN_TTS_DEFAULT_VOICE,
    QWEN_TTS_LOCAL_DEFAULT_URL,
)
from silero_config import DEFAULT_SILERO_MODEL, SILERO_MODELS

from gui_shared import (
    COSYVOICE_DEFAULT_ENGINE,
    COSYVOICE_ENGINE_LABELS,
    LOG_FILE_PATH,
    add_context_menu,
)


class UIBuilderMixin:
    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Озвучка")

        voices_tab = ttk.Frame(self.notebook)
        self.notebook.add(voices_tab, text="Голоса CosyVoice")

        outer = ttk.Frame(main_tab, padding=10)
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
        self.check_stress_btn = ttk.Button(
            btn_row, text="Проверить ударения", command=self.check_ambiguous_stress
        )
        self.check_stress_btn.pack(side="left", padx=(8, 0))
        ttk.Button(
            btn_row, text="Словарь ударений", command=self.open_stress_dictionary_editor
        ).pack(side="left", padx=(8, 0))

        # --- встроенный плеер: играет уже озвученную главу (выбранную в
        # списке слева) или очередную главу сразу после её озвучки (если
        # включено "Проигрывать после озвучки каждой главы"). Раньше
        # проигрывание шло через системный проигрыватель/winsound без
        # какой-либо возможности остановить, поставить на паузу или
        # перемотать — теперь звук идёт через pygame.mixer, которым можно
        # управлять прямо отсюда: кнопка play/пауза, стоп и полоса
        # прокрутки (перемотка кликом/перетаскиванием).
        player_frame = ttk.LabelFrame(right, text="Проигрывание", padding=(8, 4))
        player_frame.pack(side="bottom", fill="x", pady=(0, 6))

        self.player_title_label = ttk.Label(player_frame, text="Глава не выбрана", foreground="#555")
        self.player_title_label.pack(anchor="w", fill="x")

        player_seek_row = ttk.Frame(player_frame)
        player_seek_row.pack(fill="x", pady=(4, 0))
        self.player_pos_var = tk.DoubleVar(value=0.0)
        self.player_scale = ttk.Scale(
            player_seek_row, from_=0, to=1, orient="horizontal",
            variable=self.player_pos_var, command=self._on_player_scale_move, state="disabled",
        )
        self.player_scale.pack(side="left", fill="x", expand=True)
        self.player_scale.bind("<ButtonPress-1>", self._on_player_seek_start)
        self.player_scale.bind("<ButtonRelease-1>", self._on_player_seek_commit)
        self.player_time_label = ttk.Label(player_seek_row, text="", width=12, anchor="e")
        self.player_time_label.pack(side="left", padx=(6, 0))

        player_controls_row = ttk.Frame(player_frame)
        player_controls_row.pack(fill="x", pady=(4, 0))
        self.player_play_btn = ttk.Button(
            player_controls_row, text="▶ Играть", command=self._player_play_pause,
            state="disabled", width=12,
        )
        self.player_play_btn.pack(side="left")
        self.player_stop_btn = ttk.Button(
            player_controls_row, text="⏹ Стоп", command=self._player_stop, state="disabled", width=10,
        )
        self.player_stop_btn.pack(side="left", padx=(6, 0))

        self.progress_label = ttk.Label(right, text="", foreground="#555")
        self.progress_label.pack(side="bottom", fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1000, value=0)
        self.progress.pack(side="bottom", fill="x", pady=(2, 2))

        log_frame = ttk.LabelFrame(right, text="Журнал", padding=6)
        log_frame.pack(side="bottom", fill="both", expand=True, pady=(8, 0))

        log_buttons_row = ttk.Frame(log_frame)
        log_buttons_row.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(
            log_buttons_row, text="Копировать всё", command=self._copy_log_to_clipboard
        ).pack(side="left")
        ttk.Button(
            log_buttons_row, text="Открыть файл журнала", command=self._open_log_file
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            log_buttons_row, text=f"Файл: logs\\{LOG_FILE_PATH.name}", foreground="#555"
        ).pack(side="left", padx=(10, 0))

        log_text_row = ttk.Frame(log_frame)
        log_text_row.pack(side="top", fill="both", expand=True)
        self.log_text = tk.Text(log_text_row, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_text_row, orient="vertical", command=self.log_text.yview)
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

        self.cosyvoice_rest_url_var = tk.StringVar(value=COSYVOICE_DEFAULT_REST_URL)
        self.cosyvoice_engine_var = tk.StringVar(value=COSYVOICE_ENGINE_LABELS[COSYVOICE_DEFAULT_ENGINE])
        row += 1
        self.cosyvoice_tab_hint = ttk.Label(
            settings,
            text="Список голосов и добавление нового голоса из аудиофайла — "
                 "на вкладке «Голоса CosyVoice» вверху окна.",
            foreground="#555", wraplength=280, justify="left",
        )
        self.cosyvoice_tab_hint.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))

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

        # --- Qwen3-TTS: облачный (DashScope) и локальный (свой сервер) ---
        qwen_frame = ttk.LabelFrame(
            right_scroll, text="Qwen3-TTS (режимы qwen_tts / qwen_tts_local)", padding=10
        )
        qwen_frame.pack(fill="x", pady=(8, 0))

        qrow = 0
        ttk.Label(qwen_frame, text="API-ключ (DashScope, облачный режим):").grid(
            row=qrow, column=0, sticky="w", pady=3
        )
        qkey_row = ttk.Frame(qwen_frame)
        qkey_row.grid(row=qrow, column=1, columnspan=2, sticky="ew", pady=3)
        self.qwen_api_key_var = tk.StringVar(value=os.environ.get("DASHSCOPE_API_KEY", ""))
        self.qwen_api_key_entry = add_context_menu(
            ttk.Entry(qkey_row, textvariable=self.qwen_api_key_var, show="•")
        )
        self.qwen_api_key_entry.pack(side="left", fill="x", expand=True)
        self._qwen_key_show_var = tk.BooleanVar(value=False)

        def _toggle_qwen_key_visibility():
            self.qwen_api_key_entry.configure(show="" if self._qwen_key_show_var.get() else "•")

        ttk.Checkbutton(
            qkey_row, text="показать", variable=self._qwen_key_show_var, command=_toggle_qwen_key_visibility
        ).pack(side="left", padx=(6, 0))

        qrow += 1
        ttk.Label(qwen_frame, text="Голос:").grid(row=qrow, column=0, sticky="w", pady=3)
        self._qwen_voice_keys = list(QWEN_TTS_VOICES.keys())
        qwen_voice_labels = [f"{k} — {v}" for k, v in QWEN_TTS_VOICES.items()]
        default_qwen_label = next(
            (lbl for lbl, k in zip(qwen_voice_labels, self._qwen_voice_keys) if k == QWEN_TTS_DEFAULT_VOICE),
            qwen_voice_labels[0],
        )
        self.qwen_voice_var = tk.StringVar(value=default_qwen_label)
        self.qwen_voice_combo = ttk.Combobox(
            qwen_frame, textvariable=self.qwen_voice_var, values=qwen_voice_labels,
            state="readonly", width=28,
        )
        self.qwen_voice_combo.grid(row=qrow, column=1, columnspan=2, sticky="ew", pady=3)

        qrow += 1
        ttk.Separator(qwen_frame, orient="horizontal").grid(
            row=qrow, column=0, columnspan=3, sticky="ew", pady=(10, 6)
        )

        qrow += 1
        ttk.Label(
            qwen_frame, text="Локальный режим (свой GPU, без интернета/карты):",
            font=("", 9, "bold"),
        ).grid(row=qrow, column=0, columnspan=3, sticky="w")

        qrow += 1
        ttk.Label(qwen_frame, text="Адрес сервера:").grid(row=qrow, column=0, sticky="w", pady=3)
        self.qwen_local_url_var = tk.StringVar(value=QWEN_TTS_LOCAL_DEFAULT_URL)
        self.qwen_local_url_entry = add_context_menu(
            ttk.Entry(qwen_frame, textvariable=self.qwen_local_url_var)
        )
        self.qwen_local_url_entry.grid(row=qrow, column=1, columnspan=2, sticky="ew", pady=3)

        qrow += 1
        ttk.Label(qwen_frame, text="Команда автозапуска сервера:").grid(
            row=qrow, column=0, sticky="w", pady=3
        )
        self.qwen_local_start_cmd_var = tk.StringVar(value="")
        self.qwen_local_start_cmd_entry = add_context_menu(
            ttk.Entry(qwen_frame, textvariable=self.qwen_local_start_cmd_var)
        )
        self.qwen_local_start_cmd_entry.grid(row=qrow, column=1, sticky="ew", pady=3)
        self.qwen_local_start_cmd_btn = ttk.Button(
            qwen_frame, text="Обзор…", command=self._browse_qwen_local_start_cmd
        )
        self.qwen_local_start_cmd_btn.grid(row=qrow, column=2, sticky="w", padx=(6, 0), pady=3)

        qrow += 1
        self.qwen_local_auto_stop_var = tk.BooleanVar(value=True)
        self.qwen_local_auto_stop_check = ttk.Checkbutton(
            qwen_frame, text="Останавливать сервер автоматически после озвучки",
            variable=self.qwen_local_auto_stop_var,
        )
        self.qwen_local_auto_stop_check.grid(row=qrow, column=0, columnspan=3, sticky="w", pady=3)

        qrow += 1
        ttk.Label(
            qwen_frame,
            text="Облачный режим (qwen_tts): платный сервис Alibaba DashScope — нужен\n"
                 "интернет и ключ (переменная DASHSCOPE_API_KEY или поле выше).\n"
                 "Локальный режим (qwen_tts_local): модель считается на вашей видеокарте,\n"
                 "бесплатно и без карты — но сервер (Gradio-демо Qwen3-TTS, например через\n"
                 "Pinokio) нужно поставить один раз отдельно. Укажите выше путь к .bat/.exe,\n"
                 "который его запускает, — тогда программа сама поднимет и погасит сервер\n"
                 "при озвучке; если оставить поле пустым, сервер нужно будет запускать\n"
                 "вручную самому и держать включённым. Обе интеграции написаны по\n"
                 "документации и не проверены вживую — если при озвучке возникнет ошибка,\n"
                 "пришлите её текст целиком, поправим.\n"
                 "Разные голоса для диалогов — в блоке «Разные голоса для диалогов» выше.",
            foreground="#555", justify="left",
        ).grid(row=qrow, column=0, columnspan=3, sticky="w", pady=(8, 0))

        qwen_frame.columnconfigure(1, weight=1)
        self._qwen_frame = qwen_frame

        # --- интонация: паузы по знакам препинания, ударения, усиление ---
        intonation = self._intonation_frame = ttk.LabelFrame(
            right_scroll, text="Интонация и паузы (silero / silero_rest / cosyvoice)", padding=10
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
        self.accent_check = ttk.Checkbutton(
            intonation, text="Расставлять ударения автоматически (только silero/silero_rest)",
            variable=self.accent_var,
        )
        self.accent_check.grid(row=irow, column=0, columnspan=2, sticky="w", pady=3)

        irow += 1
        self.yo_var = tk.BooleanVar(value=True)
        self.yo_check = ttk.Checkbutton(
            intonation, text='Заменять "е" на "ё" где нужно (только silero/silero_rest)',
            variable=self.yo_var,
        )
        self.yo_check.grid(row=irow, column=0, columnspan=2, sticky="w", pady=3)

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

        self._build_voices_tab(voices_tab)
    def _build_voices_tab(self, tab: ttk.Frame):
        """Отдельная вкладка «Голоса CosyVoice»: список уже сохранённых
        профилей голоса (они хранятся на самом сервисе CosyVoice REST, в
        cosyvoice_voices.json — сохраняются постоянно и переживают
        перезапуск программы и сервиса) и форма для добавления нового
        голоса из аудиофайла — без всплывающего окна, прямо на вкладке."""
        pad = ttk.Frame(tab, padding=10)
        pad.pack(fill="both", expand=True)

        ttk.Label(
            pad,
            text="Голоса, склонированные из аудиофайлов для режима CosyVoice. "
                 "Каждый добавленный голос сохраняется на сервисе CosyVoice "
                 "(в cosyvoice_voices.json рядом с моделью) и остаётся доступен "
                 "после перезапуска программы и компьютера — его не нужно "
                 "добавлять заново.",
            wraplength=680, justify="left", foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        url_row = ttk.Frame(pad)
        url_row.pack(fill="x", pady=(0, 8))
        ttk.Label(url_row, text="Адрес сервиса CosyVoice:").pack(side="left")
        self.cosyvoice_rest_url_entry = add_context_menu(
            ttk.Entry(url_row, textvariable=self.cosyvoice_rest_url_var, width=32)
        )
        self.cosyvoice_rest_url_entry.pack(side="left", padx=(6, 0))
        self.cosyvoice_refresh_btn = ttk.Button(
            url_row, text="Обновить список", command=self.refresh_cosyvoice_voices
        )
        self.cosyvoice_refresh_btn.pack(side="left", padx=(6, 0))

        engine_row = ttk.Frame(pad)
        engine_row.pack(fill="x", pady=(0, 8))
        ttk.Label(engine_row, text="Движок синтеза:").pack(side="left")
        self.cosyvoice_engine_combo = ttk.Combobox(
            engine_row, textvariable=self.cosyvoice_engine_var, state="readonly",
            width=34, values=list(COSYVOICE_ENGINE_LABELS.values()),
        )
        self.cosyvoice_engine_combo.pack(side="left", padx=(6, 0))
        self.cosyvoice_engine_combo.bind("<<ComboboxSelected>>", self._on_cosyvoice_engine_changed)
        ttk.Label(
            pad,
            text="ESpeech RL-V2 — рекомендуется: тоже F5-TTS-архитектура, но менее шумный "
                 "результат, чем у XTTS-v2 (по независимым сравнениям), и поддерживает "
                 "клонирование (новее версии SFT-95K, которая использовалась раньше). "
                 "Расстановку ударений не использует — несмотря на документацию, "
                 "чекпоинт на практике произносит символ \"+\" как слово \"плюс\" вместо того, "
                 "чтобы ставить ударение. F5-TTS-Russian winter — более свежий "
                 "чекпоинт F5-TTS для русского с полной разметкой ударений (в отличие от "
                 "обычного F5-TTS-Russian ниже). F5-TTS-Russian — тоже неплохо, но без "
                 "ударений и обучена на меньшем датасете. XTTS-v2 — запасной вариант "
                 "(мультиязычная, заметно более шумное аудио). CosyVoice 3 — "
                 "экспериментальный, официально поддерживает русский, но независимых "
                 "отзывов о качестве на русском пока нет; ставится отдельной секцией "
                 "install.bat в своё окружение .venv_cosyvoice3 (несовместимо со всем "
                 "остальным по версиям зависимостей). Голоса, которые вы уже добавили, "
                 "работают со всеми движками без повторного добавления — переклонировать "
                 "не нужно. Смена движка перезапускает сервис CosyVoice (может занять "
                 "время на загрузку модели — у ESpeech и F5-TTS-Russian winter чекпоинты "
                 "больше, ~1.3–2.7 ГБ при первой загрузке; у CosyVoice3 ещё и веса "
                 "модели, ~1-2 ГБ, при первом запуске).",
            wraplength=680, justify="left", foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        body = ttk.Panedwindow(pad, orient="horizontal")
        body.pack(fill="both", expand=True)

        # --- слева: список уже сохранённых голосов ---
        list_frame = ttk.LabelFrame(body, text="Сохранённые голоса", padding=8)
        body.add(list_frame, weight=1)

        list_row = ttk.Frame(list_frame)
        list_row.pack(fill="both", expand=True)
        self.cosyvoice_voices_listbox = tk.Listbox(list_row, exportselection=False, height=14)
        cv_list_scroll = ttk.Scrollbar(list_row, orient="vertical", command=self.cosyvoice_voices_listbox.yview)
        self.cosyvoice_voices_listbox.configure(yscrollcommand=cv_list_scroll.set)
        self.cosyvoice_voices_listbox.pack(side="left", fill="both", expand=True)
        cv_list_scroll.pack(side="right", fill="y")

        ttk.Label(
            list_frame,
            text="Голос, выбранный здесь, используется как основной, если на "
                 "вкладке «Озвучка» стоит режим CosyVoice.",
            wraplength=280, foreground="#555", justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self._cv_delete_btn = ttk.Button(
            list_frame, text="Удалить выбранный голос",
            command=self._delete_selected_cosyvoice_voice,
        )
        self._cv_delete_btn.pack(anchor="w", pady=(6, 0))

        # --- справа: добавление нового голоса из аудио ---
        add_frame = ttk.LabelFrame(body, text="Добавить голос из аудиофайла", padding=8)
        body.add(add_frame, weight=1)

        ttk.Label(
            add_frame,
            text="Короткая (3-10 сек) чистая запись голоса без музыки и шума — "
                 "например, фрагмент записи любимого диктора.",
            wraplength=300, foreground="#555", justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(add_frame, text="Аудиофайл:").grid(row=1, column=0, sticky="w", pady=4)
        self._cv_add_audio_path_var = tk.StringVar()
        add_context_menu(
            ttk.Entry(add_frame, textvariable=self._cv_add_audio_path_var, width=26)
        ).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(add_frame, text="…", width=3, command=self._browse_cosyvoice_audio).grid(
            row=1, column=2, padx=(4, 0), pady=4
        )

        ttk.Label(add_frame, text="Имя голоса:").grid(row=2, column=0, sticky="w", pady=4)
        self._cv_add_name_var = tk.StringVar()
        add_context_menu(
            ttk.Entry(add_frame, textvariable=self._cv_add_name_var, width=26)
        ).grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(add_frame, text="Текст записи:").grid(row=3, column=0, sticky="nw", pady=4)
        self._cv_add_text_widget = tk.Text(add_frame, width=26, height=4, wrap="word")
        self._cv_add_text_widget.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)

        self._cv_transcribe_btn = ttk.Button(
            add_frame, text="Распознать текст (по аудио)", command=self._transcribe_cosyvoice_audio,
        )
        self._cv_transcribe_btn.grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(
            add_frame,
            text="Движок клонирования голоса — F5-TTS-Russian (обучена именно на русской "
                 "речи). Текст записи желателен для лучшего клонирования — если оставить "
                 "пустым, сервис сам распознает его при первом использовании голоса. Для "
                 "лучшего результата берите 6-15 секунд чистой речи одного голоса без "
                 "музыки и шума.",
            wraplength=300, foreground="#555", justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._cv_add_status_label = ttk.Label(add_frame, text="", foreground="#555", wraplength=300, justify="left")
        self._cv_add_status_label.grid(row=6, column=0, columnspan=3, sticky="w")

        self._cv_add_btn = ttk.Button(add_frame, text="Добавить голос", command=self._add_cosyvoice_voice_from_tab)
        self._cv_add_btn.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))

        add_frame.columnconfigure(1, weight=1)

        self.cosyvoice_voices_listbox.bind("<<ListboxSelect>>", self._on_cosyvoice_voices_listbox_select)

        self._refresh_cosyvoice_voices_listbox()
    def _bind_events(self):
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
