fb2_reader.py — озвучивание книг в формате FB2 на русском языке.

Возможности:
  * извлечение текста из .fb2 (и .fb2.zip) по главам;
  * озвучка четырьмя способами:
      1) silero      — нейросетевой синтез Silero TTS локально через
                       torch.hub, без отдельного сервера. Живая интонация,
                       естественные паузы на знаках препинания, ударения
                       расставляются автоматически. Модель скачивается один
                       раз (~50-100 МБ), дальше работает офлайн. Требует torch.
      2) silero_rest — синтез через локальный REST-сервис
                       silero_rest_service.py (форк
                       https://github.com/Flokss/Silero-REST-Service).
                       Текст главы автоматически превращается в SSML —
                       добавляются интонационные паузы на запятых, тире и
                       двоеточиях, более длинные паузы между предложениями
                       и абзацами, а вопросительные и восклицательные знаки
                       сохраняются как есть, поэтому Silero озвучивает
                       вопросы и восклицания с соответствующей интонацией.
                       Требует запущенного рядом сервиса (см. раздел ниже)
                       и пакета requests.
      3) online      — через gTTS (Google Text-to-Speech), нужен интернет
                       на каждый запуск, голос неплохой, но менее выразительный.
      4) offline     — через pyttsx3 и системный TTS (espeak и т.п.), самое
                       низкое качество, зато совсем без интернета и без torch.

Установка зависимостей:
  # для локального режима silero (лучшее качество голоса):
  pip install torch torchaudio omegaconf numpy

  # для режима silero_rest (клиент + сам сервис):
  pip install requests numpy fastapi uvicorn torch ruaccent num2words

  # для остальных режимов:
  pip install gTTS pyttsx3 pygame lxml pydub

Если команда "pip" не находится напрямую, используйте "python3 -m pip ..."
вместо "pip ...".

Для offline-режима (pyttsx3) на Linux дополнительно нужен espeak-ng:
  sudo apt install espeak-ng

---

Режим silero_rest: как запустить (2 окна терминала)

Этот режим требует отдельно запущенного локального сервиса
silero_rest_service.py — fb2_reader.py сам к модели не обращается, а
посылает HTTP-запросы на http://localhost:5010, поэтому сервис должен
работать параллельно, в отдельном окне.

Окно 1 — сервис (держать открытым всё время озвучки, не закрывать,
не нажимать Ctrl+C):

  cd audiobook
  python3 silero_rest_service.py

Дождитесь в выводе:

  Downloading Silero TTS model...          # только при первом запуске, ~50-100 МБ
  TTS Model loaded successfully
  RUAccent model loaded successfully
  INFO:     Uvicorn running on http://0.0.0.0:5010

Пока этих строк нет — сервис ещё не готов принимать запросы. Если сервис
остановится или не был запущен, fb2_reader.py в режиме silero_rest начнёт
падать с ошибкой:
  "Подключение не установлено, т.к. конечный компьютер отверг запрос на
   подключение" (WinError 10061)
— это значит, что порт 5010 никто не слушает: проверьте, что окно с
сервисом открыто и в нём видна строка "Uvicorn running on ...".

Окно 2 — сама озвучка (запускать только после того, как сервис поднялся):

  python3 fb2_reader.py book.fb2 --mode silero_rest --rest-url http://localhost:5010 --speaker xenia --outdir audiobook

Флаги для настройки интонационных пауз в режиме silero_rest:
  --sentence-break-ms   (по умолчанию 320) — пауза между предложениями
  --paragraph-break-ms  (по умолчанию 550) — пауза между абзацами
  --comma-break-ms      (по умолчанию 180) — пауза на запятых/тире/двоеточиях

---

Примеры запуска:
  # Лучшее качество без сервиса: нейросетевой голос Silero локально
  python3 fb2_reader.py book.fb2 --mode silero --speaker xenia --outdir audiobook

  # То же самое, но сразу слушать по мере озвучки
  python3 fb2_reader.py book.fb2 --mode silero --play

  # Через локальный Silero-REST-Service с интонационными паузами (SSML);
  # сервис должен быть уже запущен в отдельном окне (см. выше)
  python3 fb2_reader.py book.fb2 --mode silero_rest --rest-url http://localhost:5010 --speaker xenia --outdir audiobook

  # Озвучить книгу через gTTS и сразу проигрывать
  python3 fb2_reader.py book.fb2 --mode online --play

  # Офлайн pyttsx3-режим, начиная с 3-й главы
  python3 fb2_reader.py book.fb2 --mode offline --start 3

  # Просто посмотреть список глав, ничего не озвучивая
  python3 fb2_reader.py book.fb2 --list

Доступные голоса Silero (--speaker): aidar (муж.), baya (жен.),
kseniya (жен.), xenia (жен.), eugene (муж.), random (случайный на каждой фразе).

---

silero_rest_service.py — локальный REST-сервис (FastAPI)

Форк https://github.com/Flokss/Silero-REST-Service, дополненный поддержкой
SSML для интонационных пауз.

Эндпоинты:
  GET /getwav?text_to_speech=<текст>&speaker=xenia&sample_rate=24000
      — как в оригинальном сервисе: обычный текст без пауз/SSML.

  GET /getssmlwav?text_to_speech=<текст или SSML>&speaker=xenia&sample_rate=48000&raw_ssml=false
      — с интонационными паузами. Если raw_ssml=false (по умолчанию),
      text_to_speech передаётся как обычный текст, и сервис сам оборачивает
      его в SSML (<speak><p><s>...</s></p></speak>) с паузами на знаках
      препинания и границах предложений/абзацев. Если raw_ssml=true —
      text_to_speech должен быть уже готовым SSML-документом (именно так
      его посылает fb2_reader.py в режиме --mode silero_rest).
      Дополнительные параметры: sentence_break_ms, paragraph_break_ms,
      comma_break_ms — длительности пауз в миллисекундах.

При первом запуске сервис скачивает модель Silero (silero_model.pt,
~50-100 МБ) и модель RUAccent для расстановки ударений — это разовая
операция, дальше сервис стартует быстро и работает офлайн.

---

Частые проблемы

"Подключение не установлено, т.к. конечный компьютер отверг запрос на
подключение" / WinError 10061
  Сервис silero_rest_service.py не запущен или упал. Откройте отдельное
  окно, запустите его командой из раздела "Окно 1" выше и дождитесь строки
  "Uvicorn running on http://0.0.0.0:5010", прежде чем запускать озвучку.

'"pip" не является внутренней или внешней командой...'
  Используйте "python3 -m pip install ..." вместо "pip install ...".
