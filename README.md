fb2_reader.py — озвучивание книг в формате FB2 на русском языке.

Возможности:
  * извлечение текста из .fb2 (и .fb2.zip) по главам;
  * озвучка тремя способами:
      1) silero  — (рекомендуется) нейросетевой синтез Silero TTS.
                   Лучшее качество: живая интонация, естественные паузы
                   на знаках препинания, ударения расставляются
                   автоматически. Модель скачивается один раз (~50-100 МБ),
                   дальше работает офлайн. Требует torch.
      2) online  — через gTTS (Google Text-to-Speech), нужен интернет
                   на каждый запуск, голос неплохой, но менее выразительный.
      3) offline — через pyttsx3 и системный TTS (espeak и т.п.), самое
                   низкое качество, зато совсем без интернета и без torch.

Установка зависимостей:
  # для рекомендуемого режима silero (лучшее качество голоса):
  pip install torch torchaudio omegaconf numpy

  # для остальных режимов:
  pip install gTTS pyttsx3 pygame lxml pydub

Для offline-режима (pyttsx3) на Linux дополнительно нужен espeak-ng:
  sudo apt install espeak-ng

Примеры запуска:
  # Лучшее качество: нейросетевой голос Silero, сохранить в wav
  python3 fb2_reader.py book.fb2 --mode silero --speaker xenia --outdir audiobook

  # То же самое, но сразу слушать по мере озвучки
  python3 fb2_reader.py book.fb2 --mode silero --play

  # Озвучить книгу через gTTS и сразу проигрывать
  python3 fb2_reader.py book.fb2 --mode online --play

  # Офлайн pyttsx3-режим, начиная с 3-й главы
  python3 fb2_reader.py book.fb2 --mode offline --start 3

  # Просто посмотреть список глав, ничего не озвучивая
  python3 fb2_reader.py book.fb2 --list

Доступные голоса Silero (--speaker): aidar (муж.), baya (жен.),
kseniya (жен.), xenia (жен.), eugene (муж.), random (случайный на каждой фразе).
