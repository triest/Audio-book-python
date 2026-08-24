# CosyVoice3 (FunAudioLLM/CosyVoice, Fun-CosyVoice3-0.5B) - REST-сервис
# клонирования голоса, ЗАПУСКАЕТСЯ В КОНТЕЙНЕРЕ (не на голом Windows),
# т.к. у CosyVoice жёстко закреплены версии зависимостей (torch==2.3.1 и
# т.д.), несовместимые с тем, что уже стоит в .venv_cosyvoice для
# F5-TTS/XTTS/ESpeech (см. cosyvoice3_rest_service.py в корне репозитория
# для подробностей).
#
# Требует Docker Desktop с WSL2-бэкендом и поддержкой NVIDIA GPU (NVIDIA
# Container Toolkit) - без неё контейнер будет работать на CPU (заметно
# медленнее). Собирается и запускается через docker-compose.yml в этой же
# папке (см. start_cosyvoice3.bat / stop_cosyvoice3.bat в корне проекта).

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# git - для клонирования репозитория CosyVoice (включая сабмодуль
# Matcha-TTS); остальное - обычные системные библиотеки, которые нужны
# librosa/soundfile/pyworld при сборке из исходников (в этом базовом
# образе их по умолчанию нет).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# --recurse-submodules - обязательно: third_party/Matcha-TTS не входит в
# сам репозиторий CosyVoice, а как раз тот сабмодуль, без которого
# инференс не импортируется (см. sys.path.insert в cosyvoice3_rest_service.py).
RUN git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice

WORKDIR /opt/CosyVoice

# requirements.txt у CosyVoice уже включает fastapi/uvicorn/soundfile/
# openai-whisper - отдельно ставить их не нужно, только num2words (для
# запасного варианта нормализации чисел, см. _numbers_to_words_ru в
# cosyvoice3_rest_service.py) и huggingface_hub (скачивание весов в
# entrypoint.sh).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir num2words huggingface_hub

COPY cosyvoice3_rest_service.py /opt/CosyVoice/cosyvoice3_rest_service.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 5012

ENTRYPOINT ["/opt/entrypoint.sh"]
