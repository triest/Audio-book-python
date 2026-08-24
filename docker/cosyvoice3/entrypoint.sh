#!/bin/bash
# Скачивает веса Fun-CosyVoice3-0.5B при первом запуске контейнера (кэш
# лежит в volume pretrained_models/, см. docker-compose.yml - переживает
# пересборку/перезапуск контейнера, скачивается один раз) и запускает
# REST-сервис.
set -e

MODEL_DIR="/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Скачиваю веса Fun-CosyVoice3-0.5B (один раз, ~1-2 ГБ)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
"
fi

cd /opt/CosyVoice
exec python3 cosyvoice3_rest_service.py
