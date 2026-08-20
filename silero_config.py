#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Конфигурация и загрузка моделей Silero TTS."""

from __future__ import annotations

# Последняя русская модель Silero (август 2026): v5_5_ru
DEFAULT_SILERO_MODEL = "v5_5_ru"

SILERO_MODELS: dict[str, dict] = {
    "v5_5_ru": {
        "title": "Silero v5.5 — последняя (ударения, омографы, вопросы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        "local_file": "silero_model_v5_5_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene"),
    },
    "v5_4_ru": {
        "title": "Silero v5.4 (ударения, омографы, вопросы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_4_ru.pt",
        "local_file": "silero_model_v5_4_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia"),
    },
    "v5_ru": {
        "title": "Silero v5.0 (ударения, омографы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_ru.pt",
        "local_file": "silero_model_v5_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene"),
    },
    "v4_ru": {
        "title": "Silero v4 (устаревшая, есть random)",
        "package_url": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
        "local_file": "silero_model_v4_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene", "random"),
    },
}

SPEAKER_LABELS = {
    "aidar": "Айдар (муж.)",
    "baya": "Бая (жен.)",
    "kseniya": "Ксения-alt (жен.)",
    "xenia": "Ксения (жен.)",
    "eugene": "Евгений (муж.)",
    "random": "Случайный (только v4)",
}


def speakers_for_model(model_id: str) -> tuple[str, ...]:
    if model_id not in SILERO_MODELS:
        model_id = DEFAULT_SILERO_MODEL
    return SILERO_MODELS[model_id]["speakers"]


def speaker_choices(model_id: str = DEFAULT_SILERO_MODEL) -> dict[str, str]:
    return {s: SPEAKER_LABELS.get(s, s) for s in speakers_for_model(model_id)}


def load_silero_model(model_id: str = DEFAULT_SILERO_MODEL, device=None):
    """Загружает модель Silero: сначала pip-пакет silero, иначе torch.hub."""
    if model_id not in SILERO_MODELS:
        raise ValueError(
            f"Неизвестная модель {model_id!r}. "
            f"Доступны: {', '.join(SILERO_MODELS)}"
        )

    import torch

    if device is None:
        device = torch.device("cpu")

    # Официальный способ (pip install silero) — обычно быстрее обновляется
    try:
        from silero import silero_tts

        model, _ = silero_tts(language="ru", speaker=model_id)
        model.to(device)
        print(f"Модель {model_id} загружена (pip silero)")
        return model
    except ImportError:
        pass
    except Exception as e:
        print(f"pip silero не удалось ({e}), пробую torch.hub…")

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=model_id,
    )
    model.to(device)
    print(f"Модель {model_id} загружена (torch.hub)")
    return model


def load_silero_package(model_id: str = DEFAULT_SILERO_MODEL, device=None):
    """Загружает .pt-пакет модели для REST-сервиса (save_wav API)."""
    import os
    import torch

    if model_id not in SILERO_MODELS:
        raise ValueError(f"Неизвестная модель {model_id!r}")

    if device is None:
        device = torch.device("cpu")

    cfg = SILERO_MODELS[model_id]
    local_file = cfg["local_file"]
    if not os.path.isfile(local_file):
        print(f"Скачиваю модель {model_id} ({local_file})…")
        torch.hub.download_url_to_file(cfg["package_url"], local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    return model
