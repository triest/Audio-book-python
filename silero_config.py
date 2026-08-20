#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Конфигурация и загрузка моделей Silero TTS."""

from __future__ import annotations

import re

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
    "v5_3_ru": {
        "title": "Silero v5.3 (ударения, омографы, вопросы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_3_ru.pt",
        "local_file": "silero_model_v5_3_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene"),
    },
    "v5_2_ru": {
        "title": "Silero v5.2 (ударения, омографы, вопросы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_2_ru.pt",
        "local_file": "silero_model_v5_2_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene"),
    },
    "v5_1_ru": {
        "title": "Silero v5.1 (ударения, омографы, вопросы)",
        "package_url": "https://models.silero.ai/models/tts/ru/v5_1_ru.pt",
        "local_file": "silero_model_v5_1_ru.pt",
        "speakers": ("aidar", "baya", "kseniya", "xenia", "eugene"),
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


MODELS_YML_URL = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"


def _find_ru_tts_block(yml_text: str) -> str:
    """Вырезает из models.yml раздел tts_models: -> ru: (до следующего
    ключа того же уровня отступа, что и 'ru:')."""
    m = re.search(r"^tts_models:\s*\n(.*)", yml_text, re.S | re.M)
    if not m:
        return ""
    rest = m.group(1)
    m2 = re.search(r"^  ru:\s*\n(.*)", rest, re.S | re.M)
    if not m2:
        return ""
    block = m2.group(1)
    # обрезаем на следующем ключе с отступом в 2 пробела (например "  en:")
    m3 = re.search(r"^  \S", block, re.M)
    return block[: m3.start()] if m3 else block


def check_for_model_updates(timeout: float = 8.0) -> dict:
    """Проверяет на GitHub (models.yml проекта silero-models), появились ли
    новые версии основной многоголосой русской модели (v5_x_ru и новее),
    которых ещё нет в SILERO_MODELS. Ничего не скачивает и не изменяет —
    только сообщает.

    Возвращает {"ok": True, "new_models": [id, ...], "checked": [id, ...]}
    либо {"ok": False, "error": "..."} при сетевой ошибке.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(MODELS_YML_URL, headers={"User-Agent": "fb2_reader"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            yml_text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"ok": False, "error": str(e)}

    ru_block = _find_ru_tts_block(yml_text)
    if not ru_block:
        return {"ok": False, "error": "не удалось разобрать models.yml (изменился формат)"}

    # top-level id внутри блока ru: — строки вида "    v5_5_ru:" (4 пробела)
    ids = re.findall(r"^    ([A-Za-z0-9_]+):\s*$", ru_block, re.M)
    # интересуют только основные многоголосые модели "vN[_M]_ru"
    main_ids = [i for i in ids if re.fullmatch(r"v\d+(_\d+)?_ru", i)]

    new_models = sorted(set(main_ids) - set(SILERO_MODELS.keys()))
    return {"ok": True, "new_models": new_models, "checked": main_ids}


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
