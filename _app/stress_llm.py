"""Вердикт локальной LLM по одному сомнительному месту (кнопка «Спросить LLM» на вкладке «Проверка ударений»).

Без Tk. Только ЛОКАЛЬНЫЕ модели: встроенная (`llm_rest_service.py` в `.venv_llm`, её ставит install.bat, сервис поднимается сам
при первом вопросе) либо свой LM Studio / Ollama. Приложение остаётся в режиме «100% локально», облако не используется, а глобальный
выключатель `fb2_reader.LLM_FEATURES_ENABLED` (атрибуция, эмоции) не трогается. LM Studio / Ollama пользователь запускает сам.

Модель не придумывает ударение, а ВЫБИРАЕТ НОМЕР из готовых вариантов (так надёжнее, чем генерация) и может ответить 0 —
«не уверен». Это подсказка: выбранный вариант подставляется в форму, запись в словарь делает пользователь.
"""
from __future__ import annotations

import json
import re

BACKENDS = {"Встроенная": "builtin", "LM Studio": "lmstudio", "Ollama": "ollama"}
BUILTIN_PORT = 5013
BUILTIN_URL = f"http://127.0.0.1:{BUILTIN_PORT}"
_builtin_proc = None


def _builtin_alive() -> bool:
    import requests
    try:
        return requests.get(BUILTIN_URL + "/health", timeout=1.5).ok
    except Exception:
        return False


def ensure_builtin_service(timeout: float = 90.0) -> None:
    """Поднимает встроенный LLM-сервис (если не запущен) и ждёт /health. Модель грузится позже, при первом запросе."""
    global _builtin_proc
    if _builtin_alive():
        return
    import atexit
    import subprocess
    import sys
    from pathlib import Path
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    py = root / ".venv_llm" / "Scripts" / "python.exe"
    script = root / "llm_rest_service.py"
    if not py.exists() or not script.exists():
        raise RuntimeError("Встроенная LLM не установлена: запустите install.bat (создаст .venv_llm и скачает модель ~4.7 ГБ).")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    _builtin_proc = subprocess.Popen([str(py), str(script), "--port", str(BUILTIN_PORT)], cwd=str(root), creationflags=flags)
    atexit.register(lambda: _builtin_proc and _builtin_proc.poll() is None and _builtin_proc.terminate())
    import time
    end = time.time() + timeout
    while time.time() < end:
        if _builtin_alive():
            return
        if _builtin_proc.poll() is not None:
            raise RuntimeError(f"Встроенный LLM-сервис завершился (код {_builtin_proc.returncode}). Проверьте установку (install.bat).")
        time.sleep(1.0)
    raise RuntimeError("Встроенный LLM-сервис не ответил за отведённое время.")


def _builtin_chat(prompt: str, model: str, host: str) -> str:
    import requests
    ensure_builtin_service()
    r = requests.post(BUILTIN_URL + "/v1/chat/completions", timeout=600,
                      json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.0})
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]
MAX_CONTEXTS = 3


def build_prompt(q: dict) -> str:
    cands = list(q.get("candidates") or [])
    lines = []
    for o in (q.get("occ") or [])[:MAX_CONTEXTS]:
        s = re.sub(r"\s+", " ", str(o.get("sentence", ""))).replace("+", "").strip()
        if s:
            lines.append(f"- {s[:400]}")
    variants = "\n".join(f"{i}. {c}" for i, c in enumerate(cands, 1))
    return (
        "Ты — эксперт по русской орфоэпии. Нужно выбрать правильное ударение слова в контексте книги.\n"
        "В вариантах знак «+» стоит ПЕРЕД ударной гласной (зам+ок = замо́к, з+амок = за́мок).\n\n"
        f"Слово: {str(q.get('word', ''))}\n"
        f"Сейчас программа читает: {q.get('reading', '')}\n"
        f"Примеры из книги:\n{chr(10).join(lines)}\n\n"
        f"Варианты:\n{variants}\n\n"
        "Выбери НОМЕР правильного варианта для этих примеров. Если не уверен или это имя, ударение которого "
        "нельзя определить по правилам, ответь 0.\n"
        'Ответь СТРОГО одним JSON-объектом без пояснений вне него: {"answer": <номер или 0>, "reason": "<кратко, по-русски>"}'
    )


def parse_reply(raw: str, cands: "list[str]") -> dict:
    """-> {"index": 0..n (0 = не уверен), "form": str | None, "reason": str}"""
    m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
    if not isinstance(obj, dict):
        n = re.search(r"\b(\d+)\b", raw or "")
        obj = {"answer": int(n.group(1)) if n else 0, "reason": (raw or "").strip()[:200]}
    try:
        idx = int(re.sub(r"\D", "", str(obj.get("answer", 0))) or 0)
    except ValueError:
        idx = 0
    if not 1 <= idx <= len(cands):
        idx = 0
    return {"index": idx, "form": cands[idx - 1] if idx else None, "reason": str(obj.get("reason", "")).strip()[:300]}


def judge_moment(q: dict, backend: str, model: str, host: str = "", chat=None) -> dict:
    """Один запрос к локальной LLM. `chat(prompt, model, host)` можно подменить в тестах."""
    cands = list(q.get("candidates") or [])
    if len(cands) < 2:
        return {"index": 0, "form": None, "reason": "нет вариантов для выбора"}
    if backend != "builtin" and not model.strip():
        raise ValueError("Не указана модель: впишите идентификатор загруженной модели LM Studio / имя модели Ollama.")
    if chat is None:
        if backend == "builtin":
            chat = _builtin_chat
        else:
            import fb2_reader
            fn = fb2_reader._lmstudio_chat_request if backend == "lmstudio" else fb2_reader._ollama_chat_request

            def chat(prompt, model, host):
                return fn(prompt, model, host=host, max_tokens=300)
    return parse_reply(chat(build_prompt(q), model.strip(), host.strip()), cands)
