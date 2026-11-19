"""Diagnostic postcheck for text after silero-stress."""
from __future__ import annotations
from collections import defaultdict
import re
import json
import os
from pathlib import Path

VOWELS = "аеёиоуыэюя"
TOKEN_RE = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")

def plain(word: str) -> str:
    return word.replace("+", "").lower().replace("ё", "е")

def _reading(word: str) -> str:
    return word.lower().replace("ё", "е")

def _structural_reasons(word: str) -> list[str]:
    raw = word.replace("+", "")
    vowels = sum(ch.lower() in VOWELS for ch in raw)
    marks = word.count("+")
    reasons = []
    if marks > 1:
        reasons.append("multiple_stress_marks")
    if marks and not any(word[i + 1].lower() in VOWELS for i, ch in enumerate(word[:-1]) if ch == "+"):
        reasons.append("stress_mark_not_before_vowel")
    if vowels >= 2 and marks == 0 and "ё" not in raw.lower():
        reasons.append("missing_stress")
    return reasons

def check_text(text: str, *, corpus=None, max_items: int = 500) -> list[dict]:
    forms = defaultdict(set)
    counts = defaultdict(int)
    examples = {}
    reasons = defaultdict(set)
    for match in TOKEN_RE.finditer(text or ""):
        word = match.group(0)
        key = plain(word)
        if not key:
            continue
        counts[key] += 1
        forms[key].add(_reading(word))
        examples.setdefault(key, word)
        reasons[key].update(_structural_reasons(word))
    for key, readings in forms.items():
        if len(readings) > 1:
            reasons[key].add("unstable_reading")
    if corpus is None:
        try:
            import stress_questions
            corpus = stress_questions.load_corpus()
        except Exception:
            corpus = {}
    try:
        from stress_review import reading_conflicts
    except Exception:
        reading_conflicts = None
    if reading_conflicts and corpus:
        for key, readings in forms.items():
            known = corpus.get(key)
            if known and any(reading_conflicts(r, known) for r in readings):
                reasons[key].add("corpus_conflict")
    out = []
    for key, why in reasons.items():
        if why:
            readings = sorted(forms[key])
            try:
                from stress_review import candidates_for
                candidates = candidates_for(key, readings[0] if readings else None)
            except Exception:
                candidates = readings
            risk = "high" if len(why) >= 2 else "medium"
            out.append({"word": key, "readings": readings, "candidates": candidates,
                        "count": counts[key], "example": examples[key],
                        "reasons": sorted(why), "risk": risk, "decision": "review"})
    out.sort(key=lambda item: (-len(item["reasons"]), -item["count"], item["word"]))
    return out[:max_items]

def save_report(path, *, text: str, items: list[dict], source: str = "stress_dry_run") -> str:
    """Atomically save a review report. The report is diagnostic and never changes synthesis input."""
    p = Path(path)
    data = {"version": 1, "source": source, "items": items,
            "summary": {"items": len(items), "high": sum(x.get("risk") == "high" for x in items),
                        "medium": sum(x.get("risk") == "medium" for x in items)}}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return str(p)
