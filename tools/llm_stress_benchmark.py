"""Замер встроенной (или любой локальной) LLM на выборе ударения: `.venv\\Scripts\\python.exe tools\\llm_stress_benchmark.py [--backend builtin|lmstudio|ollama] [--model M] [--limit N] [--oracle]`.

Две выборки (обе с известным верным ответом, написаны людьми, не моделью):
  * gold   — `data/stress_gold.json` (слова с «+» в `targets`, текст кейса как контекст);
  * dict   — одиночные слова `forced_stress_words.json` (исправления ошибок цепочки) с предложением из книги.
Кандидаты строятся как в приложении (`stress_review.candidates_for`: по одному на гласную). Для каждого слова модель выбирает номер;
считаем верно / неверно / «не уверен» (0). Важна ДОЛЯ НЕВЕРНЫХ: «не уверен» безвреден (решает слух), неверный выбор вводит в заблуждение.
`--oracle` — самопроверка тракта без модели (ответ берётся из эталона, должно быть 100%).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stress_llm as sl  # noqa: E402
import stress_review as srv  # noqa: E402

BOOK = ROOT / "pepel-pobedy.fb2"


def _sentence(text: str, pos: int, end: int) -> str:
    a = max(text.rfind(c, 0, pos) for c in ".!?…\n") + 1
    nxt = [text.find(c, end) for c in ".!?…\n"]
    b = min((x for x in nxt if x != -1), default=len(text)) + 1
    return re.sub(r"\s+", " ", text[a:b]).strip()[:260]


def gold_cases():
    d = json.loads((ROOT / "data" / "stress_gold.json").read_text(encoding="utf-8"))
    out = []
    for c in d["cases"]:
        for t in c["targets"]:
            form = t["form"].lower()
            out.append({"src": "gold:" + c.get("cat", ""), "word": form.replace("+", ""), "truth": {form, *(x.lower() for x in t.get("any", []))},
                        "sentence": c["text"]})
    return out


def dict_cases(limit_per_word: int = 1):
    if not BOOK.exists():
        return []
    book = re.sub(r"<[^>]+>", " ", BOOK.read_text(encoding="utf-8", errors="ignore"))
    d = json.loads((ROOT / "forced_stress_words.json").read_text(encoding="utf-8"))
    out = []
    for k, v in d.items():
        if k.startswith("_") or " " in k or "+" not in v or "-" in k or len(k) < 4:
            continue
        m = re.search(r"(?<![А-Яа-яЁё])" + re.escape(k) + r"(?![А-Яа-яЁё])", book, re.I)
        if not m:
            continue
        out.append({"src": "dict", "word": k.lower(), "truth": {v.lower()}, "sentence": _sentence(book, m.start(), m.end())})
    return out


def run(cases, backend, model, host, oracle):
    res = {"ok": 0, "bad": 0, "unsure": 0, "skipped": 0}
    bad = []
    t0 = time.time()
    for c in cases:
        cands = srv.candidates_for(c["word"], None)
        if len(cands) < 2 or not (c["truth"] & set(cands)):
            res["skipped"] += 1
            continue
        q = {"word": c["word"], "reading": cands[0], "candidates": cands, "occ": [{"sentence": c["sentence"]}]}
        if oracle:
            chat = lambda p, m, h, c=c, cands=cands: json.dumps({"answer": cands.index(sorted(c["truth"] & set(cands))[0]) + 1})
            r = sl.judge_moment(q, backend, model, host, chat=chat)
        else:
            try:
                r = sl.judge_moment(q, backend, model, host)
            except Exception as e:
                print("ОШИБКА:", e)
                return res, bad
        if r["form"] is None:
            res["unsure"] += 1
        elif r["form"] in c["truth"]:
            res["ok"] += 1
        else:
            res["bad"] += 1
            bad.append((c["src"], c["word"], r["form"], sorted(c["truth"])[0], r["reason"]))
    res["seconds"] = round(time.time() - t0, 1)
    return res, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="builtin", choices=list(sl.BACKENDS.values()))
    ap.add_argument("--model", default="")
    ap.add_argument("--host", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    for name, cases in (("gold", gold_cases()), ("dict", dict_cases())):
        if a.limit:
            cases = cases[:a.limit]
        res, bad = run(cases, a.backend, a.model, a.host, a.oracle)
        n = res["ok"] + res["bad"] + res["unsure"]
        print(f"[{name}] кейсов {n} (пропущено {res['skipped']}): верно {res['ok']}, НЕВЕРНО {res['bad']}, не уверен {res['unsure']}; "
              f"{res.get('seconds', 0)} с", file=out)
        for b in bad[:40]:
            print("   ", " | ".join(map(str, b)), file=out)


if __name__ == "__main__":
    main()
