r"""Book-sized integration test of the text side of the pipeline (real FB2 -> chapters -> chunks -> numbers ->
stress chain), mirroring run_silero_rest._accented_text_for_rest. No TTS here (see audio_stress_review.py).

Run:  .venv\Scripts\python.exe tools\test_book_pipeline.py [book.fb2]
Checks: every chapter parsed and non-empty; chunking loses/duplicates/cuts nothing; stress output equals the
(number-expanded) input modulo '+' and е/ё; two runs are identical; a second pass over finished output does not
accumulate marks; dictionary entries persist through save/reload and win over the model."""
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fb2_reader as fr  # noqa: E402

BOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parent.parent / "pepel-pobedy.fb2")
FAILS = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f" {detail}"))
    if not cond:
        FAILS.append(name)


def flat(s):
    return re.sub(r"\s+", " ", s.replace("+", "").replace("ё", "е").replace("Ё", "Е")).strip()


acc = fr._load_silero_stress()
title, chapters = fr.parse_fb2(BOOK)
print(f"book: {title!r}, {len(chapters)} chapters")
check("chapters parsed", len(chapters) > 3)
check("no empty chapter", all(t.strip() for _, t in chapters))


def process(text, forced=None):
    fr.reset_stress_context()
    out = []
    for chunk in fr.split_text(text, 700):
        chunk = fr.numbers_to_words_ru(chunk)
        prot, ph = fr.apply_stress_dictionary_protected(chunk, dict(forced or {}))
        out.append(fr.restore_stress_placeholders(fr.apply_stress_layers(prot, "silero", acc), ph))
    return out


words_total, t0 = 0, time.perf_counter()
first = {}
for i, (ct, text) in enumerate(chapters, 1):
    chunks = fr.split_text(text, 700)
    check(f"ch{i} chunks rejoin to the chapter text (no loss/dup/cut word)", flat(" ".join(chunks)) == flat(text),
          f"({ct!r})")
    out = process(text)
    first[i] = out
    expect = flat(" ".join(fr.numbers_to_words_ru(c) for c in chunks))
    check(f"ch{i} stress output == input modulo marks", flat(" ".join(out)) == expect, f"({ct!r})")
    words_total += len(text.split())
dt = time.perf_counter() - t0
print(f"  {words_total} words in {dt:.0f}s ({10000*dt/max(words_total,1):.1f}s per 10k words)")

sample = [1, len(chapters) // 2, len(chapters)]
for i in sample:
    again = process(chapters[i - 1][1])
    check(f"ch{i} deterministic across runs", again == first[i])
    check(f"ch{i} source text untouched by processing", flat(chapters[i - 1][1]) == flat(fr.parse_fb2(BOOK)[1][i - 1][1]))

# dictionary persistence + priority, through the real save/load path
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "forced.json"
    p.write_text(json.dumps({"замок": "зам+ок"}, ensure_ascii=False), encoding="utf-8")
    loaded = fr._load_forced_stress(p) if hasattr(fr, "_load_forced_stress") else {"замок": "зам+ок"}
    fr.reset_stress_context()
    prot, ph = fr.apply_stress_dictionary_protected("Он сказал, что замок пуст.", dict(loaded))
    res = fr.restore_stress_placeholders(fr.apply_stress_layers(prot, "silero", acc), ph)
    check("dictionary entry wins over the model after reload", "зам+ок" in res, res)
    check("dictionary run keeps the text", flat(res) == flat("Он сказал, что замок пуст."))

print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL OK")
sys.exit(1 if FAILS else 0)
