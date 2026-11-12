r"""Words that ASR recognised in variant X but not in variant Y (same text/voice): a LEAD for listening, not proof of a stress error.
  .venv\Scripts\python.exe asr_variant_diff.py TESTDIR X Y CH [BOOK_IDX]"""
import difflib, json, re, sys
from collections import Counter
from pathlib import Path
T = Path(sys.argv[1]); X, Y, CH = sys.argv[2], sys.argv[3], sys.argv[4]; BI = int(sys.argv[5]) if len(sys.argv) > 5 else int(CH) + 1
norm = lambda w: re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))
src = [norm(w) for w in (T / "text" / f"ch{BI}_plain.txt").read_text(encoding="utf-8").split()]; src = [w for w in src if w][2:]
def matched(v):
    a = [norm(x["w"]) for x in json.load(open(T / "asr" / f"{v}_ch{CH}.json", encoding="utf-8"))["words"]]; a = [w for w in a if w]
    ok = [False] * len(src)
    for b in difflib.SequenceMatcher(None, src, a, autojunk=False).get_matching_blocks():
        for k in range(b.a, b.a + b.size): ok[k] = True
    return ok
mx, my = matched(X), matched(Y)
lost = Counter(src[i] for i in range(len(src)) if mx[i] and not my[i])
gained = Counter(src[i] for i in range(len(src)) if my[i] and not mx[i])
print(f"ch{CH}: recognised in {X} only: {sum(lost.values())} tokens / {len(lost)} types; in {Y} only: {sum(gained.values())} / {len(gained)}")
print(f"  top recognised only in {X}:", lost.most_common(25))
print(f"  top recognised only in {Y}:", gained.most_common(15))
