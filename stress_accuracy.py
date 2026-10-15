r"""Token-level stress accuracy on annotated chapters, with an explicit, reproducible definition.
  .venv\Scripts\python.exe stress_accuracy.py TESTDIR CH_FROM CH_TO [--list N]
ELIGIBLE token = word with >=2 vowels, no hyphen, whose plain form has EXACTLY ONE reading in the independent corpus
(stress_corpus_ru.tsv.gz, Wiktionary-derived; contains systematic errors, so corpus agreement is a LOWER bound).
 strict      = marked form == corpus reading
 adjudicated = strict + disagreements that stress_adjudication.json rules 'chain' (the chain is right); 'corpus' = chain wrong;
               'skip' = context-dependent/uncertain -> excluded from the denominator and reported.
Adjudication verdicts are written by a person/assistant reading the context - NOT by the model under test - and should be audited."""
import gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
T = Path(sys.argv[1]); a, b = int(sys.argv[2]), int(sys.argv[3]); LIST = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 0
corpus = defaultdict(set)
for l in gzip.open(Path(__file__).with_name("stress_corpus_ru.tsv.gz"), "rt", encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 2: corpus[p[0].lower().replace("ё", "е")].add(p[1].lower().replace("^", "+").replace("ё", "е"))
adj_path = Path(__file__).with_name("stress_adjudication.json")
adj = json.loads(adj_path.read_text(encoding="utf-8")) if adj_path.exists() else {}
W = re.compile(r"[А-Яа-яЁё+]+")
tot = ok = 0; bad = Counter(); skipped = 0; wrong = 0; right = 0; undecided = Counter()
for ch in range(a, b + 1):
    for m in W.finditer((T / "text" / f"ch{ch}_B_annotated.txt").read_text(encoding="utf-8")):
        w = m.group(0).lower().replace("ё", "е"); plain = w.replace("+", "")
        if len(re.findall("[аеиоуыэюя]", plain)) < 2: continue
        r = corpus.get(plain)
        if not r or len(r) != 1: continue
        tot += 1
        if w in r: ok += 1; continue
        v = adj.get(w)
        if v == "chain": right += 1
        elif v == "corpus": wrong += 1
        elif v == "skip": skipped += 1
        else: undecided[w] += 1
und = sum(undecided.values())
print(f"chapters {a}-{b}: eligible tokens {tot}")
print(f"strict (corpus agreement, lower bound): {ok}/{tot} = {ok/tot:.2%}")
den = tot - skipped - und
print(f"adjudicated: right {ok+right} / decided {den} = {(ok+right)/den:.2%}   [chain-correct disagreements {right}, chain-wrong {wrong}, skipped(context) {skipped}, UNDECIDED {und} tokens / {len(undecided)} types]")
cons = (ok + right) / (tot - skipped)
print(f"conservative (undecided counted as WRONG, context-dependent skipped): {ok+right}/{tot-skipped} = {cons:.2%}")
if LIST:
    for w, n in undecided.most_common(LIST): print(f"  {n:3d} {w}  corpus={sorted(corpus[w.replace('+','')])}")
