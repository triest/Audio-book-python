r"""Reproducible stress benchmark: raw silero-stress (baseline) vs the project's full chain.

Run:  .venv\Scripts\python.exe stress_benchmark.py [--json out.json]

Scoring (denominator = target words in stress_gold.json that have ONE canonical reading; cases
marked "optional" are reported separately). A target is correct when the output contains the
expected form (or an `any` alternative) as a whole word. Also reported: text preservation
(output minus '+' and е/ё must equal the input), determinism (two runs identical), time/10k words.
Gold set is hand-written and small - treat percentages as a regression gauge, not a population estimate."""
import json
import re
import sys
import time
import ctypes
from ctypes import wintypes
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_reader as fr  # noqa: E402

GOLD = json.loads((Path(__file__).resolve().parent / "data" / "stress_gold.json").read_text(encoding="utf-8"))
ACC = fr._load_silero_stress()
if ACC is None:
    sys.exit("silero-stress is required")


def norm(s):
    return s.replace("+", "").replace("ё", "е").replace("Ё", "Е")


def has(out, form):
    return re.search(r"(?<![А-Яа-яЁё+])" + re.escape(form) + r"(?![А-Яа-яЁё+])", out, re.I) is not None


def sys_raw(text, forced=None):
    return ACC(text, put_stress=True, put_stress_homo=True, put_yo=True, put_yo_homo=True)


def sys_chain(text, forced=None):
    fr.reset_stress_context()
    prot, ph = fr.apply_stress_dictionary_protected(text, dict(forced or {}))
    out = fr.apply_stress_layers(prot, "silero", ACC)
    return fr.restore_stress_placeholders(out, ph)


SYSTEMS = {"baseline raw silero-stress": sys_raw, "full chain (dict+gate+sanitizer)": sys_chain}

# Optional candidate: kubataba ruaccent-predictor (PyPI "ruaccent-predictor"; installs a top-level package named
# "ruaccent" that CLASHES with Den4ikAI's ruaccent). Evaluate it WITHOUT touching the venv:
#   pip install --no-deps --target <dir> ruaccent-predictor==1.2.0   and   set RUACCENT_PREDICTOR_DIR=<dir>
import os  # noqa: E402
if os.environ.get("RUACCENT_PREDICTOR_DIR"):
    sys.path.insert(0, os.environ["RUACCENT_PREDICTOR_DIR"])
    import io as _io, contextlib as _cl  # noqa: E402
    from ruaccent import load_accentor as _rp_load  # noqa: E402
    with _cl.redirect_stdout(_io.StringIO()):
        _RP = _rp_load(device="cpu")

    def sys_rp(text, forced=None):
        return _RP(text, format="synthesis")
    SYSTEMS["candidate ruaccent-predictor 1.2.0 (raw)"] = sys_rp


def score(fn, cases, use_forced=False):
    per = defaultdict(lambda: [0, 0])
    opt = [0, 0]
    preserved = 0
    misses = []
    for c in cases:
        out = fn(c["text"], c.get("forced") if use_forced else None)
        preserved += norm(out) == norm(c["text"])
        for t in c["targets"]:
            ok = any(has(out, f) for f in [t["form"]] + t.get("any", []))
            if c.get("optional"):
                opt[0] += ok; opt[1] += 1
                continue
            per[c.get("cat", "dict")][0] += ok
            per[c.get("cat", "dict")][1] += 1
            if not ok:
                misses.append((c["text"], t["form"], out))
    return per, opt, preserved, misses


def wilson(k, n, z=1.96):
    """95% Wilson interval for k hits of n. With ~68 gold targets it is wide (about +-5..8 points),
    so a change smaller than the interval is noise, not an improvement."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def overlap(a, b):
    """True when two (lo, hi) intervals overlap, i.e. the two systems are not distinguishable."""
    return a[0] <= b[1] and b[0] <= a[1]


def peak_rss_mb():
    """Peak working set of this process (Windows); 0 elsewhere. tracemalloc cannot see torch's allocations."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = wintypes.HANDLE
        gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
        gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        gpmi(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e6
    except Exception:
        return 0.0


def make_text(n_words):
    base = " ".join(c["text"] for c in GOLD["cases"])
    words = base.split()
    reps = n_words // len(words) + 1
    return " ".join(words * reps).split()[:n_words]


def verify_gold():
    """Cross-check gold targets against the independent offline corpus (Wiktionary-derived,
    stress_corpus_ru.tsv.gz). Words with several readings in the corpus are 'ambiguous' (reported,
    homographs are expected there); a gold form that is absent from a single-reading entry is a gold ERROR."""
    import gzip
    corpus = defaultdict(set)
    with gzip.open(Path(__file__).with_name("stress_corpus_ru.tsv.gz"), "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                corpus[parts[0].lower().replace("ё", "е")].add(parts[1].lower().replace("^", "+"))
    bad = 0
    for c in GOLD["cases"] + GOLD["dict_cases"]:
        for t in c["targets"]:
            if t.get("homograph") or t.get("dictonly"):
                continue   # context-dependent / user-dictionary targets: the corpus has no per-context readings
            form = t["form"].lower()
            plain = form.replace("+", "").replace("ё", "е")
            readings = {r.replace("ё", "е") for r in corpus.get(plain, ())}
            if len(plain) == 0 or not readings:
                print(f"  ?? not in corpus: {form}")
                continue
            ok_forms = {f.lower().replace("ё", "е") for f in [t["form"]] + t.get("any", [])}
            if not (readings & ok_forms):
                bad += 1
                print(f"  GOLD CONFLICT {form!r}: corpus says {sorted(readings)}  [{c['text']}]")
    print(f"gold conflicts with corpus: {bad}")
    return bad


def main():
    if "--verify-gold" in sys.argv:
        sys.exit(1 if verify_gold() else 0)
    report = {}
    cis = {}
    for name, fn in SYSTEMS.items():
        per, opt, kept, misses = score(fn, GOLD["cases"])
        tot_ok = sum(v[0] for v in per.values()); tot = sum(v[1] for v in per.values())
        print(f"\n== {name} ==")
        for cat, (ok, n) in sorted(per.items()):
            print(f"  {cat:10s} {ok:3d}/{n:<3d} {100*ok/n:5.1f}%")
        lo, hi = wilson(tot_ok, tot)
        print(f"  TOTAL      {tot_ok}/{tot} {100*tot_ok/tot:.1f}%  [95% CI {100*lo:.1f}-{100*hi:.1f}]   optional(names etc): {opt[0]}/{opt[1]}")
        print(f"  text preserved modulo stress: {kept}/{len(GOLD['cases'])}")
        for m in misses:
            print(f"  MISS {m[1]!r} in {m[0]!r} -> {m[2]!r}")
        a = [fn(c["text"]) for c in GOLD["cases"]]; b = [fn(c["text"]) for c in GOLD["cases"]]
        print(f"  deterministic (2 runs identical): {a == b}")
        cis[name] = wilson(tot_ok, tot)
        report[name] = {"per_cat": {k: v for k, v in per.items()}, "total": [tot_ok, tot], "preserved": kept,
                        "deterministic": a == b}
    names = list(cis)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = overlap(cis[names[i]], cis[names[j]])
            print(f"\n{names[i]} vs {names[j]}: " + ("intervals overlap -> NOT distinguishable on this gold set" if same
                  else "intervals do not overlap -> difference is real"))
    per, opt, kept, misses = score(sys_chain, GOLD["dict_cases"], use_forced=True)
    ok = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
    print(f"\n== dictionary cases (chain with forced dict) == {ok}/{n}, preserved {kept}/{len(GOLD['dict_cases'])}")
    for m in misses:
        print(f"  MISS {m[1]!r} in {m[0]!r} -> {m[2]!r}")
    report["dict"] = [ok, n]
    # speed / memory on ~10k words, paragraph-sized chunks like the real pipeline
    words = make_text(10000)
    paras = [" ".join(words[i:i + 60]) for i in range(0, len(words), 60)]
    for name, fn in SYSTEMS.items():
        t = time.perf_counter()
        for p in paras:
            fn(p)
        dt = time.perf_counter() - t
        peak = peak_rss_mb()
        print(f"\n{name}: {dt:.1f}s per 10,000 words, process peak working set (cumulative) {peak:.0f} MB")
        report[name]["sec_per_10k_words"] = round(dt, 1)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
