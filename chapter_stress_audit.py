r"""Text-level stress audit of chapters: exact text used per variant + contradictions with the independent
offline corpus. Usage: .venv\Scripts\python.exe chapter_stress_audit.py OUTDIR CH [CH...]   (1-based, preface counts)
Writes OUTDIR/text/chN_plain.txt, chN_B_annotated.txt and OUTDIR/audit_chN.json."""
import gzip, json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb2_reader as fr

out = Path(sys.argv[1]); (out / "text").mkdir(parents=True, exist_ok=True)
acc = fr._load_silero_stress()
corpus = defaultdict(set)
for l in gzip.open(Path(__file__).with_name("stress_corpus_ru.tsv.gz"), "rt", encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 2: corpus[p[0].lower().replace("ё", "е")].add(p[1].lower().replace("^", "+"))
stress_dict = fr.load_stress_dictionary(fr.DEFAULT_FORCED_STRESS_PATH) if hasattr(fr, "load_stress_dictionary") else {}
title, chapters = fr.parse_fb2(Path(__file__).with_name("pepel-pobedy.fb2"))
W = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")
for ch in map(int, sys.argv[2:]):
    name, text = chapters[ch - 1]
    fr.reset_stress_context(); annotated = []
    for chunk in fr.split_text(text, 700):
        chunk = fr.numbers_to_words_ru(chunk)
        prot, ph = fr.apply_stress_dictionary_protected(chunk, dict(stress_dict))
        annotated.append(fr.restore_stress_placeholders(fr.apply_stress_layers(prot, "silero", acc), ph))
    ann = "\n".join(annotated)
    (out / "text" / f"ch{ch}_plain.txt").write_text(text, encoding="utf-8")
    (out / "text" / f"ch{ch}_B_annotated.txt").write_text(ann, encoding="utf-8")
    contradictions, unmarked, seen = [], defaultdict(int), set()
    for m in W.finditer(ann):
        w = m.group(0); plain = w.replace("+", "").lower().replace("ё", "е")
        vowels = len(re.findall("[аеёиоуыэюя]", plain))
        if vowels < 2 or "-" in w: continue
        readings = corpus.get(plain)
        if "+" not in w and "ё" not in w.lower():
            unmarked[plain] += 1; continue
        if readings and len(readings) == 1 and w.lower().replace("ё", "е") not in {r.replace("ё", "е") for r in readings}:
            key = (plain, w.lower())
            if key in seen: continue
            seen.add(key)
            ctx = ann[max(0, m.start() - 50): m.end() + 40].replace("\n", " ")
            contradictions.append({"word": w, "corpus": sorted(readings), "context": ctx})
    json.dump({"chapter": ch, "title": name, "words": len(text.split()),
               "contradictions_with_corpus": contradictions,
               "unmarked_multisyllable_top": sorted(unmarked.items(), key=lambda kv: -kv[1])[:40]},
              open(out / f"audit_ch{ch}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ch{ch} {name}: {len(text.split())} words, {len(contradictions)} distinct corpus contradictions, {len(unmarked)} unmarked word types")
