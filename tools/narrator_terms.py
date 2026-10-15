r"""What the old audiobooks' narrator ASR transcripts contain for acronyms and names (Whisper writes spelled-out acronyms in capitals).
  python narrator_terms.py ASR_DIR"""
import json, glob, re, sys, collections
files = sorted(glob.glob(sys.argv[1] + "/*.json"))
allw = []
for f in files:
    allw += [w["w"] for w in json.load(open(f, encoding="utf-8"))["words"]]
print(len(files), "files,", len(allw), "words")
caps = collections.Counter(); low = collections.Counter()
for w in allw:
    t = re.sub(r"[^А-Яа-яЁё0-9-]", "", w)
    if re.fullmatch(r"[А-ЯЁ]{2,6}(-\d+)?", t): caps[t] += 1
    if t: low[t.lower().replace("ё", "е")] += 1
print("UPPERCASE tokens (narrator spells acronyms):", caps.most_common(40))
for k in ["лак", "лаки", "лаков", "рэб", "руф", "бец", "бицы", "мид", "бгб", "кфм", "втк", "екф", "гкф", "мвб", "нрх", "утк", "нлак", "ниокр", "макквин", "мак-квин", "макгиннес", "лафолле", "лафоле", "хонор", "харрингтон", "нимиц", "грейсон", "мантикора", "тремэйн", "хенке", "кромарти", "трумэн", "пьер", "сен-жюст", "протектор", "супердредноут", "супердредноуты"]:
    if low.get(k): print(f"  {k}: {low[k]}")
