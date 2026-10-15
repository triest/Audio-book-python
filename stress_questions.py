r"""Ask-the-user workflow for uncertain stress ("if the model does not know, ask").

The accentor does expose a softmax probability, but it is a weak error signal (see CLAUDE.md: 70% of its errors have p>=0.99), and measurements showed that automatic second opinions are WORSE than the
model (the independent corpus disagrees with the chain on ~95 word types in chapters 1-50: the chain is right on 91,
the corpus on 4), so uncertainty is detected by cheap structural signals and the decision is handed to a person:

  name         capitalised word (seen capitalised in mid-sentence) unknown to the offline corpus  -> character/place names
  unknown      lower-case word with >=2 vowels unknown to the corpus (rare terms, coinages)
  inconsistent the SAME spelling got different marks in different places of the book (homograph OR a model slip)

Words already in forced_stress_words.json (single or phrase key) are never asked about again.
Names are grouped by lemma (pymorphy3): one question covers all case forms whose stem contains the stress.

  .venv\Scripts\python.exe stress_questions.py scan  book.fb2 OUTDIR [--top 60] [--speaker xenia]
      -> OUTDIR\questions.json, OUTDIR\index.html (audio for every candidate reading, radio buttons, "Export" button)
  .venv\Scripts\python.exe stress_questions.py apply OUTDIR\decisions.json
      -> writes the chosen forms to forced_stress_words.json (atomic) + provenance in stress_review_log.jsonl

Nothing is written without an explicit decision; "skip" and "model is right" leave the dictionary untouched
(the second is remembered in OUTDIR\decisions.json only)."""
import argparse
import datetime
import gzip
import html
import json
import os
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class _LogPath:
    """stress_review_log.jsonl next to the scripts / the exe (resolved lazily: frozen builds keep user files beside the exe)."""
    def __fspath__(self):
        import fb2_reader as fr
        return str(fr._app_paths.user_file("stress_review_log.jsonl"))


LOG = _LogPath()
VOW = "аеёиоуыэюя"
TOK = re.compile(r"[А-Яа-яЁё+]+")


def plain(w):
    return w.replace("+", "").lower().replace("ё", "е")


def n_vowels(w):
    return sum(ch in VOW for ch in w.lower())


def load_corpus():
    c = defaultdict(set)
    import fb2_reader as fr
    with gzip.open(fr.DEFAULT_OFFLINE_STRESS_CORPUS_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                c[p[0].lower().replace("ё", "е")].add(p[1].lower().replace("^", "+"))
    return c


def dictionary_words(path):
    """plain forms already decided by the user (single keys and every word of phrase keys)."""
    d = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    out = set()
    for k in d:
        if k.startswith("_"):
            continue
        for w in k.split():
            out.add(plain(w))
    return out


def _in_morph_dict(morph, pl):
    """True when pymorphy3 knows the word from its dictionary (regular morphology such as gerunds: the corpus just
    lacks the form, the model handles it). Coinages (манти, хевы) are only guessed by pymorphy3 -> still asked."""
    if morph is None:
        return False
    try:
        return any(type(step[0]).__name__ == "DictionaryAnalyzer" for step in morph.parse(pl)[0].methods_stack)
    except Exception:
        return False


def candidates_for(pl, first=None):
    """one reading per vowel (stress before the vowel); the model's reading first. 'ё' is always stressed."""
    if "ё" in pl:
        return [pl.replace("ё", "+ё")] if "+" not in pl else [pl]
    res = []
    for i, ch in enumerate(pl):
        if ch in VOW:
            res.append(pl[:i] + "+" + pl[i:])
    if first and first in res:
        res.remove(first)
        res.insert(0, first)
    return res


def sentence_around(marked, start, end):
    a = max(marked.rfind(c, 0, start) for c in ".!?…\n")
    a = 0 if a < 0 else a + 1
    bs = [i for i in (marked.find(c, end) for c in ".!?…\n") if i >= 0]
    b = min(bs) + 1 if bs else len(marked)
    s = marked[a:b].strip()
    return s if len(s) <= 260 else marked[max(0, start - 110):end + 110].strip()


def scan(book, top):
    import fb2_reader as fr
    title, chapters = fr.parse_fb2(Path(book))
    qs = scan_chapters(chapters)
    print(f"{title}: {len(qs)} questions ({Counter(q['kind'] for q in qs)}); top {top} go to the page")
    return qs


def _ok_path():
    import fb2_reader as fr
    return Path(fr._app_paths.user_file("stress_questions_ok.json"))


def load_ok_words():
    """words the user marked 'the model is right' (never asked again)."""
    p = _ok_path()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def add_ok_words(words):
    p = _ok_path()
    cur = load_ok_words() | {plain(w) for w in words}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(cur), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def render_sentence(sentence, word, form):
    """sentence with the reading `form` set on every occurrence of `word` (other words keep their marks)."""
    def sub(m):
        if plain(m.group(0)) == word:
            return form[:1].upper() + form[1:] if m.group(0).lstrip("+")[:1].isupper() else form
        return m.group(0)
    return TOK.sub(sub, sentence)


def scan_chapters(chapters, progress_cb=None, cancel=None, skip_words=None):
    """chapters: [(title, text)]. Returns the ranked list of questions (see module docstring). The GUI tab calls this
    in a worker thread; `cancel` is a threading.Event (-> InterruptedError between chapters); `skip_words` are
    plain forms never to ask about (answered 'model is right' earlier)."""
    import fb2_reader as fr
    corpus = load_corpus()
    decided = dictionary_words(fr.DEFAULT_FORCED_STRESS_PATH) | set(skip_words or ())
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
    except Exception:
        morph = None
    forms = defaultdict(Counter)          # plain -> marked(lower) -> count
    caps = Counter(); mid_caps = Counter(); total = Counter()
    ctx = defaultdict(list)               # plain -> [(chapter, marked sentence)]
    for ci, (ct, text) in enumerate(chapters, 1):
        if cancel is not None and cancel.is_set():
            raise InterruptedError
        if progress_cb:
            progress_cb(f"Глава {ci}/{len(chapters)}: {ct}")
        marked, _ = fr.stress_dry_run(text)
        for m in TOK.finditer(marked):
            w = m.group(0)
            pl = plain(w)
            if n_vowels(pl) < 2 or pl in decided or "ё" in w.lower():
                continue
            total[pl] += 1
            forms[pl][w.lower()] += 1
            if w[:1].isupper() or w[:2].replace("+", "")[:1].isupper():
                caps[pl] += 1
                before = marked[:m.start()].rstrip()
                if before and before[-1] not in ".!?…\n«\"—-:" :
                    mid_caps[pl] += 1
            if len(ctx[pl]) < 3:
                ctx[pl].append((ci, sentence_around(marked, m.start(), m.end())))
    qs = {}
    for pl, n in total.items():
        known = corpus.get(pl)
        kind = None
        if not known and mid_caps[pl] >= 1 and caps[pl] >= 0.7 * n:
            kind = "name"
        elif not known and "-" not in pl and caps[pl] < 0.7 * n and not _in_morph_dict(morph, pl):
            kind = "unknown"
        else:
            f = forms[pl]
            if len(f) > 1 and sorted(f.values())[-2] >= 2:
                kind = "inconsistent"
        if kind:
            qs[pl] = {"word": pl, "kind": kind, "count": n, "forms": dict(forms[pl]),
                      "contexts": [{"chapter": c, "sentence": s} for c, s in ctx[pl]]}
    # group names by lemma: one question per lemma, answer propagates to forms sharing the stressed stem
    groups = defaultdict(list)
    for pl, q in qs.items():
        key = pl
        if q["kind"] in ("name", "unknown") and morph:
            key = morph.parse(pl)[0].normal_form.replace("ё", "е")
        groups[(q["kind"], key)].append(pl)
    out = []
    for (kind, key), members in groups.items():
        members.sort(key=lambda p: -qs[p]["count"])
        head = qs[members[0]]
        model_form = max(head["forms"].items(), key=lambda kv: kv[1])[0]
        out.append({"id": len(out) + 1, "kind": kind, "word": head["word"], "group": members,
                    "count": sum(qs[p]["count"] for p in members), "seen_forms": head["forms"],
                    "model_form": model_form,
                    "candidates": candidates_for(head["word"], model_form) if kind != "inconsistent"
                    else list(head["forms"]),
                    "contexts": head["contexts"]})
    out.sort(key=lambda q: -q["count"])
    for i, q in enumerate(out, 1):
        q["id"] = i
    return out


def make_page(out, qs, speaker):
    import numpy as np
    import torch
    imp = torch.package.PackageImporter(str(HERE / "silero_model_v5_5_ru.pt"))
    model = imp.load_pickle("tts_models", "model")
    model.to("cpu")
    out.mkdir(parents=True, exist_ok=True)
    blocks = []
    for q in qs:
        sent = q["contexts"][0]["sentence"] if q["contexts"] else q["word"]
        clips = []
        for k, form in enumerate(q["candidates"], 1):
            text = render_sentence(sent, q["word"], form)
            audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000, put_accent=False, put_yo=False).numpy()
            fn = f"{q['id']:03d}_{k}.wav"
            with wave.open(str(out / fn), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            clips.append((form, fn))
        radios = "".join(
            f"<label><input type=radio name=q{q['id']} value=\"{html.escape(f)}\"> <b>{html.escape(f)}</b></label> "
            f"<audio controls preload=none src='{fn}'></audio><br>" for f, fn in clips)
        radios += (f"<label><input type=radio name=q{q['id']} value=__model__> модель права / не менять</label> "
                   f"<label><input type=radio name=q{q['id']} value=__skip__ checked> пропустить</label>")
        others = ", ".join(q["group"][1:6])
        warn = ("<p style='color:#a00'>Слово встречается с разными ударениями: возможно, это омограф. Запись в словарь "
                "применится ко ВСЕМ вхождениям; отвечайте, только если слово точно читается одним способом.</p>"
                if q["kind"] == "inconsistent" else "")
        blocks.append(
            f"<section data-id={q['id']} data-word=\"{html.escape(q['word'])}\" data-group=\"{html.escape(' '.join(q['group']))}\">"
            f"<h3>{q['id']}. {html.escape(q['word'])} <small>[{q['kind']}, {q['count']}×"
            f"{', формы: ' + html.escape(others) if others else ''}]</small></h3>"
            f"<p><i>{html.escape(sent)}</i></p>{warn}{radios}</section>")
    page = ("<!doctype html><meta charset=utf-8><title>Вопросы по ударению</title>"
            "<style>body{font:15px sans-serif;max-width:860px;margin:2em auto}section{border-top:1px solid #aaa;padding:.6em 0}"
            "small{color:#666}</style><h2>Вопросы по ударению</h2>"
            "<p>Для каждого слова послушайте варианты и выберите верный. «Экспорт» скачивает decisions.json; положите его в эту "
            "папку и выполните <code>stress_questions.py apply decisions.json</code>.</p>"
            + "".join(blocks) +
            "<p><button onclick=ex()>Экспорт decisions.json</button></p><script>function ex(){var d=[];"
            "document.querySelectorAll('section').forEach(function(s){var r=s.querySelector('input:checked');"
            "if(r&&r.value!='__skip__')d.push({id:+s.dataset.id,word:s.dataset.word,group:s.dataset.group.split(' '),choice:r.value});});"
            "var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,1)],{type:'application/json'}));"
            "a.download='decisions.json';a.click();}</script>")
    (out / "index.html").write_text(page, encoding="utf-8")


def propagate(chosen, group, seen):
    """chosen: marked form of group[0]; other case forms get the same stress when the stressed vowel lies inside
    the stem they share with the chosen form. Returns {plain_form: marked_form}."""
    head = chosen.replace("+", "")
    idx = chosen.index("+")                       # stressed vowel index in the plain word
    res = {head: chosen}
    for w in group[1:]:
        L = 0
        while L < min(len(w), len(head)) and w[L] == head[L]:
            L += 1
        if idx < L and idx < len(w):
            res[w] = w[:idx] + "+" + w[idx:]
    return res


def answer_forms(question, chosen):
    """{plain_form: marked_form} for the chosen reading of a question (spreads to the case forms of a name group)."""
    c = chosen.lower()
    group = question.get("group") or [plain(c)]
    return propagate(c, group, None) if len(group) > 1 else {plain(c): c}


def log_answer(pl, form, previous, source):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.datetime.now().isoformat(timespec="seconds"), "word": pl, "form": form,
                            "previous": previous, "source": source}, ensure_ascii=False) + chr(10))


def apply(path):
    import fb2_reader as fr
    dec = json.loads(Path(path).read_text(encoding="utf-8"))
    dpath = Path(fr.DEFAULT_FORCED_STRESS_PATH)
    data = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else {}
    n = 0
    for d in dec:
        c = d["choice"]
        if c.startswith("__") or "+" not in c:
            continue
        forms = propagate(c.lower(), d.get("group") or [plain(c)], None) if len(d.get("group", [])) > 1 else {plain(c): c.lower()}
        for pl, form in forms.items():
            prev = data.get(pl)
            data[pl] = form
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"date": datetime.datetime.now().isoformat(timespec="seconds"), "word": pl, "form": form,
                                    "previous": prev, "source": "stress_questions (user listened)"}, ensure_ascii=False) + "\n")
            n += 1
            print(f"{pl} -> {form}" + (f"   (was {prev})" if prev else ""))
    tmp = dpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dpath)
    print(f"{n} entries written to {dpath}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("book"); s.add_argument("out")
    s.add_argument("--top", type=int, default=60); s.add_argument("--speaker", default="xenia")
    a = sub.add_parser("apply"); a.add_argument("decisions")
    args = ap.parse_args()
    if args.cmd == "scan":
        qs = scan(args.book, args.top)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "questions.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
        make_page(out, qs[:args.top], args.speaker)
        print(f"page: {out / 'index.html'}")
    else:
        apply(args.decisions)


if __name__ == "__main__":
    main()
