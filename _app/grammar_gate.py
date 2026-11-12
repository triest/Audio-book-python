"""Grammar gate: resolves case/number stress homographs ("земли", "сети", "корпуса", "голоса"...)
with a small neural morphological tagger instead of letting the accent networks guess.

How it works (no phrase lists, no dictionary of sentences):
  1. The sentence is tagged by slovnet's morphology model (Natasha project, CPU, ~25 ms/sentence):
     part of speech + Case + Number for every token.
  2. A surface form that is a known paradigm homograph (table below) is given the stress that belongs
     to the tagged grammatical form:  земли = Sing -> земл+и (gen. sg.), Plur -> з+емли (nom./acc. pl.).
  3. The tagger's number is cross-checked with the modifier right before the noun
     ("подобающие земли" -> the adjective is plural, so the noun is plural too).
  4. Anything that is not a NOUN/PROPN in the tagger's opinion is left to the accent networks.

The table is a PARADIGM table (a stress pattern per lexeme), so one entry covers all contexts.
To teach the gate a new word, add the stem-stressed plural/singular form to one of the lists below.

Switch off: AUDIOBOOK_GRAMMAR_GATE=0.  Models: ./grammar_gate_models (fetch with: fb2_reader fetch-models).
"""
import os
import re
import sys
import threading
import traceback
from pathlib import Path

_VOWELS = "аеёиоуыэюя"

# F-pattern: singular ending-stressed, plural stem-stressed (the surface form is the same: земли).
#   земл+и (gen. sg.)  /  з+емли (nom./acc. pl.)
_F_WORDS = [
    "з+емли", "н+оги", "р+уки", "в+оды", "ст+ены", "г+оры", "ст+ороны", "ц+ены", "сп+ины",
    "д+оски", "д+уши", "г+оловы", "тр+ети", "тр+авы", "в+ойны", "стр+аны", "р+еки", "гл+авы",
    "б+ороды", "ст+аи", "л+ица", "+окна", "п+исьма",
]
# M-pattern: singular stem-stressed, plural ending-stressed.
#   г+олоса (gen. sg.)  /  голос+а (nom./acc. pl.)
_M_WORDS = [
    "г+олоса", "гл+аза", "к+орпуса", "п+ояса", "м+еста", "пр+ава", "сл+ова", "д+ела", "т+ела",
    "г+орода", "б+ерега", "д+ома", "л+еса", "п+оезда", "+острова", "ц+вета", "в+ечера", "п+аспорта",
    "уч+ителя", "+адреса", "н+омера", "д+окторы", "пр+офессоры", "г+ода",
]
# S-pattern: the stress never moves ("сети" is stem-stressed in every case and number).
_S_WORDS = ["с+ети"]

# words whose surface form is also something else (verb/adverb); the gate only touches nouns anyway
_PLAIN = lambda s: s.replace("+", "")


def _end_form(stem_form: str) -> str:
    p = stem_form.replace("+", "")
    for i in range(len(p) - 1, -1, -1):
        if p[i].lower() in _VOWELS:
            return p[:i] + "+" + p[i:]
    return p


def _build_table() -> dict:
    t = {}
    for w in _F_WORDS:
        t[_PLAIN(w)] = ("F", w, _end_form(w))
    for w in _M_WORDS:
        t[_PLAIN(w)] = ("M", w, _end_form(w))
    for w in _S_WORDS:
        t[_PLAIN(w)] = ("S", w, w)
    return t


# plain word -> (pattern, stem-stressed form, end-stressed form)
PARADIGM_TABLE = _build_table()
# Wrong plurals that LOOK like gen. sg. forms must not be touched when the word is a verb etc.:
_NOUN_POS = {"NOUN", "PROPN"}
_MODIFIER_POS = {"ADJ", "DET", "NUM"}

_state = {"tried": False, "tagger": None, "stats": {"changed": 0, "seen": 0}}


def _models_dir() -> Path:
    """AUDIOBOOK_GRAMMAR_DIR > <app dir>/grammar_gate_models > bundled copy (sys._MEIPASS) > sources."""
    import app_paths
    env = os.environ.get("AUDIOBOOK_GRAMMAR_DIR")
    if env:
        return Path(env)
    for base in (app_paths.app_dir(), app_paths.bundle_dir()):      # first folder that really holds the models
        if (base / "grammar_gate_models" / "slovnet_morph_news_v1.tar").exists():
            return base / "grammar_gate_models"
    return app_paths.user_file("grammar_gate_models")


def enabled() -> bool:
    return os.environ.get("AUDIOBOOK_GRAMMAR_GATE", "1") != "0"


_MODEL_URLS = {
    "navec_news_v1_1B_250K_300d_100q.tar": "https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar",
    "slovnet_morph_news_v1.tar": "https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_morph_news_v1.tar",
}


def fetch_models(dest: "Path | None" = None) -> Path:
    """Downloads the two tagger model files (about 29 MB) next to the program (the .exe folder when frozen).
    Called by `fb2_reader fetch-models` and by install.bat."""
    import urllib.request
    import app_paths
    dest = Path(dest) if dest else app_paths.user_file("grammar_gate_models")
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in _MODEL_URLS.items():
        f = dest / name
        if f.exists() and f.stat().st_size > 1_000_000:
            print("ok:", name)
            continue
        print("downloading", name)
        urllib.request.urlretrieve(url, f)
    print("grammar models ->", dest)
    return dest


def _run_guarded(fn, timeout: float, what: str):
    """Runs fn() in a daemon thread. If it does not finish in `timeout` seconds the gate is switched off for
    the rest of the run, the stack of the stuck thread is printed (so the hang is visible in the log) and
    TimeoutError is raised - the pipeline then carries on without the gate instead of freezing."""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except BaseException as e:           # noqa: BLE001 - re-raised in the caller
            box["e"] = e

    t = threading.Thread(target=work, name="grammar-gate", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        _state["disabled"] = True
        frame = sys._current_frames().get(t.ident)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "(no stack)"
        print(f"grammar gate: '{what}' did not finish in {timeout:.0f}s - gate is OFF for this run.\n"
              f"stuck at:\n{stack}")
        raise TimeoutError(what)
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _load():
    """Loads the tagger once; returns it or None (missing packages/models -> the gate is silently off)."""
    if _state["tried"]:
        return _state["tagger"]
    _state["tried"] = True
    try:
        from navec import Navec
        from slovnet import Morph
        d = _models_dir()
        navec_file = d / "navec_news_v1_1B_250K_300d_100q.tar"
        morph_file = d / "slovnet_morph_news_v1.tar"
        if not navec_file.exists() or not morph_file.exists():
            print("grammar gate: models not found (run: fb2_reader fetch-models) - gate is off.")
            return None
        print("grammar gate: loading models ...")

        def _build():
            m = Morph.load(str(morph_file))
            m.navec(Navec.load(str(navec_file)))
            return m
        tagger = _run_guarded(_build, 180, "model load")
        _state["tagger"] = tagger
        print("grammar gate: slovnet morphology tagger loaded.")
    except Exception as e:   # missing slovnet/navec/razdel, broken model file...
        print(f"grammar gate: unavailable ({type(e).__name__}: {e}) - gate is off.")
    return _state["tagger"]


def _decide(word: str, number: str):
    pat, stem_form, end_form = PARADIGM_TABLE[word.lower()]
    if pat == "S":
        return stem_form
    if pat == "F":
        return end_form if number == "Sing" else stem_form
    return stem_form if number == "Sing" else end_form          # "M"


def _match_case(src: str, form: str) -> str:
    """form is lowercase with a '+'; copy the capitalisation of the source word."""
    plain = src.replace("+", "")
    if plain.isupper() and len(plain) > 1:
        return form.upper()
    if plain[:1].isupper():
        k = 1 if form[:1] == "+" else 0       # a vowel-initial stressed form is "+остров...": capitalise the LETTER
        return form[:k] + form[k].upper() + form[k + 1:]
    return form


_NO_GOVERNOR_POS = {"ADV", "CCONJ", "SCONJ", "PART", "PUNCT", "INTJ"}
# adverbs/quantifiers that DO govern a genitive ("не менее трети", "много земли")
_GOVERNING_WORDS = {"менее", "более", "много", "мало", "немного", "немало", "несколько", "сколько", "столько",
                    "около", "свыше", "почти", "больше", "меньше", "довольно", "достаточно"}
_PL_VERB_RE = re.compile(r"(?:ли|лись|ут|ют|ат|ят)$", re.I)


_MOD_PLUR_END = ("ые", "ие", "ыми", "ими")
_MOD_SING_END = ("ого", "его", "ому", "ему", "ой", "ей", "ую", "юю", "ая", "яя", "ое", "ее", "ом", "ем")


def _modifier_number(tok):
    """Number of an agreeing modifier. The ending is more reliable than the tagger for adjectives
    ("вешние воды": the tagger says Gen Sing, the ending -ие says plural)."""
    w = tok.text.lower()
    if tok.pos in ("ADJ", "VERB", "DET") and len(w) > 4:
        if w.endswith(_MOD_PLUR_END):
            return "Plur"
        if w.endswith(_MOD_SING_END):
            return "Sing"
    return tok.feats.get("Number")


def _plural_subject(tagged, i: int) -> bool:
    """True when tagged[i] stands where a subject stands (no governor to its left) and a finite plural
    verb follows in the same sentence."""
    j = i - 1
    while j >= 0 and tagged[j].pos == "PUNCT":
        j -= 1
    if j >= 0 and (tagged[j].pos not in _NO_GOVERNOR_POS or tagged[j].text.lower() in _GOVERNING_WORDS):
        return False
    for k in range(i + 1, len(tagged)):
        t = tagged[k]
        if t.pos in ("VERB", "AUX") and t.feats.get("Number") == "Plur" and t.feats.get("VerbForm") != "Part":
            return True
        if _PL_VERB_RE.search(t.text) and len(t.text) > 4 and t.pos != "PUNCT":
            return True
    return False


_VSE_KEYS = ("все", "всё")


def _vse_form(tagged, i: int):
    """"все" (plural: "все адмиралы", "мы все", "все до единого") vs "всё" (neuter: "всё равно"): the book
    spells both "все", so the tagger's number decides. Returns "все" / "вс+ё" or None to leave the word alone.
    "все, кто ..." is tagged singular by slovnet but is plural: a following "кто" forces "все"."""
    tg = tagged[i]
    if tg.pos not in ("PRON", "DET"):
        return None
    k = i + 1
    while k < len(tagged) and tagged[k].pos == "PUNCT":
        k += 1
    if k < len(tagged) and tagged[k].text.lower() == "кто":
        return "все"
    number = tg.feats.get("Number")
    if number == "Plur":
        return "все"
    if number == "Sing" and tg.feats.get("Gender") == "Neut":
        return "вс+ё"
    return None


def grammar_gate(text: str) -> str:
    """Guarded entry point (see _run_guarded); the work is done by _grammar_gate_impl."""
    if not text or not enabled() or _state.get("disabled"):
        return text
    if _load() is None:                     # loads once (guarded); None = packages/models missing or load hung
        return text
    try:
        return _run_guarded(lambda: _grammar_gate_impl(text), 30 + len(text) / 100.0, "tagging")
    except TimeoutError:
        return text


def _grammar_gate_impl(text: str) -> str:
    """text: already stressed (with '+'), may contain STRESSDICT placeholders. Returns text with the
    homograph words re-stressed according to the tagger; everything else is untouched."""
    if not text or not enabled() or _state.get("disabled"):
        return text
    tagger = _load()
    if tagger is None:
        return text
    try:
        from razdel import sentenize, tokenize
    except Exception:
        return text
    out_lines = []
    for line in text.split("\n"):
        plain = line.replace("+", "")
        if not re.search(r"[А-Яа-яЁё]", plain) or not any(
                re.search(r"(?<![А-Яа-яЁё])" + re.escape(k) + r"(?![А-Яа-яЁё])", plain, re.I) for k in tuple(PARADIGM_TABLE) + _VSE_KEYS):
            out_lines.append(line)
            continue
        # plain index -> marked index
        pmap, pi = [], 0
        for mi, ch in enumerate(line):
            if ch == "+":
                continue
            pmap.append(mi)
        edits = []      # (marked_start, marked_end, new_form)
        for sent in sentenize(plain):
            toks = list(tokenize(sent.text))
            if not toks:
                continue
            words = [t.text for t in toks]
            # the tagger reads a capitalised first word as a proper name ("Города лежали" ->
            # PROPN Gen Sing); lower-case the first alphabetic token of the sentence for tagging
            for k, w in enumerate(words):
                if re.search(r"[А-Яа-яЁё]", w):
                    if w[:1].isupper() and not w.isupper():
                        words[k] = w.lower()
                    break
            # the stress layer may already have turned "все" into "всё"; the book spells both "все", so tag the
            # neutral spelling and let the context (not the earlier guess) decide
            words = ["все" if w.lower() == "всё" and w[:1].islower() else "Все" if w == "Всё" else w for w in words]
            try:
                tagged = tagger(words).tokens
            except Exception:
                continue
            for i, (tk, tg) in enumerate(zip(toks, tagged)):
                key = tk.text.lower()
                if key in _VSE_KEYS:
                    form = _vse_form(tagged, i)
                    if form:
                        a, b = sent.start + tk.start, sent.start + tk.stop
                        if b <= len(pmap):
                            ms, me = pmap[a], pmap[b - 1] + 1
                            if ms > 0 and line[ms - 1] == "+":
                                ms -= 1
                            new = _match_case(line[ms:me], form)
                            if line[ms:me] != new:
                                edits.append((ms, me, new))
                    continue
                if key not in PARADIGM_TABLE or tg.pos not in _NOUN_POS:
                    continue
                number = tg.feats.get("Number")
                # agreement check with the nearest modifier on the left (skipping quotes)
                j = i - 1
                while j >= 0 and tagged[j].text in ("«", "»", '"', "“", "”", "„", "(", "—", "-"):
                    j -= 1
                if j >= 0 and (tagged[j].pos in _MODIFIER_POS
                               or (tagged[j].pos == "VERB" and tagged[j].feats.get("VerbForm") == "Part")):
                    mod_num = _modifier_number(tagged[j])
                    if mod_num in ("Sing", "Plur"):
                        number = mod_num
                # A genitive/dative singular needs a governor on its left (a noun, preposition, numeral,
                # adjective...). A word with nothing to govern it ("Города лежали...", "Тогда голоса
                # стихли") is the subject, i.e. nominative; the tagger reads such words as Gen Sing
                # when there is little context. A plural predicate later in the sentence settles it.
                if tg.feats.get("Case") in ("Gen", "Dat", "Loc") and number == "Sing" and _plural_subject(tagged, i):
                    number = "Plur"
                _state["stats"]["seen"] += 1
                if number not in ("Sing", "Plur"):
                    continue
                a = sent.start + tk.start
                b = sent.start + tk.stop
                if b > len(pmap):
                    continue
                ms, me = pmap[a], pmap[b - 1] + 1
                if ms > 0 and line[ms - 1] == "+":
                    ms -= 1                    # the word already carries a leading "+": replace it, do not stack a second one
                new = _match_case(line[ms:me], _decide(key, number))
                if line[ms:me] != new:
                    edits.append((ms, me, new))
        for ms, me, new in sorted(edits, reverse=True):
            line = line[:ms] + new + line[me:]
            _state["stats"]["changed"] += 1
        out_lines.append(line)
    return "\n".join(out_lines)


def stats() -> dict:
    return dict(_state["stats"])


def clear_caches() -> None:
    """Resets the per-run counters (the tagger itself stays loaded). Called by the GUI's
    "Re-run Synthesis" so a new run starts from a clean state."""
    _state["stats"] = {"changed": 0, "seen": 0}
