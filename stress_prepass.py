"""Предпроход «сомнительные места с ранжированием» (без Tk; вызывается из вкладки «Проверка ударений» и тестируется отдельно).

Одна проходка цепочки ударений по книге (`stress_dry_run`, та же, что идёт на озвучку), а затем для каждого слова собирается СУММА слабых признаков,
каждый из которых по отдельности ненадёжен (замеры: уверенность silero-stress ловит лишь ~29% его ошибок, корпус расходится с цепочкой в основном
по делу о омографах и т. п.). Вместе они дают список «слушать в первую очередь»:

  lowconf   пословная модель silero-stress (`accentor._get_model_preds`) не уверена в слове (вероятность лучшего слога < порога); слово из
            списка исключений акцентора считается уверенным
  conflict  независимый корпус знает слово, но ни одно его чтение не совпадает с чтением цепочки
  unstable  в книге у слова разные чтения (≥2, второе встречается ≥2 раза) — омограф или нестабильность модели
  name      слово с заглавной буквы в середине предложения, которого нет в корпусе (имя/название)
  unknown   слова нет ни в корпусе, ни в словаре pymorphy3 (термин/выдумка)
  ctxdiff   чтение цепочки отличается от чтения пословной модели без контекста (решал контекст/шлюз/санитайзер: омограф или правка)

Веса подобраны обратной проверкой на «Пепле победы» (слова, которые пришлось править в словаре, как «положительные»; выборка смещена: их находили
в том числе по расхождению с корпусом, поэтому вес conflict завышен; уверенность модели ниже 0,5 — 13% попаданий, 0,9–0,99 — шум).
Балл = сумма весов признаков × (1 + 0,5·ln(число вхождений)): частые спорные слова всплывают наверх, ведь ошибка повторяется на каждом вхождении.
Это ЛИД для прослушивания, а не вердикт; уверенные ошибки (когда модель уверена и неправа) этим способом не находятся — их ловят слух и словарь.
Возвращает «моменты» в формате `stress_review` (kind="prepass", поля `score`, `signals`), готовые для того же списка и редактора."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import stress_review as srv

VOWELS = "аеёиоуыэюя"
LOWCONF_THRESHOLD = 0.9
WEIGHTS = {"conflict": 3.0, "unstable": 1.0, "name": 1.0, "unknown": 1.5, "ctxdiff": 0.5}
LOWCONF_MAX = 2.0
MIN_SCORE = 2.5
SIGNAL_TITLES = {"lowconf": "модель не уверена", "conflict": "расходится с корпусом", "unstable": "разные чтения в книге",
                 "name": "имя", "unknown": "нет в словарях", "ctxdiff": "решил контекст"}


def stressed_vowel_index(marked: str):
    """0-based index of the stressed vowel among the vowels of a marked word ('з+амок' -> 0); None if no mark; 'ё' counts as stressed."""
    s = marked.lower()
    if "+" in s:
        before = s[:s.index("+")]
        return sum(1 for c in before if c in VOWELS)
    if "ё" in s:
        return sum(1 for c in s[:s.index("ё")] if c in VOWELS)
    return None


def word_confidence(words, accentor=None, batch=256):
    """{plain word: (p_max, argmax_vowel_index)} from the context-free word model of silero-stress. Words from the accentor's exceptions list
    are certain (1.0). Returns {} when the model is unavailable (the signal is then simply absent)."""
    words = sorted(set(words))
    if accentor is None:
        try:
            import fb2_reader as fr
            wrapper = fr._load_silero_stress()
            accentor = wrapper.accentor if wrapper is not None else None
        except Exception:
            accentor = None
    if accentor is None:
        return {}
    out = {}
    try:
        import torch
        exc = getattr(accentor, "exceptions", None)
        todo = []
        for w in words:
            try:
                if exc is not None and w in exc:
                    out[w] = (1.0, None)
                    continue
            except Exception:
                pass
            todo.append(w)
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                probs = accentor._get_model_preds(chunk)[0]
                for w, row in zip(chunk, probs):
                    n_v = sum(1 for c in w if c in VOWELS)
                    r = row[:max(n_v, 1)]
                    p, idx = r.max(0)
                    out[w] = (float(p), int(idx))
    except Exception as e:
        print(f"stress_prepass: уверенность модели недоступна ({type(e).__name__}: {e})")
        return {}
    return out


def lowconf_weight(p) -> float:
    """Weights from the backtest on the 50-chapter book (see module docstring): p >= 0.9 is noise on its own (0.2% hits), 0.5..0.9 a weak hint,
    p < 0.5 a real one (13% of those words had to be fixed in the dictionary)."""
    if p is None or p >= LOWCONF_THRESHOLD or p >= 0.9:
        return 0.0
    return 0.5 if p >= 0.5 else LOWCONF_MAX


def score_word(signals: dict, count: int) -> float:
    """signals: {name: weight or True}. Sum of weights x (1 + 0.5 ln count)."""
    base = 0.0
    for k, v in signals.items():
        base += float(v) if k == "lowconf" else WEIGHTS.get(k, 0.0)
    return base * (1.0 + 0.5 * math.log(max(count, 1)))


def scan_prepass(chapters, only_chapters=None, skip_keys=(), skip_words=(), progress_cb=None, cancel=None, book_overrides_path=None,
                 min_score=MIN_SCORE, max_moments=400, max_occ=40, accentor=None):
    """chapters: [(title, text)]. Returns moments (see stress_review) sorted by score, best first."""
    import fb2_reader as fr
    import stress_questions as sq
    corpus = sq.load_corpus()
    decided = sq.dictionary_words(fr.DEFAULT_FORCED_STRESS_PATH) | {srv.plain(w) for w in skip_words}
    if book_overrides_path:
        try:
            decided |= sq.dictionary_words(book_overrides_path)
        except Exception:
            pass
    skip = set(skip_keys)
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
    except Exception:
        morph = None

    forms = defaultdict(Counter)        # plain -> marked(lower) -> count
    caps = Counter(); mid_caps = Counter(); total = Counter()
    occs = defaultdict(list)            # plain -> occurrences
    idxs = [i for i in range(len(chapters)) if only_chapters is None or i in only_chapters]
    for n, i in enumerate(idxs):
        if cancel is not None and cancel.is_set():
            raise InterruptedError
        if progress_cb:
            progress_cb(f"Предпроход, глава {i}: цепочка ударений ({n + 1}/{len(idxs)})…")
        marked_text, _info = fr.stress_dry_run(chapters[i][1], book_overrides_path=book_overrides_path)
        for pi, para in enumerate(marked_text.split("\n")):
            for m in srv.TOK.finditer(para):
                w = m.group(0)
                pl = srv.plain(w)
                if len(pl) < 3 or sum(1 for c in pl if c in VOWELS) < 2 or pl in decided or "ё" in w.lower():
                    continue
                total[pl] += 1
                forms[pl][w.lower()] += 1
                if w[:1].isupper() or w[:2].replace("+", "")[:1].isupper():
                    caps[pl] += 1
                    before = para[:m.start()].rstrip()
                    if before and before[-1] not in ".!?…\n«\"—-:":
                        mid_caps[pl] += 1
                if len(occs[pl]) < max_occ:
                    a, b = srv.sentence_bounds(para, m.start(), m.end())
                    sent = para[a:b].strip()
                    start = a
                    if len(sent) > 260:
                        start = max(0, m.start() - 110)
                        sent = para[start:m.end() + 110].strip()
                    k = sum(1 for t in srv.TOK.finditer(para[start:m.start()]) if srv.plain(t.group(0)) == pl)
                    occs[pl].append({"chapter": i, "para": pi, "sentence": sent, "idx": k, "word": pl})

    if progress_cb:
        progress_cb("Предпроход: считаю признаки…")
    conf = word_confidence(total.keys(), accentor=accentor)
    per = {}                                        # plain -> (signals, score, top reading)
    for pl, cnt in total.items():
        reading = max(forms[pl].items(), key=lambda kv: kv[1])[0]
        sig = {}
        known = corpus.get(pl)
        if known and srv.reading_conflicts(reading, known):
            sig["conflict"] = True
        f = forms[pl]
        if len(f) > 1 and sorted(f.values())[-2] >= 2:
            sig["unstable"] = True
        if not known and mid_caps[pl] >= 1 and caps[pl] >= 0.7 * cnt:
            sig["name"] = True
        elif not known and "-" not in pl and caps[pl] < 0.7 * cnt and not sq._in_morph_dict(morph, pl):
            sig["unknown"] = True
        c = conf.get(pl)
        if c is not None:
            wlow = lowconf_weight(c[0])
            if wlow > 0:
                sig["lowconf"] = round(wlow, 2)
            if c[1] is not None and stressed_vowel_index(reading) not in (None, c[1]):
                sig["ctxdiff"] = True
        if sig:
            per[pl] = (sig, cnt, reading, c[0] if c else None)

    # one moment per lemma for names / unknown words (case forms share the stressed stem), else per word
    groups = defaultdict(list)
    for pl, (sig, cnt, reading, p) in per.items():
        key = pl
        if ("name" in sig or "unknown" in sig) and morph:
            try:
                key = morph.parse(pl)[0].normal_form.replace("ё", "е")
            except Exception:
                key = pl
        groups[key].append(pl)

    moments = []
    for key, members in groups.items():
        members.sort(key=lambda p: -total[p])
        head = members[0]
        sig = {}
        for p in members:
            for k, v in per[p][0].items():
                if k not in sig or (k == "lowconf" and v > sig[k]):
                    sig[k] = v
        cnt = sum(total[p] for p in members)
        score = score_word(sig, cnt)
        if score < min_score:
            continue
        reading = per[head][2]
        mkey = srv.moment_key(head, reading)
        if mkey in skip:
            continue
        occ = []
        for p in members:
            occ += occs[p]
        occ = occ[:max_occ]
        moments.append({
            "id": 0, "word": head, "raw": head, "reading": reading, "kind": "prepass", "corpus": sorted(corpus.get(head, ())),
            "occ": occ, "count": cnt, "state": "", "chapters": sorted({o["chapter"] for o in occ}), "group": members,
            "candidates": srv.candidates_for(head, reading), "score": round(score, 2), "signals": sig,
            "confidence": per[head][3]})
    moments.sort(key=lambda m: (-m["score"], -m["count"], m["word"]))
    moments = moments[:max_moments]
    for n, m in enumerate(moments):
        m["id"] = n
    return moments


def describe_signals(sig: dict) -> str:
    return ", ".join(SIGNAL_TITLES.get(k, k) + (f" ({v})" if k == "lowconf" else "") for k, v in sig.items())
