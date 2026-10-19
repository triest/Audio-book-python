"""Логика вкладки «Сомнительные места» (без Tk, тестируется отдельно): найти места, где ударение цепочки расходится с независимым
корпусом (или слова из списка пользователя), показать каждое с контекстом, помочь послушать и либо подтвердить, либо исправить.

Единица работы — «момент»: слово (plain) в конкретном чтении цепочки («земл+и») + все его вхождения в книге.
  scan_moments(chapters, ...)      -> список моментов (рабочий поток; прогресс и отмена как в stress_questions.scan_chapters)
  locate_in_asr(asr_words, ...)    -> (начало, конец) слова в готовой главе по кэшу ASR (tools/asr_cache_cuda.py)
  render_occurrence(...)           -> предложение с заданным чтением ТОЛЬКО на одном вхождении
  phrase_for_occurrence(...)       -> (ключ, значение) фразовой записи «только это место» для файла правок книги
  load_state / save_state          -> «подтверждено»/«исправлено» (stress_review_state.json рядом со словарями)
Ударение в звуке по-прежнему подтверждает только слух: инструмент показывает место, даёт послушать и записывает решение."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

TOK = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")
VOW = "аеёиоуыэюя"


def plain(w: str) -> str:
    return w.replace("+", "").lower().replace("ё", "е")


def norm_reading(w: str) -> str:
    return w.lower().replace("ё", "е")


def _state_path() -> Path:
    import fb2_reader as fr
    return Path(fr._app_paths.user_file("stress_review_state.json"))


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    p = _state_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def moment_key(word: str, reading: str) -> str:
    return f"{plain(word)}|{norm_reading(reading)}"


def reading_conflicts(reading: str, corpus_readings) -> bool:
    """True when the corpus knows the word but none of its readings equals the chain's one (modulo е/ё). The corpus keeps ONE reading of
    a homograph and has no «ё», so a conflict is a LEAD to listen, not an error."""
    if not corpus_readings:
        return False
    r = norm_reading(reading)
    if any(norm_reading(c) == r for c in corpus_readings):
        return False
    if "ё" in reading and not any("+" in c for c in corpus_readings):
        return False
    return True


def sentence_bounds(marked: str, start: int, end: int):
    a = max(marked.rfind(c, 0, start) for c in ".!?…\n")
    a = 0 if a < 0 else a + 1
    bs = [i for i in (marked.find(c, end) for c in ".!?…\n") if i >= 0]
    b = min(bs) + 1 if bs else len(marked)
    return a, b


def scan_moments(chapters, only_chapters=None, manual_words=(), use_corpus=True, skip_keys=(), progress_cb=None, cancel=None,
                 book_overrides_path=None, max_occ=40):
    """chapters: [(title, text)]; only_chapters: set of chapter indexes (None = all).
    A moment is made for every marked word that disagrees with the corpus (use_corpus) and for every word in manual_words.
    Words already decided in the user dictionaries/book phrases and keys in skip_keys (state: ok/fixed) are left out."""
    import fb2_reader as fr
    import stress_questions as sq
    corpus = sq.load_corpus() if use_corpus else {}
    decided = sq.dictionary_words(fr.DEFAULT_FORCED_STRESS_PATH)
    if book_overrides_path and Path(book_overrides_path).exists():
        decided |= sq.dictionary_words(book_overrides_path)
    manual = {plain(w) for w in manual_words if w.strip()}
    skip = set(skip_keys)
    moments: dict = {}
    idxs = [i for i in range(len(chapters)) if only_chapters is None or i in only_chapters]
    for n, i in enumerate(idxs):
        if cancel is not None and cancel.is_set():
            raise InterruptedError
        title, text = chapters[i]
        if progress_cb:
            progress_cb(f"Глава {i}: цепочка ударений ({n + 1}/{len(idxs)})…")
        marked_text, _info = fr.stress_dry_run(text, book_overrides_path=book_overrides_path)
        for pi, para in enumerate(marked_text.split("\n")):
            for m in TOK.finditer(para):
                w = m.group(0)
                pl = plain(w)
                if len(pl) < 3:
                    continue
                is_manual = pl in manual
                flagged = False
                if use_corpus and "+" in w and pl not in decided:
                    flagged = reading_conflicts(w, corpus.get(pl))
                if not (is_manual or flagged):
                    continue
                key = moment_key(w, w)
                if key in skip and not is_manual:
                    continue
                a, b = sentence_bounds(para, m.start(), m.end())
                sent = para[a:b].strip()
                if len(sent) > 260:
                    lo = max(0, m.start() - 110)
                    sent = para[lo:m.end() + 110].strip()
                before = para[a:m.start()] if sent == para[a:b].strip() else para[max(0, m.start() - 110):m.start()]
                k = sum(1 for t in TOK.finditer(before) if plain(t.group(0)) == pl)
                mom = moments.setdefault(key, {
                    "id": 0, "word": pl, "raw": w.lower().replace("+", ""), "reading": w.lower(), "kind": "manual" if is_manual else "diff",
                    "corpus": sorted(corpus.get(pl, ())) if corpus else [], "occ": [], "count": 0, "state": "",
                    "chapters": set()})
                mom["count"] += 1
                mom["chapters"].add(i)
                if len(mom["occ"]) < max_occ:
                    mom["occ"].append({"chapter": i, "para": pi, "sentence": sent, "idx": k})
    out = sorted(moments.values(), key=lambda q: (-q["count"], q["word"]))
    for n, q in enumerate(out):
        q["id"] = n
        q["chapters"] = sorted(q["chapters"])
        q["candidates"] = candidates_for(q["raw"], q["reading"])
    return out


def candidates_for(pl: str, first: "str | None" = None):
    """one reading per vowel (stress before the vowel), the chain's reading first; 'ё' is always stressed."""
    if "ё" in pl:
        return [pl.replace("ё", "+ё")]
    res = [pl[:i] + "+" + pl[i:] for i, ch in enumerate(pl) if ch in VOW]
    if first and first in res:
        res.remove(first)
        res.insert(0, first)
    return res


def render_occurrence(sentence: str, word: str, form: str, idx: int) -> str:
    """sentence with `form` set on the idx-th occurrence of `word` only (other words keep the chain's marks)."""
    pl = plain(word)
    seen = [-1]

    def sub(m):
        if plain(m.group(0)) != pl:
            return m.group(0)
        seen[0] += 1
        if seen[0] != idx:
            return m.group(0)
        f = form
        return f[:1].upper() + f[1:] if m.group(0).lstrip("+")[:1].isupper() else f
    return TOK.sub(sub, sentence)


def plain_words(sentence: str):
    return [plain(t.group(0)) for t in TOK.finditer(sentence)]


def phrase_for_occurrence(sentence: str, word: str, form: str, idx: int, left: int = 1, right: int = 1):
    """(key, value) for the book phrase file: the target word with `left`/`right` neighbours. The value is the chain's marked words with the
    chosen reading on the target. Neighbours make the entry apply to this context only, not to every occurrence of the word."""
    pl = plain(word)
    toks = list(TOK.finditer(sentence))
    hits = [n for n, t in enumerate(toks) if plain(t.group(0)) == pl]
    if idx >= len(hits):
        raise ValueError("вхождение не найдено в предложении")
    n = hits[idx]
    lo, hi = max(0, n - left), min(len(toks), n + right + 1)
    vals = []
    for j in range(lo, hi):
        t = toks[j].group(0)
        vals.append(form.lower() if j == n else t.lower())
    key = " ".join(plain(t.group(0)) for t in toks[lo:hi])
    return key, " ".join(vals)


def affected_chapters(moment) -> list:
    return list(moment.get("chapters", []))


def locate_in_asr(asr_words, sentence: str, word: str, idx: int = 0, win: int = 3):
    """(start, end) seconds of the idx-th occurrence of `word` inside `sentence` in a finished chapter, from the ASR cache
    ([{"w","s","e"}]). The sentence is aligned by a sliding window of its words; returns None if nothing matches well."""
    toks = plain_words(sentence)
    pl = plain(word)
    hits = [n for n, t in enumerate(toks) if t == pl]
    if idx >= len(hits) or not asr_words:
        return None
    n = hits[idx]
    lo, hi = max(0, n - win), min(len(toks), n + win + 1)
    ctx = toks[lo:hi]
    asr = [re.sub(r"[^а-я]", "", x["w"].lower().replace("ё", "е")) for x in asr_words]
    best, best_i = 0.0, None
    L = len(ctx)
    for i in range(0, max(1, len(asr) - L + 1)):
        seg = asr[i:i + L]
        sc = sum(1 for a, b in zip(ctx, seg) if a == b)
        if sc <= best:
            continue
        best, best_i = sc, i
    if best_i is None or best < max(2, L // 2):
        return None
    j = best_i + (n - lo)
    for off in (0, -1, 1, -2, 2):
        if 0 <= j + off < len(asr) and asr[j + off] == pl:
            j += off
            break
    j = min(max(j, 0), len(asr) - 1)
    return float(asr_words[j]["s"]), float(asr_words[j]["e"])


def find_chapter_wav(outdir, chapter: int):
    p = Path(outdir)
    wavs = sorted(p.glob(f"{chapter + 1:03d}_*.wav"))
    return wavs[0] if wavs else None


def load_asr_cache(outdir, chapter: int):
    c = Path(outdir) / "_asr" / f"ch{chapter}.json"
    try:
        return json.loads(c.read_text(encoding="utf-8"))["words"]
    except Exception:
        return None


def cut_clip_bytes(wav_path, start: float, end: float, pad_before=2.0, pad_after=2.0) -> bytes:
    """a playable WAV (bytes) of [start-pad_before, end+pad_after] cut from a finished chapter."""
    import io
    import wave
    with wave.open(str(wav_path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(max(int((start - pad_before) * sr), 0))
        frames = w.readframes(int((end - start + pad_before + pad_after) * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(sr)
        o.writeframes(frames)
    return buf.getvalue()
