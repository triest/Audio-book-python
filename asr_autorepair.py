"""Auto-repair of Silero sound artifacts by ASR (hum "Ммм" in place of a word, a stretched word, a swallowed piece).

After a chunk is synthesised, a long-lived whisper worker (tools/asr_worker.py in .venv_cosyvoice3) transcribes it; the transcript is aligned with
the source words (difflib). If it has defects, the chunk is re-synthesised with a tweaked SSML:
  1. the comma pause right before the damaged word is dropped (the case that fixed "курортом, но там" -> "Ммм, на том");
  2. all comma pauses of the chunk are dropped;
and the attempt with the fewest defects wins (the original stays if nothing is better). Any ASR failure leaves the chunk as it was.
ASR does not judge stress; this only finds and retries places where the text was not spoken as written.
Enabled by run_silero_rest(asr_repair=True) / --asr-repair. Env: AUDIOBOOK_ASR_PYTHON (python of the ASR venv), AUDIOBOOK_ASR_MODEL."""
import difflib
import io
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRETCH_S = 2.5          # one word longer than this (Whisper may also stretch a word over a pause: only a trigger for a retry)
MISSING_WORDS = 8        # swallowed run of source words
_HUM_RE = re.compile(r"^(м+|э+м*|хм+|ам+|ы+|а{4,}|у{4,}|о{4,})$")
_COMMA_BREAK_RE = re.compile(r',<break time="\d+ms"/>')
_PROSODY_RE = re.compile(r"</?prosody[^>]*>")
_BREAK_BEFORE_WORD_RE = re.compile(r',<break time="\d+ms"/>(\s*)([^\s<,.!?;:—]+)')


def norm(w: str) -> str:
    return re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))


def source_words(text: str) -> list:
    import fb2_reader as f
    body = f.numbers_to_words_ru(f.strip_footnote_markers(text)) if hasattr(f, "strip_footnote_markers") else f.numbers_to_words_ru(text)
    return [w for w in (norm(x) for x in body.split()) if w]


# Longest normal pause between two neighbouring words inside a chunk, by what the source has after the first word
# (99.9th percentile over the 50 chapters of the sample book: '' 0.16, ',' 0.64, '.' 0.98, paragraph 1.6 s).
# A longer silence-or-sound that Whisper did not write down is a hum / stretched onset ("Ммм" before "на").
GAP_LIMIT_S = {"": 0.35, "c": 0.75, "s": 0.95, "p": 1.6}
GAP_TYPICAL_S = {"": 0.0, "c": 0.26, "s": 0.42, "p": 0.82}   # median pause by boundary type: what a cut leaves behind


def source_boundaries(text: str):
    """-> (normalised words, kinds): kind[i] is what follows word i in the source: 'p' paragraph end, 's' sentence end,
    'c' comma / colon / dash, '' nothing. Aligned 1:1 with source_words(text)."""
    import fb2_reader as f
    body = f.numbers_to_words_ru(f.strip_footnote_markers(text)) if hasattr(f, "strip_footnote_markers") else f.numbers_to_words_ru(text)
    words, kinds = [], []
    paras = [p for p in body.split("\n") if p.strip()]
    for pi, para in enumerate(paras):
        toks = para.split()
        for ti, t in enumerate(toks):
            n = norm(t)
            if not n:
                if kinds and kinds[-1] == "" and re.search(r"[—–-]", t):
                    kinds[-1] = "c"
                continue
            last = ti == len(toks) - 1
            if last and pi < len(paras) - 1:
                k = "p"
            elif re.search(r"[.!?…][\"'»)]*$", t):
                k = "s"
            elif re.search(r"[,;:][\"'»)]*$", t):
                k = "c"
            else:
                k = ""
            words.append(n)
            kinds.append(k)
    return words, kinds


def chunk_defects(src: list, words: list, lead_ok: bool = False, kinds: "list | None" = None) -> list:
    """src — normalised source words, words — ASR items {"w","s","e"}. -> [{"kind","src_index","text"}]
    lead_ok — first chunk of a chapter: its first words (the title) are followed by a long pause that Whisper
    glues to the word ("ладно (3.3s)"), so a stretch there is not a defect.
    kinds — source_boundaries() kinds: enables the "gap" check (an unexplained long pause between two correctly
    recognised neighbouring words: the hum that Whisper did not transcribe)."""
    kept = [(norm(x["w"]), x) for x in words if norm(x["w"])]
    asr = [k for k, _ in kept]
    out = []
    sm = difflib.SequenceMatcher(None, src, asr, autojunk=False)
    to_src = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                to_src[j1 + d] = i1 + d
            continue
        src_hum = any(_HUM_RE.match(s) for s in src[i1:i2])
        for j in range(j1, j2):
            if _HUM_RE.match(asr[j]) and not src_hum:
                out.append({"kind": "hum", "src_index": min(i1, len(src) - 1), "text": " ".join(asr[max(j1 - 1, 0):j2 + 1]),
                            "t0": kept[j][1]["s"], "t1": kept[j][1]["e"]})
                break
        if tag in ("delete", "replace") and i2 - i1 >= MISSING_WORDS and j2 - j1 < (i2 - i1) / 2:
            out.append({"kind": "missing", "src_index": i1, "text": " ".join(src[i1:i2])[:100]})
    for j, (w, x) in enumerate(kept):
        if x["e"] - x["s"] > STRETCH_S and len(w) <= 14 and j in to_src and not (lead_ok and j < 4):
            out.append({"kind": "stretch", "src_index": to_src[j], "text": f"{w} ({x['e'] - x['s']:.1f}s)"})
    if kinds is not None and len(kinds) == len(src):
        for j in range(len(kept) - 1):
            if j in to_src and j + 1 in to_src and to_src[j + 1] == to_src[j] + 1:
                a, b = kept[j][1], kept[j + 1][1]
                gap = b["s"] - a["e"]
                kind = kinds[to_src[j]]
                if gap > GAP_LIMIT_S[kind]:
                    out.append({"kind": "gap", "src_index": to_src[j + 1], "text": f"{kept[j][0]} ({gap:.1f}s) {kept[j + 1][0]}",
                                "t0": a["e"], "t1": b["s"], "typical": GAP_TYPICAL_S[kind]})
    return out


def loud_frames(wav_bytes: bytes):
    """Boolean array, one value per 10 ms frame: is the frame loud (a voiced sound, not the silence of a planned pause)."""
    import numpy as np
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).reshape(-1, ch).astype(np.float64).mean(axis=1)
    a = a - a.mean()
    hop = max(int(sr * 0.01), 1)
    nfr = len(a) // hop
    if nfr < 4:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((a[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1))
    return rms > max(float(np.percentile(rms, 95)) * 0.08, 1e-9)


def voiced_spans(wav_bytes, t0: float, t1: float, margin_s: float = 0.03, min_s: float = 0.08) -> list:
    """Sound inside the window [t0, t1] of a wav: runs of loud 10 ms frames (a hum), with margin_s kept away from the window edges
    so the neighbouring words are not clipped. Silence (the planned pause) is left alone."""
    loud = loud_frames(wav_bytes) if isinstance(wav_bytes, (bytes, bytearray)) else wav_bytes   # or a precomputed loud_frames()
    nfr = len(loud)
    if nfr < 4:
        return []
    f0, f1 = int((t0 + margin_s) / 0.01), int((t1 - margin_s) / 0.01)
    spans, start = [], None
    for i in range(max(f0, 0), min(f1, nfr)):
        if loud[i] and start is None:
            start = i
        if (not loud[i] or i == min(f1, nfr) - 1) and start is not None:
            end = i + 1 if loud[i] else i
            if (end - start) * 0.01 >= min_s:
                spans.append((start * 0.01, end * 0.01))
            start = None
    return spans


def cut_spans(wav_bytes: bytes, spans: list, fade_s: float = 0.012) -> bytes:
    """Removes the time spans [(t0, t1), ...] from a PCM wav and joins the rest with a short crossfade (no clicks)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        params = w.getparams()
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    import numpy as np
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).reshape(-1, ch).astype(np.float64)
    n = a.shape[0]
    cuts = sorted((max(int(t0 * sr), 0), min(int(t1 * sr), n)) for t0, t1 in spans if t1 > t0)
    fade = max(int(fade_s * sr), 1)
    parts, pos = [], 0
    for c0, c1 in cuts:
        if c0 > pos:
            parts.append(a[pos:c0])
        pos = max(pos, c1)
    parts.append(a[pos:])
    out = parts[0]
    for nxt in parts[1:]:
        k = min(fade, len(out), len(nxt))
        if k > 0:
            ramp = np.linspace(0.0, 1.0, k)[:, None]
            blend = out[-k:] * (1.0 - ramp) + nxt[:k] * ramp
            out = np.concatenate([out[:-k], blend, nxt[k:]])
        else:
            out = np.concatenate([out, nxt])
    info = np.iinfo(dt)
    pcm = np.clip(np.rint(out), info.min, info.max).astype(dt).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(pcm)
    return buf.getvalue()


def join_wavs(parts: list, gap_ms: int = 270) -> bytes:
    """Joins PCM wavs of the same format with gap_ms of silence between them."""
    frames, params = [], None
    for b in parts:
        with wave.open(io.BytesIO(b), "rb") as w:
            params = params or w.getparams()
            frames.append(w.readframes(w.getnframes()))
    silence = bytes(int(params.framerate * gap_ms / 1000) * params.sampwidth * params.nchannels)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(silence.join(frames))
    return buf.getvalue()


class SplitSsml(list):
    """SSML documents to be synthesised as separate requests and joined (gap_ms of silence between them)."""
    gap_ms = 270


_TAG_RE = re.compile(r"<(/?)([a-zA-Z]+)([^>]*?)(/?)>")
_TOKEN_RE = re.compile(r"<[^>]+>|[^<\s]+")


def split_before_word(ssml: str, src: list, src_index: int):
    """Splits the SSML at the start of the sentence that holds the src_index-th source word, re-opening the same enclosing tags
    (speak/p/prosody) in the second part. Returns a SplitSsml, or the unchanged ssml when the word is in the first sentence
    or cannot be located (then the attempt is simply a repeat)."""
    if not (0 <= src_index < len(src)):
        return ssml
    target = src[src_index]
    k = src[:src_index].count(target)
    pos, seen = None, 0
    for m in _TOKEN_RE.finditer(ssml):
        tok = m.group(0)
        if tok.startswith("<") or norm(tok) != target:
            continue
        if seen == k:
            pos = m.start()
            break
        seen += 1
    if pos is None:
        return ssml
    s_open = ssml.rfind("<s>", 0, pos)
    if s_open <= 0:
        return ssml
    head_raw, tail_raw = ssml[:s_open], ssml[s_open:]
    stack = []
    for m in _TAG_RE.finditer(head_raw):
        closing, name, _attrs, selfclosing = m.groups()
        if selfclosing or name == "break":
            continue
        if closing:
            if stack and stack[-1][0] == name:
                stack.pop()
        else:
            stack.append((name, m.group(0)))
    gap = 270
    tm = re.search(r'<break time="(\d+)ms"/>\s*$', head_raw)
    if tm:
        gap = int(tm.group(1))
        head_raw = head_raw[:tm.start()]
    head = head_raw + "".join("</%s>" % n for n, _ in reversed(stack))
    tail = "".join(t for _, t in stack) + tail_raw
    if "<s>" not in head or "<s>" not in tail:
        return ssml
    out = SplitSsml([head, tail])
    out.gap_ms = gap
    return out


_RATE_RE = re.compile(r'rate="(\d+)%"')


def nudge_rate(ssml: str, delta: int = 5) -> str:
    """A small change of the speaking rate: a different input for the model, the same text."""
    if "<prosody" in ssml:
        return _RATE_RE.sub(lambda m: 'rate="%d%%"' % (int(m.group(1)) + delta), ssml)
    return ssml.replace("<p>", '<p><prosody rate="%d%%">' % (100 + delta), 1).replace("</p>", "</prosody></p>", 1) if "<p>" in ssml else ssml


def tweaks_for(defects: list, src: list):
    """Ordered list of (name, function ssml -> ssml)."""
    targets = {src[d["src_index"]] for d in defects if 0 <= d["src_index"] < len(src)}

    def drop_before_targets(ssml: str) -> str:
        def sub(m):
            return "," + m.group(1) + m.group(2) if norm(m.group(2)) in targets else m.group(0)
        return _BREAK_BEFORE_WORD_RE.sub(sub, ssml)

    def drop_all_commas(ssml: str) -> str:
        return _COMMA_BREAK_RE.sub(",", ssml)

    def drop_prosody(ssml: str) -> str:
        # smart-emotion <prosody rate pitch> shifts can make Silero hum at the start of a word
        return _PROSODY_RE.sub("", ssml)

    first = next((d["src_index"] for d in defects if 0 <= d["src_index"] < len(src)), -1)

    def split_sentence(ssml: str):
        return split_before_word(ssml, src, first)

    return [("drop pause before the damaged word", drop_before_targets),
            ("synthesise the damaged sentence separately", split_sentence),
            ("speaking rate -5", lambda t: nudge_rate(t, -5)),
            ("drop all comma pauses", drop_all_commas),
            ("drop emotion prosody", lambda t: drop_prosody(t)),
            ("drop emotion prosody and all comma pauses", lambda t: drop_all_commas(drop_prosody(t))),
            ("speaking rate +5", lambda t: nudge_rate(t, 5))]


class AsrWorker:
    def __init__(self, log=print, python: "str | None" = None):
        self.log = log
        self.proc = None
        self._q: "queue.Queue[str]" = queue.Queue()
        self._python = python or os.environ.get("AUDIOBOOK_ASR_PYTHON") or str(ROOT / ".venv_cosyvoice3" / "Scripts" / "python.exe")
        self.ok = False

    def start(self, timeout: float = 300.0) -> bool:
        worker = ROOT / "tools" / "asr_worker.py"
        if not Path(self._python).exists() or not worker.exists():
            self.log(f"asr_repair: ASR-окружение не найдено ({self._python}) — автопочинка отключена.")
            return False
        try:
            self.proc = subprocess.Popen([self._python, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.log(f"asr_repair: не удалось запустить ASR ({e}) — автопочинка отключена.")
            return False
        threading.Thread(target=self._read, daemon=True).start()
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            line = ""
        self.ok = line == "@@ready"
        if not self.ok:
            self.log("asr_repair: ASR-процесс не запустился — автопочинка отключена.")
            self.close()
        return self.ok

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("@@"):
                self._q.put(line.rstrip("\n"))
        self._q.put("@@eof")

    def transcribe(self, wav_path: str, timeout: float = 180.0) -> list:
        self.proc.stdin.write(json.dumps({"wav": wav_path}) + "\n")
        self.proc.stdin.flush()
        line = self._q.get(timeout=timeout)
        if line == "@@eof":
            self.ok = False
            raise RuntimeError("ASR-процесс завершился")
        data = json.loads(line[2:])
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["words"]

    def close(self):
        self.ok = False
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.proc.kill()


class AsrRepair:
    def __init__(self, outdir: Path, log=print, max_attempts: int = 7):
        self.log = log
        self.max_attempts = max_attempts
        self.report = Path(outdir) / "asr_repairs.jsonl"
        self.worker = AsrWorker(log=log)
        self.stats = {"checked": 0, "defective": 0, "fixed": 0, "kept_defect": 0}
        self.ok = self.worker.start()

    def _defects(self, src, kinds, wav_bytes, lead_ok: bool = False) -> list:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(wav_bytes)
            return chunk_defects(src, self.worker.transcribe(tmp), lead_ok=lead_ok, kinds=kinds)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def check_and_fix(self, plain_chunk: str, wav_bytes: bytes, resynth, tag: str = "", first_chunk: bool = False) -> bytes:
        """resynth(tweak) -> new wav bytes, tweak: function ssml -> ssml. Returns the best wav (the original if nothing is better or ASR fails)."""
        if not self.ok:
            return wav_bytes
        try:
            src, kinds = source_boundaries(plain_chunk)
            if len(src) < 3:
                return wav_bytes
            self.stats["checked"] += 1
            base = self._defects(src, kinds, wav_bytes, first_chunk)
            if not base:
                return wav_bytes
            self.stats["defective"] += 1
            best, best_defects, tried, seen = wav_bytes, base, [], {}
            for name, fn in tweaks_for(base, src)[: self.max_attempts]:
                def capture(ssml, fn=fn):
                    seen.setdefault("ssml", ssml)
                    return fn(ssml)
                try:
                    cand = resynth(capture)
                    d = self._defects(src, kinds, cand, first_chunk)
                except Exception as e:  # noqa: BLE001
                    self.log(f"asr_repair {tag}: попытка «{name}» не удалась ({e})")
                    continue
                tried.append({"tweak": name, "defects": len(d)})
                if len(d) < len(best_defects):
                    best, best_defects = cand, d
                if not best_defects:
                    break
            cut_from = [d for d in best_defects if d["kind"] in ("hum", "gap") and "t0" in d]
            if cut_from:
                # last resort: cut the hum out of the best candidate (a transcribed hum: its own words; an untranscribed one:
                # the loud runs inside the unexplained pause), kept only if ASR confirms it is better
                try:
                    spans = []
                    for d in cut_from:
                        spans += [(d["t0"], d["t1"])] if d["kind"] == "hum" else voiced_spans(best, d["t0"], d["t1"])
                    if not spans:
                        raise ValueError("нет звонких участков для вырезания")
                    cut = cut_spans(best, spans)
                    dc = self._defects(src, kinds, cut, first_chunk)
                    tried.append({"tweak": "cut the hum out of the audio", "defects": len(dc)})
                    if len(dc) < len(best_defects):
                        best, best_defects = cut, dc
                except Exception as e:  # noqa: BLE001
                    self.log(f"asr_repair {tag}: вырезание гула не удалось ({e})")
            self.stats["fixed" if len(best_defects) < len(base) else "kept_defect"] += 1
            what = "; ".join(d["kind"] + ": " + d["text"] for d in base[:3])
            self.log(f"asr_repair {tag}: дефектов {len(base)} -> {len(best_defects)} ({what})")
            with open(self.report, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"tag": tag, "before": base, "after": best_defects, "tried": tried,
                                     "text": plain_chunk[:120], "ssml": seen.get("ssml", "")}, ensure_ascii=False) + "\n")
            return best
        except Exception as e:  # noqa: BLE001
            self.log(f"asr_repair {tag}: проверка пропущена ({e})")
            return wav_bytes

    def close(self):
        self.worker.close()
        if self.ok:
            self.log(f"asr_repair: проверено {self.stats['checked']}, с дефектом {self.stats['defective']}, "
                     f"исправлено {self.stats['fixed']}, осталось с дефектом {self.stats['kept_defect']}.")
