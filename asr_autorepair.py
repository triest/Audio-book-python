"""Auto-repair of Silero sound artifacts by ASR (hum "Ммм" in place of a word, a stretched word, a swallowed piece).

After a chunk is synthesised, a long-lived whisper worker (tools/asr_worker.py in .venv_cosyvoice3) transcribes it; the transcript is aligned with
the source words (difflib). If it has defects, the chunk is re-synthesised with a tweaked SSML:
  1. the comma pause right before the damaged word is dropped (the case that fixed "курортом, но там" -> "Ммм, на том");
  2. all comma pauses of the chunk are dropped;
and the attempt with the fewest defects wins (the original stays if nothing is better). Any ASR failure leaves the chunk as it was.
ASR does not judge stress; this only finds and retries places where the text was not spoken as written.
Enabled by run_silero_rest(asr_repair=True) / --asr-repair. Env: AUDIOBOOK_ASR_PYTHON (python of the ASR venv), AUDIOBOOK_ASR_MODEL."""
import difflib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRETCH_S = 2.5          # one word longer than this (Whisper may also stretch a word over a pause: only a trigger for a retry)
MISSING_WORDS = 8        # swallowed run of source words
_HUM_RE = re.compile(r"^(м+|э+м*|хм+|ам+|ы+|а{4,}|у{4,}|о{4,})$")
_COMMA_BREAK_RE = re.compile(r',<break time="\d+ms"/>')
_BREAK_BEFORE_WORD_RE = re.compile(r',<break time="\d+ms"/>(\s*)([^\s<,.!?;:—]+)')


def norm(w: str) -> str:
    return re.sub(r"[^а-я]", "", w.lower().replace("ё", "е"))


def source_words(text: str) -> list:
    import fb2_reader as f
    body = f.numbers_to_words_ru(f.strip_footnote_markers(text)) if hasattr(f, "strip_footnote_markers") else f.numbers_to_words_ru(text)
    return [w for w in (norm(x) for x in body.split()) if w]


def chunk_defects(src: list, words: list) -> list:
    """src — normalised source words, words — ASR items {"w","s","e"}. -> [{"kind","src_index","text"}]"""
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
                out.append({"kind": "hum", "src_index": min(i1, len(src) - 1), "text": " ".join(asr[max(j1 - 1, 0):j2 + 1])})
                break
        if tag in ("delete", "replace") and i2 - i1 >= MISSING_WORDS and j2 - j1 < (i2 - i1) / 2:
            out.append({"kind": "missing", "src_index": i1, "text": " ".join(src[i1:i2])[:100]})
    for j, (w, x) in enumerate(kept):
        if x["e"] - x["s"] > STRETCH_S and len(w) <= 14 and j in to_src:
            out.append({"kind": "stretch", "src_index": to_src[j], "text": f"{w} ({x['e'] - x['s']:.1f}s)"})
    return out


def tweaks_for(defects: list, src: list):
    """Ordered list of (name, function ssml -> ssml)."""
    targets = {src[d["src_index"]] for d in defects if 0 <= d["src_index"] < len(src)}

    def drop_before_targets(ssml: str) -> str:
        def sub(m):
            return "," + m.group(1) + m.group(2) if norm(m.group(2)) in targets else m.group(0)
        return _BREAK_BEFORE_WORD_RE.sub(sub, ssml)

    def drop_all_commas(ssml: str) -> str:
        return _COMMA_BREAK_RE.sub(",", ssml)

    return [("drop pause before the damaged word", drop_before_targets), ("drop all comma pauses", drop_all_commas)]


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
    def __init__(self, outdir: Path, log=print, max_attempts: int = 2):
        self.log = log
        self.max_attempts = max_attempts
        self.report = Path(outdir) / "asr_repairs.jsonl"
        self.worker = AsrWorker(log=log)
        self.stats = {"checked": 0, "defective": 0, "fixed": 0, "kept_defect": 0}
        self.ok = self.worker.start()

    def _defects(self, src, wav_bytes) -> list:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(wav_bytes)
            return chunk_defects(src, self.worker.transcribe(tmp))
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def check_and_fix(self, plain_chunk: str, wav_bytes: bytes, resynth, tag: str = "") -> bytes:
        """resynth(tweak) -> new wav bytes, tweak: function ssml -> ssml. Returns the best wav (the original if nothing is better or ASR fails)."""
        if not self.ok:
            return wav_bytes
        try:
            src = source_words(plain_chunk)
            if len(src) < 3:
                return wav_bytes
            self.stats["checked"] += 1
            base = self._defects(src, wav_bytes)
            if not base:
                return wav_bytes
            self.stats["defective"] += 1
            best, best_defects, tried = wav_bytes, base, []
            for name, fn in tweaks_for(base, src)[: self.max_attempts]:
                try:
                    cand = resynth(fn)
                    d = self._defects(src, cand)
                except Exception as e:  # noqa: BLE001
                    self.log(f"asr_repair {tag}: попытка «{name}» не удалась ({e})")
                    continue
                tried.append({"tweak": name, "defects": len(d)})
                if len(d) < len(best_defects):
                    best, best_defects = cand, d
                if not best_defects:
                    break
            self.stats["fixed" if len(best_defects) < len(base) else "kept_defect"] += 1
            what = "; ".join(d["kind"] + ": " + d["text"] for d in base[:3])
            self.log(f"asr_repair {tag}: дефектов {len(base)} -> {len(best_defects)} ({what})")
            with open(self.report, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"tag": tag, "before": base, "after": best_defects, "tried": tried,
                                     "text": plain_chunk[:120]}, ensure_ascii=False) + "\n")
            return best
        except Exception as e:  # noqa: BLE001
            self.log(f"asr_repair {tag}: проверка пропущена ({e})")
            return wav_bytes

    def close(self):
        self.worker.close()
        if self.ok:
            self.log(f"asr_repair: проверено {self.stats['checked']}, с дефектом {self.stats['defective']}, "
                     f"исправлено {self.stats['fixed']}, осталось с дефектом {self.stats['kept_defect']}.")
