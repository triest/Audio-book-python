r"""Downloads an openai-whisper model into ~\.cache\whisper with resume, a SHA-256 check and timeouts (used by install.bat).

  .venv_cosyvoice3\Scripts\python.exe tools\fetch_whisper_model.py medium

The URL (and the expected SHA-256, which is a path segment of it) is taken from the installed whisper package, so it is always the
one that package expects. Exit code 0: the model is in place and verified; 1: it could not be fetched (the program then falls back to
the small model, see tools/asr_worker.py). Never hangs silently: socket timeout 30 s, a progress line every ~5 %, 5 attempts."""
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path


def model_url(name: str) -> str:
    import whisper
    return whisper._MODELS[name]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, attempts: int = 5) -> bool:
    url = model_url(name)
    expected = url.split("/")[-2]
    dst = Path.home() / ".cache" / "whisper" / f"{name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256_of(dst) == expected:
        print(f"whisper {name}: already in place and verified ({dst})")
        return True
    part = dst.with_suffix(".pt.part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = getattr(resp, "status", 200)
                if have and status != 206:       # the server ignored Range: start over
                    have = 0
                total = have + int(resp.headers.get("Content-Length") or 0)
                mode = "ab" if have else "wb"
                done, next_mark, t0 = have, 5, time.time()
                with open(part, mode) as out:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if total and done * 100 // total >= next_mark:
                            speed = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"whisper {name}: {done * 100 // total}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB, {speed:.1f} MB/s)", flush=True)
                            next_mark = done * 100 // total + 5
        except Exception as e:  # noqa: BLE001
            print(f"whisper {name}: attempt {attempt}/{attempts} failed: {e}", flush=True)
            time.sleep(min(5 * attempt, 20))
            continue
        if sha256_of(part) == expected:
            os.replace(part, dst)
            print(f"whisper {name}: downloaded and verified ({dst})")
            return True
        print(f"whisper {name}: checksum mismatch, starting over", flush=True)
        part.unlink(missing_ok=True)
    return False


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "medium"
    ok = fetch(model)
    if not ok:
        print(f"whisper {model}: NOT downloaded - the ASR check will use the small model (still works, weaker).")
    sys.exit(0 if ok else 1)
