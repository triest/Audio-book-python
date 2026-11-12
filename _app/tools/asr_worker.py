r"""Long-lived ASR worker for asr_autorepair.py (runs in .venv_cosyvoice3: openai-whisper on CUDA, see asr_cache_cuda.py).

Protocol: one JSON per stdin line {"wav": "<path>"}; answer on stdout as a line "@@" + JSON {"words": [{"w","s","e"}]} or {"error": ".."}.
First line "@@ready" once the model is loaded. Everything else whisper prints goes to stderr."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_cache_cuda import load_16k  # noqa: E402


def _exit_with_parent():
    """The GUI closed (cross, crash, kill) while a long chapter is being decoded: stdin EOF would only be noticed after the job,
    so a watcher thread ends the process as soon as the parent (AUDIOBOOK_PARENT_PID) is gone."""
    pid = int(os.environ.get("AUDIOBOOK_PARENT_PID") or 0)
    if not pid or sys.platform != "win32":
        return
    import ctypes
    import threading
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
    if not handle:
        return

    def watch():
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def main():
    _exit_with_parent()
    import torch
    import whisper
    model_name = os.environ.get("AUDIOBOOK_ASR_MODEL", "medium")
    cache = os.path.join(os.path.expanduser("~"), ".cache", "whisper", model_name + ".pt")
    if model_name != "small" and not os.path.exists(cache):
        # a model that is not cached would be downloaded (hours on a slow link): fall back to the always-installed small
        sys.stderr.write(f"asr_worker: {cache} not found, using small\n")
        model_name = "small"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=dev)
    real_out = sys.stdout
    sys.stdout = sys.stderr  # nothing but protocol lines may reach the pipe
    real_out.write("@@ready\n")
    real_out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            wav = json.loads(line)["wav"]
            r = model.transcribe(load_16k(wav), language="ru", word_timestamps=True, temperature=0.0,
                                 condition_on_previous_text=False, fp16=(dev == "cuda"), verbose=False)
            words = [{"w": w["word"].strip(), "s": round(w["start"], 2), "e": round(w["end"], 2)}
                     for sg in r["segments"] for w in sg.get("words", [])]
            out = {"words": words}
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        real_out.write("@@" + json.dumps(out, ensure_ascii=True) + "\n")
        real_out.flush()


if __name__ == "__main__":
    main()
