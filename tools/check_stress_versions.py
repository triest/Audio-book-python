r"""Verifies the stress-placement stack against stress_versions.lock.json: package versions, SHA-256 of the
model weights, an inference smoke test with ALL network access disabled (proves offline operation), and
(optionally) records a new lock.

  .venv\Scripts\python.exe check_stress_versions.py            # verify (exit 1 on any mismatch)
  .venv\Scripts\python.exe check_stress_versions.py --write    # record the CURRENT environment as the lock

Upgrade:  install into a COPY first (python -m venv .venv_test; .venv_test\Scripts\pip install silero-stress==X),
run this script + test_accentuation.py + stress_benchmark.py there, and only then change install.bat + --write.
Rollback: pip install the versions listed in stress_versions.lock.json (git keeps older lock files)."""
import gzip
import hashlib
import importlib.metadata as md
import json
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LOCK = HERE / "stress_versions.lock.json"
PACKAGES = ["silero-stress", "silero", "torch", "slovnet", "navec", "razdel", "pymorphy3", "numpy"]
SMOKE = [("Дверной замок заклинило.", "зам+ок"), ("Он сказал себе, что замок пуст.", "з+амок"),
         ("Всё, что он видел.", "вс+ё"), ("Зеленый лес шумел.", "зел+ёный")]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def current() -> dict:
    import silero_stress
    pkg = Path(silero_stress.__file__).parent
    files = {"silero_stress/data/accentor.pt": pkg / "data" / "accentor.pt"}
    for p in sorted((HERE / "grammar_gate_models").glob("*.tar")):
        files["grammar_gate_models/" + p.name] = p
    corpus = HERE / "stress_corpus_ru.tsv.gz"
    if corpus.exists():
        files[corpus.name] = corpus
    return {"python": sys.version.split()[0],
            "packages": {p: (md.version(p) if _has(p) else None) for p in PACKAGES},
            "files": {k: {"sha256": sha(v), "bytes": v.stat().st_size} for k, v in files.items() if v.exists()}}


def _has(p):
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


def offline_smoke() -> list:
    """Runs real inference with sockets disabled; returns a list of problems."""
    real = socket.socket.connect

    def blocked(*a, **k):
        raise OSError("network disabled by check_stress_versions")
    socket.socket.connect = blocked
    problems = []
    try:
        from silero_stress import load_accentor
        acc = load_accentor()
        for text, form in SMOKE:
            out = acc(text, put_stress=True, put_stress_homo=True, put_yo=True, put_yo_homo=True).lower()
            if form not in out:
                problems.append(f"inference: {text!r} -> {out!r}, expected {form!r}")
        for empty in ("", "   ", "...", "— — —", "123"):
            try:
                acc(empty)
            except Exception as e:  # must not crash on odd input
                problems.append(f"crash on {empty!r}: {e}")
    except Exception as e:
        problems.append(f"load/inference failed offline: {e}")
    finally:
        socket.socket.connect = real
    return problems


def main():
    cur = current()
    if "--write" in sys.argv:
        LOCK.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"lock written: {LOCK}")
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else None
    bad = []
    for p, v in cur["packages"].items():
        print(f"  {p:14s} {v}")
        if lock and lock["packages"].get(p) != v:
            bad.append(f"{p}: installed {v}, lock {lock['packages'].get(p)}")
    for k, v in cur["files"].items():
        print(f"  {k:48s} {v['sha256'][:16]}… {v['bytes']} B")
        if lock and lock["files"].get(k, {}).get("sha256") != v["sha256"]:
            bad.append(f"{k}: checksum differs from lock")
    bad += offline_smoke()
    print("python", cur["python"], "| no lock file yet" if not lock else "")
    print("\n".join("MISMATCH " + b for b in bad) if bad else "ALL OK (versions, checksums, offline inference)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
