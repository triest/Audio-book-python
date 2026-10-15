"""One place that answers "where are my files?" for BOTH ways of running the program:
  * from source (python fb2_reader.py ...)  -> everything lives next to the .py files;
  * frozen with PyInstaller (one .exe)       -> read-only assets are unpacked to sys._MEIPASS, while
    everything the user edits or the program writes (dictionaries, downloaded models, output) lives next to
    the .exe, never in the temporary unpack folder (it changes on every start).

Lookup order for an asset (resolve_asset):
  1. an environment variable override (when given),
  2. <app_dir>/<rel>      - next to the .exe / script: the user's own copy always wins, so models and
                            dictionaries can be replaced or updated without rebuilding the .exe,
  3. <bundle_dir>/<rel>   - what PyInstaller packed into the .exe (sys._MEIPASS),
  4. <source_dir>/<rel>   - the checkout, for odd layouts.
"""
import os
import sys
from pathlib import Path

_SOURCE_DIR = Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Read-only folder with the packed assets (sys._MEIPASS when frozen, the source folder otherwise)."""
    return Path(getattr(sys, "_MEIPASS", _SOURCE_DIR))


def app_dir() -> Path:
    """Folder of the .exe (frozen) or of the sources: the user-visible, writable program folder."""
    return Path(sys.executable).resolve().parent if is_frozen() else _SOURCE_DIR


def user_file(rel: str) -> Path:
    """A file/folder the user edits or the program writes (dictionaries, caches, output, models)."""
    return app_dir() / rel


def resolve_asset(rel: str, env: "str | None" = None) -> Path:
    """Best existing location of a read-only asset (see module docstring); if none exists, the
    user-visible location is returned so error messages and downloads point to the right place."""
    if env and os.environ.get(env):
        return Path(os.environ[env])
    for base in (app_dir(), bundle_dir(), _SOURCE_DIR):
        p = base / rel
        if p.exists():
            return p
    return app_dir() / rel


def describe() -> str:
    return (f"frozen={is_frozen()} app_dir={app_dir()} bundle_dir={bundle_dir()} "
            f"python={sys.version.split()[0]}")


SEED_FILES = ("stress_dictionary.json", "forced_stress_words.json", "manual_stress_overrides.json",
              "config.json", "fb2_reader_settings.json")


def seed_defaults() -> list:
    """Frozen only: the first start copies the default editable files packed in the .exe next to it, so the user
    edits real files in the .exe folder (a later start never overwrites them). Returns what was copied."""
    copied = []
    if not is_frozen():
        return copied
    import shutil
    for rel in SEED_FILES:
        dst, src = app_dir() / rel, bundle_dir() / rel
        if not dst.exists() and src.exists():
            try:
                shutil.copy2(src, dst)
                copied.append(rel)
            except OSError:
                pass
    return copied
