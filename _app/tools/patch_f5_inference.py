"""Keep non-inference imports lazy in the F5-TTS runtime.

F5-TTS imports its training stack, ASR frontend and plotting library
unconditionally. ESpeech inference needs none of them when the voice profile
has a transcript. Preserve those features, but load each only when requested.
This patch is applied by install.bat after installing f5-tts.
"""

from pathlib import Path
import sys


package_root = Path(sys.prefix) / "Lib" / "site-packages" / "f5_tts"
package_init = package_root / "model" / "__init__.py"
utils_infer = package_root / "infer" / "utils_infer.py"

if not package_init.is_file():
    print("f5-tts is not installed; skipping its inference import optimization.")
    raise SystemExit(0)

source = package_init.read_text(encoding="utf-8")
eager_import = "from f5_tts.model.trainer import Trainer"
trainer_marker = "# AUDIOBOOK_LAZY_TRAINER_IMPORT"
changed = False
if trainer_marker not in source and eager_import not in source:
    print(f"Unexpected f5-tts model initializer; not patching: {package_init}")
    raise SystemExit(1)
lazy_import = '''# AUDIOBOOK_LAZY_TRAINER_IMPORT
def __getattr__(name):
    if name == "Trainer":
        from f5_tts.model.trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''
if trainer_marker not in source:
    package_init.write_text(source.replace(eager_import, lazy_import, 1), encoding="utf-8")
    changed = True

source = utils_infer.read_text(encoding="utf-8")
asr_marker = "# AUDIOBOOK_LAZY_ASR_IMPORT"
eager_asr_import = "from transformers import pipeline"
asr_init = "    global asr_pipe\n    asr_pipe = pipeline("
if asr_marker not in source:
    if eager_asr_import not in source or asr_init not in source:
        print(f"Unexpected F5-TTS inference utilities; not patching: {utils_infer}")
        raise SystemExit(1)
    source = source.replace(eager_asr_import, "", 1)
    source = source.replace(
        asr_init,
        "    global asr_pipe\n    # AUDIOBOOK_LAZY_ASR_IMPORT\n"
        "    from transformers import pipeline\n    asr_pipe = pipeline(",
        1,
    )
    utils_infer.write_text(source, encoding="utf-8")
    changed = True

source = utils_infer.read_text(encoding="utf-8")
plot_marker = "# AUDIOBOOK_LAZY_MATPLOTLIB_IMPORT"
eager_plot_imports = (
    "import matplotlib\n\n\nmatplotlib.use(\"Agg\")\n\nimport matplotlib.pylab as plt"
)
plot_function = "def save_spectrogram(spectrogram, path):\n    plt.figure(figsize=(12, 4))"
if plot_marker not in source:
    if eager_plot_imports not in source or plot_function not in source:
        print(f"Unexpected F5-TTS plotting imports; not patching: {utils_infer}")
        raise SystemExit(1)
    source = source.replace(eager_plot_imports, "", 1)
    source = source.replace(
        plot_function,
        "def save_spectrogram(spectrogram, path):\n"
        "    # AUDIOBOOK_LAZY_MATPLOTLIB_IMPORT\n"
        "    import matplotlib\n"
        "    matplotlib.use(\"Agg\")\n"
        "    import matplotlib.pylab as plt\n"
        "    plt.figure(figsize=(12, 4))",
        1,
    )
    utils_infer.write_text(source, encoding="utf-8")
    changed = True

if changed:
    print("Applied lazy training, ASR and plotting imports for the F5-TTS inference runtime.")
else:
    print("F5-TTS inference import optimizations are already applied.")
