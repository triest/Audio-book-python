@echo off
rem Sets up a Python environment and installs all dependencies for this
rem program into a .venv folder next to this file. Does not touch the
rem system Python at all -- everything is installed only into that local
rem folder. Safe to run again (e.g. after updating the files) -- anything
rem already installed is simply skipped.
rem
rem NOTE: this file's own messages are in English on purpose -- Windows
rem consoles vary a lot in code page / font support, and Cyrillic text in
rem .bat files has caused garbled output ("mojibake") and broken commands
rem on some systems. The program itself (fb2_reader.py / fb2_reader_gui.py)
rem prints and displays everything in Russian as usual -- only this
rem installer script's own scaffolding text is in English.

setlocal enabledelayedexpansion

rem Optional environments are opt-in so ESpeech-only users do not download
rem XTTS/ASR weights, CosyVoice3, local LLM or unrelated TTS stacks.
rem AUDIOBOOK_INSTALL_ESPEECH_ASR=1 enables Whisper/Transformers ASR.
rem AUDIOBOOK_INSTALL_F5_FALLBACK=1 downloads F5-Russian fallback weights.
rem AUDIOBOOK_INSTALL_XTTS=1 enables the XTTS fallback and its ~2 GB model.
rem AUDIOBOOK_INSTALL_COSYVOICE3=1, AUDIOBOOK_INSTALL_LLM=1 and
rem AUDIOBOOK_INSTALL_OTHER_TTS=1 enable their separate optional stacks.

rem Keep pip quiet and deterministic on repeat runs.  pip already skips a
rem package whose installed version satisfies the requirement; these settings
rem also avoid the version check prompt and prefer cached binary wheels.
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"
set "PIP_PREFER_BINARY=1"
set "PIP_UPGRADE_STRATEGY=only-if-needed"

cd /d "%~dp0"

rem Re-launch itself once, piping all output (both what you see on screen
rem AND everything below) through PowerShell's Tee-Object into
rem install_log.txt next to this file - so if something goes wrong, you
rem can just send that file instead of copy-pasting the console. The
rem INSTALL_LOG_TEE marker prevents this from looping forever (the
rem re-launched copy inherits it and skips straight past this block).
if not defined INSTALL_LOG_TEE (
    set "INSTALL_LOG_TEE=1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~f0' 2>&1 | Tee-Object -FilePath '%~dp0install_log.txt'"
    exit /b
)

echo ===============================================
echo   FB2 Audiobook Reader - installing dependencies
echo ===============================================
echo.

rem --- find Python ---
set "PYLAUNCHER="
where py >nul 2>nul
if not errorlevel 1 set "PYLAUNCHER=py -3"
if not defined PYLAUNCHER (
    where python >nul 2>nul
    if not errorlevel 1 set "PYLAUNCHER=python"
)

if not defined PYLAUNCHER (
    echo Python was not found on this computer.
    echo.
    echo Install Python 3.10 or newer from:
    echo   https://www.python.org/downloads/
    echo During install, make sure to check the box at the bottom of the
    echo first screen: "Add python.exe to PATH"
    echo Then run this file again.
    echo.
    pause
    exit /b 1
)

echo Found Python:
%PYLAUNCHER% --version
echo.

rem --- create the virtual environment, if it does not exist yet ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv environment ...
    %PYLAUNCHER% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment - see the message above.
        pause
        exit /b 1
    )
    echo Done.
    echo.
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

rem Narrow repair path for an interrupted CosyVoice/F5 dependency upgrade.
rem Set AUDIOBOOK_REPAIR_ESPEECH_RUNTIME=1 to repair only the FastAPI
rem runtime and apply the F5-TTS inference-only import optimization.
if defined AUDIOBOOK_REPAIR_ESPEECH_RUNTIME (
    set "CV_PY=%~dp0.venv_cosyvoice\Scripts\python.exe"
    if not exist "!CV_PY!" (
        echo CosyVoice environment is missing - run install.bat normally first.
        exit /b 1
    )
    "!CV_PY!" -m pip install --no-deps "pydantic==2.14.3" "pydantic-core==2.50.4"
    if errorlevel 1 exit /b 1
    "!CV_PY!" _app\tools\patch_f5_inference.py
    if errorlevel 1 exit /b 1
    exit /b 0
)

echo Checking dependencies and model files...
echo Missing or incompatible packages will be downloaded automatically.
echo.

"%VENV_PY%" -m pip install pip --quiet

rem Base dependencies - needed almost always (fb2 parsing, GUI, downloads)
"%VENV_PY%" -m pip install lxml numpy tqdm requests
if errorlevel 1 goto :warn_partial

rem Offline stress dictionary (~1.68M Russian words/forms, ~10 MB gzipped,
rem source: https://github.com/Koziev/NLP_Datasets) - used by fb2_reader.py
rem as the first (fast, no network) source for word stress when a word is
rem not in stress_dictionary.json, well before it would fall back to
rem looking the word up online (Wiktionary), which cuts down a lot on how
rem often that online fallback is even needed - and how often it can hit
rem the site's own rate limit (HTTP 429) on a long book. Best-effort only:
rem _app\tools\download_stress_corpus.py never fails the install if this can't be
rem fetched (no internet right now, GitHub unreachable, etc.) - it just
rem prints a warning and fb2_reader.py keeps working without it.
"%VENV_PY%" _app\tools\download_stress_corpus.py

rem Local Silero synthesis (the main, recommended mode)
rem CUDA build of torch in the MAIN venv (plain PyPI "torch" is CPU-only; Silero TTS can use the GPU).
rem Skip the download when any working CUDA build is already present.  The
rem old command installed torch/torchaudio again on every run because the
rem unqualified package names could resolve to a different wheel index.
"%VENV_PY%" -c "import importlib.metadata as m,sys; from pathlib import Path; t=m.version('torch'); a=m.version('torchaudio'); r=Path(m.distribution('torch').locate_file('torch')); ar=Path(m.distribution('torchaudio').locate_file('torchaudio')); ok='+cu' in t and t.split('+')[0]==a.split('+')[0] and any(p.is_file() and p.stat().st_size>1024 for p in r.glob('_C*.pyd')) and all((r/'lib'/n).is_file() and (r/'lib'/n).stat().st_size>1024 for n in ('torch_cuda.dll','c10_cuda.dll')) and any(p.is_file() and p.stat().st_size>1024 for p in (ar/'lib').glob('*.pyd')); sys.exit(0 if ok else 1)" 2>nul
if errorlevel 1 (
    "%VENV_PY%" -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 "%VENV_PY%" -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
)
"%VENV_PY%" -m pip install silero omegaconf
if errorlevel 1 goto :warn_partial

rem Silero via a separate REST service (SSML pauses, optional)
"%VENV_PY%" -m pip install fastapi uvicorn num2words
if errorlevel 1 goto :warn_partial

rem silero-stress - the ONLY accentuation model (context-aware stress, ~0.5 s per chapter chunk on CPU).
rem pinned to the verified version (stress_versions.lock.json); upgrade procedure: see _app\tools\check_stress_versions.py
"%VENV_PY%" -m pip install silero-stress==1.5
if errorlevel 1 echo WARNING: silero-stress failed to install - there will be no automatic stress marks ^(only the dictionary^).

rem faster-whisper - speech recognition (ASR) for checking finished chapters (_app\tools\audio_verify.py: coverage/WER/CER).
rem Dev/QA tool only: the app does not import it, so it is not packed into the .exe. Optional: a failure is not fatal.
rem The recognition model itself is downloaded from Hugging Face on first use, not here.
"%VENV_PY%" -m pip install faster-whisper "av>=11,<16"
if errorlevel 1 echo WARNING: faster-whisper failed to install - audio checks by speech recognition will be unavailable.

rem Leftovers of the removed RUAccent / ruaccent-predictor engines: drop them so they cannot bloat the .exe.
"%VENV_PY%" -m pip uninstall -y ruaccent ruaccent-predictor >nul 2>nul
if exist "%~dp0ruaccent_predictor_pkg" rmdir /s /q "%~dp0ruaccent_predictor_pkg"

rem Grammar gate (case/number stress homographs): slovnet morphology tagger + 29 MB of models. Optional:
rem if anything here fails the gate stays off and the rest of the pipeline is unaffected.
"%VENV_PY%" -m pip install slovnet navec razdel
"%VENV_PY%" _app\fb2_reader.py fetch-models
if errorlevel 1 echo WARNING: grammar models were not downloaded - the grammar gate stays off ^(and the .exe will not contain them^) until you run install.bat again.

rem PyInstaller - builds the single-file .exe at the end of this script.
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 echo WARNING: pyinstaller failed to install - the .exe build step below will be skipped.

rem pymorphy3 - Russian morphology analyzer, used to pick the correct
rem stress for a handful of common words whose stress depends on
rem grammatical case ("??????" - genitive singular vs nominative plural,
rem etc. - see resolve_case_ambiguous_nouns in fb2_reader.py). Optional:
rem if it fails to install, that specific feature is silently skipped and
rem everything else keeps working as before.
"%VENV_PY%" -m pip install pymorphy3 pymorphy3-dicts-ru
if errorlevel 1 echo WARNING: pymorphy3 failed to install - context-based stress for a few case-ambiguous words (e.g. "??????") will be less accurate.

rem dashscope - client library for Qwen3-TTS (Alibaba's cloud TTS API,
rem "qwen_tts" mode). Optional: only needed if you actually use that mode;
rem if it fails to install, the other TTS modes (Silero, Yandex, etc.) are
rem unaffected - qwen_tts mode will just show a clear error until it's
rem installed.
"%VENV_PY%" -m pip install dashscope
if errorlevel 1 echo WARNING: dashscope failed to install - the "Qwen3-TTS" mode will not work until it is installed manually.

rem gradio_client - talks to a locally running Qwen3-TTS Gradio server
rem ("qwen_tts_local" mode) - no cloud account/card needed, runs on your own
rem GPU via a separately installed server (e.g. through Pinokio). Optional,
rem same as dashscope above.
"%VENV_PY%" -m pip install gradio_client
if errorlevel 1 echo WARNING: gradio_client failed to install - the "Qwen3-TTS (????????)" mode will not work until it is installed manually.

rem Some of the packages above can pull in an older version of click
rem (used by third-party dependencies like huggingface-hub) - pip
rem sometimes prints a warning about this: "dependency resolver does
rem not... conflicts". That by itself is NOT an error (everything still
rem installs and the program works) - but we pin a compatible version
rem explicitly so the warning stops appearing, just in case.
"%VENV_PY%" -m pip install "click>=8.4.2" --quiet

rem Google TTS and system TTS (optional modes)
"%VENV_PY%" -m pip install gTTS pyttsx3 pydub
if errorlevel 1 goto :warn_partial

"%VENV_PY%" -m pip check >nul
if errorlevel 1 echo WARNING: dependency check found conflicts in the main environment.

rem Piper TTS (optional mode) - small/fast local CPU voices, no cloning.
rem Voice files (.onnx) are downloaded on first use into piper_voices/
rem (see .gitignore) - not needed here, just the engine itself.
"%VENV_PY%" -m pip install piper-tts
if errorlevel 1 (
    echo WARNING: piper-tts failed to install - the "Piper" mode will not be
    echo available, everything else still works. You can try running
    echo install.bat again.
)

rem Built-in player in the program window (pause/stop/seek while
rem listening to chapters) - we check not just that pip reported
rem success, but that pygame actually imports (sometimes a wheel
rem installs but does not match the Python version/system, and then
rem pip itself will not show an error, but the module still will not work).
rem "pygame" on PyPI sometimes has no ready-made wheel yet for the very
rem latest Python versions, which makes pip try to compile it from
rem source and fail (needs a C/C++ compiler that most computers do not
rem have). "pygame-ce" is a community-maintained drop-in replacement
rem (same "import pygame" in code) that tends to publish wheels for new
rem Python versions faster - we try it automatically as a fallback.
"%VENV_PY%" -m pip install pygame --only-binary :all: --quiet
if not errorlevel 1 (
    "%VENV_PY%" -c "import pygame"
    if not errorlevel 1 goto :pygame_ok
)

echo Plain "pygame" has no ready-made package for this Python version -
echo trying "pygame-ce" instead ^(same thing, different distribution^)...
"%VENV_PY%" -m pip uninstall pygame -y --quiet >nul 2>nul
"%VENV_PY%" -m pip install pygame-ce --only-binary :all: --quiet
if errorlevel 1 goto :warn_pygame
"%VENV_PY%" -c "import pygame"
if errorlevel 1 goto :warn_pygame
goto :pygame_ok

:warn_pygame
echo.
echo WARNING: pygame failed to install or does not run on this Python.
echo The program will still work, but the built-in player (pause/stop/
echo seek while listening to chapters) will not be available - the
echo system's default player will open instead, without those controls.
echo This is usually because the Python version is too new and pygame
echo does not have a ready-made package for it yet - try installing
echo Python 3.11 or 3.12 from https://www.python.org/downloads/ (it can
echo be installed side by side with the existing version), delete the
echo .venv folder, and run install.bat again so it rebuilds the
echo environment on that Python instead.

:pygame_ok
call :build_exe
echo.
echo Main dependencies installed successfully.
echo (If you saw a warning above like "dependency resolver does not
echo currently take into account..." about click or another package -
echo that is not an error, just an informational pip warning; as long as
echo it says "Successfully installed" above, everything is fine and the
echo program will work.)
echo.

rem ===============================================
rem   CosyVoice (neural TTS with voice cloning)
rem   Optional, separate environment (.venv_cosyvoice).
rem   Failures here are NOT fatal - the rest of the
rem   program works fine without this mode.
rem ===============================================
echo ===============================================
echo   CosyVoice (voice cloning) - optional setup
echo ===============================================
echo.

if not exist "CosyVoice" (
    where git >nul 2>nul
    if errorlevel 1 (
        echo WARNING: git was not found - cannot download CosyVoice.
        echo Install git from https://git-scm.com/downloads and run
        echo install.bat again if you want the voice-cloning mode.
        goto :cosyvoice_done
    )
    echo Downloading CosyVoice ^(this can take a minute^)...
    git clone --depth 1 --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git
    if errorlevel 1 (
        echo WARNING: failed to download CosyVoice - voice cloning mode
        echo will not be available. You can try running install.bat again.
        goto :cosyvoice_done
    )
)

rem Older clones (before this fix) may be missing the Matcha-TTS submodule
rem content, which CosyVoice needs internally ("import matcha...") - make
rem sure it is present and up to date every run, just in case.
if exist "CosyVoice\.git" (
    pushd CosyVoice
    git submodule update --init --recursive
    popd
)

rem --- find a Python version CosyVoice's pinned dependencies support
rem     (torch 2.3.1 etc. do not have wheels for very new Python
rem     versions) - prefer 3.10, then 3.11, then 3.9, in that order.
set "CV_PYLAUNCHER="
for %%V in (3.10 3.11 3.9) do (
    if not defined CV_PYLAUNCHER (
        py -%%V --version >nul 2>nul
        if not errorlevel 1 set "CV_PYLAUNCHER=py -%%V"
    )
)

set "CV_PINNED=1"
if not defined CV_PYLAUNCHER (
    echo No Python 3.9/3.10/3.11 found - CosyVoice's exact tested package
    echo versions may not be available for your Python. Will try with
    echo the newest available versions instead ^(may or may not work^).
    set "CV_PYLAUNCHER=%PYLAUNCHER%"
    set "CV_PINNED=0"
)

if not exist ".venv_cosyvoice\Scripts\python.exe" (
    echo Creating .venv_cosyvoice environment using: !CV_PYLAUNCHER!
    !CV_PYLAUNCHER! -m venv .venv_cosyvoice
    if errorlevel 1 (
        echo WARNING: failed to create .venv_cosyvoice - voice cloning mode
        echo will not be available.
        goto :cosyvoice_done
    )
)

set "CV_PY=%~dp0.venv_cosyvoice\Scripts\python.exe"

rem setuptools/wheel are required to build openai-whisper (a dependency of
rem f5-tts) from source if the exact pinned version it wants is not
rem available as a pre-built wheel. Just "--upgrade setuptools" is NOT
rem enough (and was the actual root cause of repeated "ModuleNotFoundError:
rem No module named 'pkg_resources'" failures, even with --no-build-isolation
rem below) - setuptools removed the pkg_resources module entirely starting
rem around version 81, and openai-whisper's old setup.py still imports it.
rem Pinning below that keeps pkg_resources available.
"%CV_PY%" -m pip install pip wheel --quiet
"%CV_PY%" -m pip install "setuptools<81" --quiet

echo Installing CosyVoice dependencies - each package is installed
echo separately on purpose, so that one failure does not block the rest.
echo.

"%CV_PY%" -m pip install fastapi
if errorlevel 1 echo WARNING: fastapi failed to install.
"%CV_PY%" -m pip install uvicorn
if errorlevel 1 echo WARNING: uvicorn failed to install.
"%CV_PY%" -m pip install python-multipart
if errorlevel 1 echo WARNING: python-multipart failed to install.
"%CV_PY%" -m pip install soundfile
if errorlevel 1 echo WARNING: soundfile failed to install.
"%CV_PY%" -m pip install num2words
if errorlevel 1 echo WARNING: num2words failed to install - numbers in text may be read incorrectly or skipped.

echo Installing torch/torchaudio ^(this is the big one, GB-sized^)...
rem NOTE: plain "pip install torch" pulls the CPU-only build - it has no
rem idea an NVIDIA GPU exists unless we explicitly point it at PyTorch's own
rem CUDA package index. We try that first (much bigger download, includes
rem CUDA runtime), and only fall back to the plain CPU build if it fails
rem (e.g. on a computer with no NVIDIA GPU, this index may not have a
rem matching wheel, or the user is offline from a different mirror).
rem
rem NOTE 2: deliberately NOT pinned to torch==2.3.1 anymore. That pin was
rem for CosyVoice's own requirements.txt, which is no longer used (the
rem voice cloning engine was switched to XTTS-v2/coqui-tts - see below).
rem coqui-tts pulls in a recent "transformers", which as of writing
rem refuses to use any torch older than 2.5 (it silently falls back to
rem "no GPU / CPU-only-ish" behavior instead of a hard error, which is
rem confusing) - so we install an unpinned, current torch/torchaudio here
rem instead of an old fixed version.
rem Do not use --force-reinstall here.  It deleted several GB of perfectly
rem good wheels on every run, even when the requested versions were unchanged.
"%CV_PY%" -c "import importlib.metadata as m,sys; from pathlib import Path; t=m.version('torch'); a=m.version('torchaudio'); r=Path(m.distribution('torch').locate_file('torch')); ar=Path(m.distribution('torchaudio').locate_file('torchaudio')); ok=t in ('2.11.0+cu126','2.11.0+cu124') and a==t and all((r/'lib'/n).is_file() and (r/'lib'/n).stat().st_size>1024 for n in ('torch_cuda.dll','c10_cuda.dll')) and any(p.is_file() and p.stat().st_size>1024 for p in r.glob('_C*.pyd')) and any(p.is_file() and p.stat().st_size>1024 for p in (ar/'lib').glob('*.pyd')); sys.exit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 (
    "%CV_PY%" -m pip install --no-deps torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo WARNING: CUDA ^(cu126^) build of torch failed - trying cu124 index...
        "%CV_PY%" -c "import importlib.metadata as m,sys; sys.exit(0 if m.version('torch')=='2.11.0+cu124' and m.version('torchaudio')=='2.11.0+cu124' else 1)" >nul 2>nul
        if errorlevel 1 "%CV_PY%" -m pip install --no-deps torch==2.11.0+cu124 torchaudio==2.11.0+cu124 --index-url https://download.pytorch.org/whl/cu124
        if errorlevel 1 (
            echo WARNING: CUDA ^(cu124^) build failed - trying the plain CPU build...
            "%CV_PY%" -m pip install torch torchaudio
            if errorlevel 1 echo WARNING: torch/torchaudio failed to install.
        )
    )
)

if "!CV_PINNED!"=="1" (
"%CV_PY%" -m pip install "numpy>=2.0,<2.6"
    if errorlevel 1 echo WARNING: numpy failed to install.
) else (
    "%CV_PY%" -m pip install "numpy>=2.0,<2.6"
    if errorlevel 1 echo WARNING: numpy failed to install.
)

echo Installing only the ESpeech/F5 inference runtime packages...
"%CV_PY%" -m pip install librosa pydub pypinyin rjieba torchdiffeq vocos "x-transformers==2.11.24" tqdm huggingface_hub num2words
if errorlevel 1 echo WARNING: one or more ESpeech inference packages failed to install.

rem ASR is only needed to transcribe voice references that do not provide a
rem transcript. The ESpeech synthesis path works without these packages when
the voice profile includes its reference text. Opt in with
rem AUDIOBOOK_INSTALL_ESPEECH_ASR=1 to install them and fetch Whisper weights.
if "%AUDIOBOOK_INSTALL_ESPEECH_ASR%"=="1" (
    "%CV_PY%" -m pip install openai-whisper==20231117 "transformers==4.51.3"
    if errorlevel 1 echo WARNING: optional ESpeech ASR dependencies failed to install.
    "%CV_PY%" _app\tools\fetch_whisper_model.py small
    if errorlevel 1 echo WARNING: Whisper small weights were not downloaded.
)

"%CV_PY%" -m pip check >nul
if errorlevel 1 echo WARNING: dependency check found conflicts in the CosyVoice environment.

rem The GUI sources live in _app, while this installer runs from the
rem repository root.  Copy the service from the actual source location;
rem keep the root-level path as a compatibility fallback for old layouts.
if exist "_app\cosyvoice_rest_service.py" (
    if not exist "_app\CosyVoice" mkdir "_app\CosyVoice"
    copy /Y "_app\cosyvoice_rest_service.py" "_app\CosyVoice\cosyvoice_rest_service.py" >nul
) else if exist "cosyvoice_rest_service.py" (
    copy /Y "cosyvoice_rest_service.py" "CosyVoice\cosyvoice_rest_service.py" >nul
)
if exist "_app\CosyVoice\cosyvoice_rest_service.py" (
    set "CV_MODEL_DIR=%~dp0_app/CosyVoice"
) else (
    set "CV_MODEL_DIR=%~dp0CosyVoice"
)

rem The service uses ESpeech RL-V2 as the selected Russian engine. Other
rem model weights and fallback runtimes are optional and are not downloaded
rem in an ESpeech-only install.
echo.
echo Installing f5-tts ^(ESpeech RL-V2 inference runtime^)...
rem --no-build-isolation: pip's build isolation builds packages from source
rem (like the old openai-whisper==20231117 that f5-tts pins, which has no
rem pre-built wheel for newer Python) inside a throwaway "overlay"
rem environment that does NOT inherit the setuptools/wheel installed above
rem into .venv_cosyvoice itself - so even with them installed, the build
rem still failed with "No module named 'pkg_resources'". Without isolation,
rem it builds using .venv_cosyvoice directly, which does have them.
rem Keep the already verified Torch/CUDA build intact. F5-TTS declares many
rem training/UI/ASR dependencies; the inference runtime list above is the
rem only set used by ESpeech when a reference transcript is present.
"%CV_PY%" -m pip install "f5-tts==1.1.22" huggingface_hub --no-build-isolation --no-deps
"%CV_PY%" _app\tools\patch_f5_inference.py
if errorlevel 1 echo WARNING: could not apply the F5-TTS inference import optimization.
rem ESpeech is the selected engine: prefetch only its 2.7 GB checkpoint and
rem the small Vocos vocoder. hf_hub_download reuses valid local files/cache.
"%CV_PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('charactr/vocos-mel-24khz')"
if errorlevel 1 echo WARNING: could not pre-download the Vocos vocoder; service will try again on first use.
"%CV_PY%" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ESpeech/ESpeech-TTS-1_RL-V2', filename='espeech_tts_rlv2.pt', local_dir='%CV_MODEL_DIR%/pretrained_models/espeech'); hf_hub_download(repo_id='ESpeech/ESpeech-TTS-1_RL-V2', filename='vocab.txt', local_dir='%CV_MODEL_DIR%/pretrained_models/espeech')"
if errorlevel 1 echo WARNING: could not pre-download the ESpeech checkpoint; service will try again on first use.

if "%AUDIOBOOK_INSTALL_F5_FALLBACK%"=="1" (
    echo Installing optional F5-Russian fallback weights...
    "%CV_PY%" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hotstone228/F5-TTS-Russian', filename='model_last.safetensors', local_dir='%CV_MODEL_DIR%/pretrained_models/f5-tts-russian'); hf_hub_download(repo_id='hotstone228/F5-TTS-Russian', filename='vocab.txt', local_dir='%CV_MODEL_DIR%/pretrained_models/f5-tts-russian'); hf_hub_download(repo_id='Misha24-10/F5-TTS_RUSSIAN', filename='F5TTS_v1_Base_v4_winter/model_212000.safetensors', local_dir='%CV_MODEL_DIR%/pretrained_models/f5-tts-russian-winter'); hf_hub_download(repo_id='Misha24-10/F5-TTS_RUSSIAN', filename='F5TTS_v1_Base/vocab.txt', local_dir='%CV_MODEL_DIR%/pretrained_models/f5-tts-russian-winter')"
    if errorlevel 1 echo WARNING: F5-Russian fallback weights were not downloaded.
)

rem Optional XTTS fallback. It downloads Coqui/Transformers packages and a
rem separate ~2 GB model; keep it opt-in for ESpeech-only installations.
if "%AUDIOBOOK_INSTALL_XTTS%"=="1" (
echo.
echo Installing optional coqui-tts ^(XTTS-v2 fallback^)...
"%CV_PY%" -m pip install coqui-tts
if errorlevel 1 (
    echo WARNING: coqui-tts failed to install - voice cloning mode will not
    echo work. You can try running install.bat again.
) else (
    rem coqui-tts 0.27.5 itself requires transformers>=4.57 (see its
    rem pyproject.toml) - but the newest transformers available at
    rem install time can be even newer and has, in practice, removed
    rem internals that XTTS's own code still imports (e.g. "cannot import
    rem name 'isin_mps_friendly' from 'transformers.pytorch_utils'").
    rem Going too far the other way (e.g. 4.49) breaks it just as badly in
    rem the opposite direction ("cannot import name
    rem 'is_torchcodec_available'" - a newer addition XTTS's __init__.py
    rem also needs). Pin to the 4.57.x line specifically - new enough to
    rem have is_torchcodec_available, old enough to still have
    rem isin_mps_friendly.
    "%CV_PY%" -m pip install "transformers>=4.57,<4.58"
    if errorlevel 1 echo WARNING: could not pin transformers to the 4.57.x line - XTTS may fail to load.
    rem NOTE: an earlier version of this script re-pinned torch back to
    rem 2.3.1+cu121 here "just in case" - that was backwards and actively
    rem broke things: coqui-tts's "transformers" dependency refuses to use
    rem any torch older than 2.5, so forcing 2.3.1 back made XTTS fail to
    rem load with a misleading "PyTorch not found" error even though torch
    rem imported fine on its own. Do NOT downgrade torch after coqui-tts -
    rem only reinstall it if it is missing entirely (pip leaves an
    rem already-satisfying version alone, so this is a harmless no-op most
    rem of the time).
    "%CV_PY%" -c "import torch" >nul 2>nul
    if errorlevel 1 (
        echo torch missing after coqui-tts install - reinstalling CUDA build...
        "%CV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
        if errorlevel 1 "%CV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    )

    rem ruaccent auto-marks stressed vowels in Russian text ("+" before the
    rem stressed vowel) before it goes into XTTS - XTTS-v2
    rem has no dedicated stress module for Russian (unlike English) and
    rem regularly stresses the wrong syllable without this; the "+" marker
    rem matches the convention used by the Russian TTS corpora XTTS itself
    rem was trained on. Not fatal if this fails to install - the service
    rem just synthesizes without stress marks in that case.
    echo Installing ruaccent ^(Russian stress marking, improves pronunciation^)...
    "%CV_PY%" -m pip install ruaccent
    if errorlevel 1 echo WARNING: ruaccent failed to install - Russian stress placement may be less accurate.

    echo.
    echo Pre-downloading the XTTS-v2 model ^(~2 GB, only needs to happen
    echo once - if this step is skipped or fails, the service will just
    echo download it the first time voice cloning is used instead^)...
    set "COQUI_TOS_AGREED=1"
    "%CV_PY%" -c "import os; os.environ['COQUI_TOS_AGREED']='1'; from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
    if errorlevel 1 (
        echo WARNING: could not pre-download the XTTS-v2 model - it will be
        echo downloaded automatically the first time you use voice cloning
        echo instead ^(just makes the very first use slower^).
    )
)
)

rem Keep FastAPI's Pydantic runtime internally consistent. A previous
rem interrupted pip resolver left pydantic 2.7.0 next to pydantic-core
rem 2.50.4, making the REST service fail at import time.
"%CV_PY%" -c "import sys; from importlib.metadata import version; sys.exit(0 if version('pydantic')=='2.14.3' and version('pydantic-core')=='2.50.4' else 1)" >nul 2>nul
if errorlevel 1 "%CV_PY%" -m pip install --no-deps "pydantic==2.14.3" "pydantic-core==2.50.4"
if errorlevel 1 echo WARNING: Pydantic versions are inconsistent - the CosyVoice REST service may fail to start.

echo.
echo ESpeech setup finished ^(see any WARNING lines above for details^).
echo If the ESpeech dependencies and model files are ready, the voice mode
echo will start its background service when selected in the program.

rem ===============================================
rem   CosyVoice3 (FunAudioLLM/CosyVoice, Fun-CosyVoice3-0.5B)
rem   Optional, SEPARATE environment (.venv_cosyvoice3) and
rem   SEPARATE folder (CosyVoice3\) - its pinned dependency
rem   versions (torch==2.3.1 etc.) are incompatible with what
rem   .venv_cosyvoice already has installed above for F5-TTS/
rem   XTTS/ESpeech (newer torch), so it cannot share that venv.
rem   Failures here are NOT fatal - everything above still works.
rem
rem   (This used to be a Docker container instead - see
rem   docker\cosyvoice3\ - but Docker Desktop/WSL2 turned out too
rem   unreliable on the target machine, so it now runs the same
rem   way as the rest of CosyVoice: a plain subprocess in its own
rem   venv on Windows directly.)
rem ===============================================
echo.
echo ===============================================
echo   CosyVoice3 (experimental) - optional setup
echo ===============================================
echo.
if not "%AUDIOBOOK_INSTALL_COSYVOICE3%"=="1" (
    echo Skipping CosyVoice3. Set AUDIOBOOK_INSTALL_COSYVOICE3=1 to install it.
    goto :cosyvoice3_done
)

if not exist "CosyVoice3" (
    where git >nul 2>nul
    if errorlevel 1 (
        echo WARNING: git was not found - cannot download CosyVoice3.
        echo Install git from https://git-scm.com/downloads and run
        echo install.bat again if you want this engine.
        goto :cosyvoice3_done
    )
    echo Downloading CosyVoice3 ^(FunAudioLLM/CosyVoice, this can take a
    echo minute^)...
    git clone --depth 1 --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git CosyVoice3
    if errorlevel 1 (
        echo WARNING: failed to download CosyVoice3 - this engine will not
        echo be available. You can try running install.bat again.
        goto :cosyvoice3_done
    )
)

if exist "CosyVoice3\.git" (
    pushd CosyVoice3
    git submodule update --init --recursive
    popd
)

rem Same pinned-dependency constraint as the CosyVoice3 requirements.txt
rem (torch==2.3.1 etc. do not have wheels for very new Python versions) -
rem reuse whichever suitable interpreter was already found above for the
rem main CosyVoice setup, if any; otherwise look again.
set "CV3_PYLAUNCHER=%CV_PYLAUNCHER%"
if not defined CV3_PYLAUNCHER (
    for %%V in (3.10 3.11 3.9) do (
        if not defined CV3_PYLAUNCHER (
            py -%%V --version >nul 2>nul
            if not errorlevel 1 set "CV3_PYLAUNCHER=py -%%V"
        )
    )
)
if not defined CV3_PYLAUNCHER set "CV3_PYLAUNCHER=%PYLAUNCHER%"

if not exist ".venv_cosyvoice3\Scripts\python.exe" (
    echo Creating .venv_cosyvoice3 environment using: !CV3_PYLAUNCHER!
    !CV3_PYLAUNCHER! -m venv .venv_cosyvoice3
    if errorlevel 1 (
        echo WARNING: failed to create .venv_cosyvoice3 - CosyVoice3 will
        echo not be available.
        goto :cosyvoice3_done
    )
)

set "CV3_PY=%~dp0.venv_cosyvoice3\Scripts\python.exe"

rem setuptools/wheel are required to build openai-whisper (one of
rem CosyVoice3's dependencies) from source. Just "--upgrade setuptools" is
rem NOT enough - setuptools removed the pkg_resources module entirely
rem starting around version 81, and openai-whisper's old setup.py still
rem imports it, which is what actually caused the repeated
rem "ModuleNotFoundError: No module named 'pkg_resources'" failures (even
rem with --no-build-isolation below). Pinning below that keeps pkg_resources
rem available.
"%CV3_PY%" -m pip install --upgrade pip wheel --quiet
"%CV3_PY%" -m pip install "setuptools<81" --quiet

echo Installing CosyVoice3 dependencies - this is a large install (PyTorch
echo with CUDA, several audio/ML libraries) and can take a long while,
echo especially on a slow connection.
rem --no-build-isolation - see the comment on the f5-tts install above for
rem why this is needed (same openai-whisper build issue applies here).
"%CV3_PY%" -m pip install -r CosyVoice3\requirements.txt --no-build-isolation
if errorlevel 1 (
    echo WARNING: some CosyVoice3 dependencies failed to install - this
    echo engine may not work. You can try running install.bat again.
) else (
    "%CV3_PY%" -m pip install num2words huggingface_hub
    copy /Y "cosyvoice3_rest_service.py" "CosyVoice3\cosyvoice3_rest_service.py" >nul
    "%CV3_PY%" -m pip check >nul
    if errorlevel 1 echo WARNING: dependency check found conflicts in the CosyVoice3 environment.
    echo.
    echo CosyVoice3 dependencies installed. Model weights ^(~1-2 GB^) are
    echo NOT downloaded here - they download automatically the first time
    echo you select the "CosyVoice 3" engine in the program ^(can take a
    echo while, watch the program's log window for progress^).
)

:cosyvoice3_done
echo.

:cosyvoice_done
echo.

rem ===============================================
rem   Built-in local LLM (llama-cpp-python + Qwen2.5-7B-Instruct GGUF)
rem   Optional, SEPARATE environment (.venv_llm). Used only by the
rem   "Ask LLM" button on the stress-review tab - the audiobook
rem   pipeline itself stays LLM-free. Prebuilt CUDA 12.4 wheel (no
rem   compiler needed). Failures here are NOT fatal.
rem ===============================================
echo.
echo ===============================================
echo   Built-in local LLM - optional setup
echo ===============================================
if not "%AUDIOBOOK_INSTALL_LLM%"=="1" (
    echo Skipping local LLM. Set AUDIOBOOK_INSTALL_LLM=1 to install it.
    goto :llm_done
)
set "LLM_PYLAUNCHER="
for %%V in (3.11 3.12 3.10) do (
    if not defined LLM_PYLAUNCHER (
        py -%%V --version >nul 2>nul
        if not errorlevel 1 set "LLM_PYLAUNCHER=py -%%V"
    )
)
if not defined LLM_PYLAUNCHER (
    echo WARNING: Python 3.10-3.12 not found - built-in LLM will not be available.
    goto :llm_done
)
if not exist ".venv_llm\Scripts\python.exe" (
    !LLM_PYLAUNCHER! -m venv .venv_llm
    if errorlevel 1 (
        echo WARNING: failed to create .venv_llm - built-in LLM will not be available.
        goto :llm_done
    )
)
set "LLM_PY=%~dp0.venv_llm\Scripts\python.exe"
"%LLM_PY%" -m pip install --upgrade pip --quiet
"%LLM_PY%" -m pip install "llama-cpp-python==0.3.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --only-binary=llama-cpp-python
if errorlevel 1 (
    echo WARNING: llama-cpp-python CUDA wheel failed to install - built-in LLM will not be available.
    goto :llm_done
)
"%LLM_PY%" -m pip install "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*" fastapi uvicorn huggingface_hub
"%LLM_PY%" "%~dp0llm_rest_service.py" --download
if errorlevel 1 (
    echo WARNING: failed to download the LLM weights ^(~4.7 GB^) - run install.bat again.
) else (
    echo Built-in LLM is ready: the "Ask LLM" button uses it ^(starts on first use, unloads from VRAM when idle^).
)

:llm_done
echo.

rem Optional local VoxCPM2 environment (Python 3.10-3.12; kept separate from
rem the main Python 3.14 environment and from Silero/ESpeech).
if not "%AUDIOBOOK_INSTALL_OTHER_TTS%"=="1" (
    echo Skipping VoxCPM2 and Kokoro-RU. Set AUDIOBOOK_INSTALL_OTHER_TTS=1 to install them.
    goto :optional_tts_done
)
if not exist "%~dp0.venv_voxcpm\Scripts\python.exe" (
    echo Creating optional .venv_voxcpm environment...
    py -3.11 -m venv "%~dp0.venv_voxcpm"
)
if exist "%~dp0.venv_voxcpm\Scripts\python.exe" (
    set "VOX_PY=%~dp0.venv_voxcpm\Scripts\python.exe"
    "!VOX_PY!" -m pip install --upgrade pip wheel --quiet
    "!VOX_PY!" -m pip install voxcpm fastapi uvicorn soundfile
    "!VOX_PY!" -c "import sys,torch,torchaudio; sys.exit(0 if torch.__version__.startswith('2.11.0+cu126') and torchaudio.__version__.startswith('2.11.0+cu126') else 1)" >nul 2>nul
    if errorlevel 1 "!VOX_PY!" -m pip install --no-deps torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    "!VOX_PY!" -c "from voxcpm import VoxCPM; print('VoxCPM2 environment is ready')"
    if errorlevel 1 echo WARNING: VoxCPM2 environment check failed - the mode will be unavailable.
) else echo WARNING: Python 3.11 was not found - skipping optional VoxCPM2 environment.
echo.

rem Optional local Kokoro-RU environment. It is isolated because its torch and
rem kokoro frontend versions must not alter the proven ESpeech environment.
if not exist "%~dp0.venv_kokoro\Scripts\python.exe" (
    echo Creating optional .venv_kokoro environment...
    py -3.11 -m venv "%~dp0.venv_kokoro"
)
if exist "%~dp0.venv_kokoro\Scripts\python.exe" (
    set "KOKORO_PY=%~dp0.venv_kokoro\Scripts\python.exe"
    "!KOKORO_PY!" -m pip install --upgrade pip wheel --quiet
    "!KOKORO_PY!" -m pip install kokoro ruaccent huggingface_hub fastapi uvicorn soundfile
    "!KOKORO_PY!" -c "import sys,torch,torchaudio; sys.exit(0 if torch.__version__.startswith('2.11.0+cu126') and torchaudio.__version__.startswith('2.11.0+cu126') else 1)" >nul 2>nul
    if errorlevel 1 "!KOKORO_PY!" -m pip install --no-deps torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    "!KOKORO_PY!" -c "from huggingface_hub import snapshot_download; snapshot_download('zaakirio/kokoro-ru', allow_patterns=['ru_g2p.py','kokoro-config.json','espeak-data/*','kokoro-ru-v2-base.pth','kokoro-ru-v2-dima.pth','voices/*.pt']); snapshot_download('ruaccent/accentuator')"
    "!KOKORO_PY!" -c "from kokoro import KModel; from ruaccent import RUAccent; print('Kokoro-RU environment is ready')"
    if errorlevel 1 echo WARNING: Kokoro-RU environment check failed - the mode will be unavailable.
) else echo WARNING: Python 3.11 was not found - skipping optional Kokoro-RU environment.
echo.

:optional_tts_done
goto :done

:warn_partial
call :build_exe
echo.
echo WARNING: some packages failed to install (see the error messages above).
echo The program will still start - whatever failed to install will just
echo be unavailable as a separate voice mode (e.g. if torch failed, local
echo Silero will not work; if pyttsx3 failed, offline mode will not work).
echo You can try running install.bat again.
goto :done

:done
echo.
echo ===============================================
echo   All done.
if exist "%~dp0fb2_reader.exe" echo   Run fb2_reader.exe ^(the single-file build^), or run_gui.bat ^(from sources^).
if not exist "%~dp0fb2_reader.exe" echo   Run run_gui.bat ^(the .exe was not built - see the messages above^).
echo ===============================================
echo.
pause
exit /b 0

rem ===============================================
rem   Single-file .exe build (PyInstaller, main.spec).
rem   Called after all Python dependencies and the grammar models are in
rem   place, because main.spec packs them (silero-stress, slovnet, tkinter).
rem   Result: dist\fb2_reader.exe, copied next to this file as
rem   fb2_reader.exe - there it finds (and seeds) the editable files
rem   (stress_dictionary.json, forced_stress_words.json, ...) and the
rem   models it must NOT carry inside (silero_model_v5_5_ru.pt,
rem   pretrained_models\). Failures here are not fatal.
rem ===============================================
:build_exe
if defined BUILD_DONE exit /b 0
set "BUILD_DONE=1"
echo.
echo ===============================================
echo   Building the single-file fb2_reader.exe ^(PyInstaller^)
echo ===============================================
"%VENV_PY%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo WARNING: PyInstaller is not installed - skipping the .exe build.
    exit /b 0
)
if not exist "%~dp0grammar_gate_models\slovnet_morph_news_v1.tar" echo WARNING: grammar models are missing - the .exe will be built WITHOUT the grammar gate.
"%VENV_PY%" -c "import silero_stress" >nul 2>nul
if errorlevel 1 echo WARNING: silero-stress is not installed - the .exe would have no automatic stress marks.
rem a running fb2_reader.exe locks the file and makes the build fail
taskkill /F /IM fb2_reader.exe >nul 2>nul
rem Keep PyInstaller's analysis cache between runs.  Deleting build\ and
rem dist\ here made every install rebuild the whole executable from scratch.
rem strictly ONE monolithic file (main.spec reads this variable; it is also its default)
set "AUDIOBOOK_ONEFILE=1"
"%VENV_PY%" -m PyInstaller main.spec --noconfirm --distpath "%~dp0dist" --workpath "%~dp0build"
if errorlevel 1 (
    echo WARNING: the .exe build failed - see the messages above. The program still runs from sources ^(run_gui.bat^).
    exit /b 0
)
if exist "%~dp0dist\fb2_reader.exe" (
    copy /Y "%~dp0dist\fb2_reader.exe" "%~dp0fb2_reader.exe" >nul
    echo Built: %~dp0fb2_reader.exe
    echo Check it with:  fb2_reader.exe selftest
)
exit /b 0

