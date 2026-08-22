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

cd /d "%~dp0"

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

echo Installing dependencies - on first run this can take a few minutes
echo (torch especially, usually 200-700 MB). Downloading...
echo.

"%VENV_PY%" -m pip install --upgrade pip --quiet

rem Base dependencies - needed almost always (fb2 parsing, GUI, downloads)
"%VENV_PY%" -m pip install lxml numpy tqdm requests
if errorlevel 1 goto :warn_partial

rem Local Silero synthesis (the main, recommended mode)
"%VENV_PY%" -m pip install silero torch torchaudio omegaconf
if errorlevel 1 goto :warn_partial

rem Silero via a separate REST service (SSML pauses, optional)
"%VENV_PY%" -m pip install fastapi uvicorn ruaccent num2words
if errorlevel 1 goto :warn_partial

rem Some of the packages above can pull in an older version of click
rem (used by third-party dependencies like huggingface-hub) - pip
rem sometimes prints a warning about this: "dependency resolver does
rem not... conflicts". That by itself is NOT an error (everything still
rem installs and the program works) - but we pin a compatible version
rem explicitly so the warning stops appearing, just in case.
"%VENV_PY%" -m pip install "click>=8.4.2" --upgrade --quiet

rem Google TTS and system TTS (optional modes)
"%VENV_PY%" -m pip install gTTS pyttsx3 pydub
if errorlevel 1 goto :warn_partial

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

"%CV_PY%" -m pip install --upgrade pip --quiet

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
"%CV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo WARNING: CUDA ^(cu124^) build of torch failed - trying cu121 index...
    "%CV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo WARNING: CUDA builds of torch failed - trying the plain
        echo ^(CPU-only^) build instead...
        "%CV_PY%" -m pip install torch torchaudio
        if errorlevel 1 echo WARNING: torch/torchaudio failed to install.
    )
)

if "!CV_PINNED!"=="1" (
    "%CV_PY%" -m pip install numpy==1.26.4
    if errorlevel 1 (
        "%CV_PY%" -m pip install numpy
        if errorlevel 1 echo WARNING: numpy failed to install.
    )
) else (
    "%CV_PY%" -m pip install numpy
    if errorlevel 1 echo WARNING: numpy failed to install.
)

if "!CV_PINNED!"=="1" (
    "%CV_PY%" -m pip install onnxruntime-gpu==1.18.0
    if errorlevel 1 (
        echo WARNING: onnxruntime-gpu failed - trying CPU-only onnxruntime...
        "%CV_PY%" -m pip install onnxruntime
        if errorlevel 1 echo WARNING: onnxruntime failed to install.
    )
) else (
    "%CV_PY%" -m pip install onnxruntime
    if errorlevel 1 echo WARNING: onnxruntime failed to install.
)

"%CV_PY%" -m pip install openai-whisper==20231117
if errorlevel 1 (
    "%CV_PY%" -m pip install openai-whisper
    if errorlevel 1 echo WARNING: openai-whisper failed to install.
)
"%CV_PY%" -m pip install transformers==4.51.3
if errorlevel 1 (
    "%CV_PY%" -m pip install transformers
    if errorlevel 1 echo WARNING: transformers failed to install.
)
"%CV_PY%" -m pip install diffusers==0.29.0
if errorlevel 1 (
    "%CV_PY%" -m pip install diffusers
    if errorlevel 1 echo WARNING: diffusers failed to install.
)
"%CV_PY%" -m pip install funasr
if errorlevel 1 echo WARNING: funasr failed to install.
"%CV_PY%" -m pip install modelscope
if errorlevel 1 echo WARNING: modelscope failed to install.
"%CV_PY%" -m pip install huggingface_hub
if errorlevel 1 echo WARNING: huggingface_hub failed to install.
"%CV_PY%" -m pip install hyperpyyaml
if errorlevel 1 echo WARNING: hyperpyyaml failed to install.
"%CV_PY%" -m pip install conformer
if errorlevel 1 echo WARNING: conformer failed to install.
"%CV_PY%" -m pip install gdown
if errorlevel 1 echo WARNING: gdown failed to install.
"%CV_PY%" -m pip install inflect
if errorlevel 1 echo WARNING: inflect failed to install.
"%CV_PY%" -m pip install wetext
if errorlevel 1 echo WARNING: wetext failed to install.

rem The "Matcha-TTS" submodule that CosyVoice vendors under third_party\ (see
rem the "import matcha..." lines inside cosyvoice\flow\flow_matching.py) has
rem its own extra dependencies that are not otherwise pulled in by anything
rem above - without these, model loading fails with errors like
rem "ModuleNotFoundError: No module named 'lightning'" the first time you
rem actually try to synthesize something (the service still starts up fine,
rem it just fails to load the model at that later point).
"%CV_PY%" -m pip install lightning==2.2.4
if errorlevel 1 (
    "%CV_PY%" -m pip install lightning
    if errorlevel 1 echo WARNING: lightning failed to install.
)
"%CV_PY%" -m pip install hydra-core==1.3.2
if errorlevel 1 (
    "%CV_PY%" -m pip install hydra-core
    if errorlevel 1 echo WARNING: hydra-core failed to install.
)
"%CV_PY%" -m pip install omegaconf==2.3.0
if errorlevel 1 (
    "%CV_PY%" -m pip install omegaconf
    if errorlevel 1 echo WARNING: omegaconf failed to install.
)
"%CV_PY%" -m pip install rich==13.7.1
if errorlevel 1 (
    "%CV_PY%" -m pip install rich
    if errorlevel 1 echo WARNING: rich failed to install.
)
"%CV_PY%" -m pip install rootutils
if errorlevel 1 echo WARNING: rootutils failed to install.
"%CV_PY%" -m pip install einops
if errorlevel 1 echo WARNING: einops failed to install.
"%CV_PY%" -m pip install matplotlib==3.7.5
if errorlevel 1 (
    "%CV_PY%" -m pip install matplotlib
    if errorlevel 1 echo WARNING: matplotlib failed to install.
)
"%CV_PY%" -m pip install networkx==3.1
if errorlevel 1 (
    "%CV_PY%" -m pip install networkx
    if errorlevel 1 echo WARNING: networkx failed to install.
)
"%CV_PY%" -m pip install onnx==1.16.0
if errorlevel 1 (
    "%CV_PY%" -m pip install onnx
    if errorlevel 1 echo WARNING: onnx failed to install.
)
"%CV_PY%" -m pip install pyworld==0.3.4
if errorlevel 1 (
    "%CV_PY%" -m pip install pyworld
    if errorlevel 1 echo WARNING: pyworld failed to install.
)
"%CV_PY%" -m pip install x-transformers==2.11.24
if errorlevel 1 (
    "%CV_PY%" -m pip install x-transformers
    if errorlevel 1 echo WARNING: x-transformers failed to install.
)
"%CV_PY%" -m pip install pyarrow==18.1.0
if errorlevel 1 (
    "%CV_PY%" -m pip install pyarrow
    if errorlevel 1 echo WARNING: pyarrow failed to install.
)
"%CV_PY%" -m pip install grpcio==1.57.0
if errorlevel 1 (
    "%CV_PY%" -m pip install grpcio
    if errorlevel 1 echo WARNING: grpcio failed to install.
)
"%CV_PY%" -m pip install grpcio-tools==1.57.0
if errorlevel 1 (
    "%CV_PY%" -m pip install grpcio-tools
    if errorlevel 1 echo WARNING: grpcio-tools failed to install.
)
"%CV_PY%" -m pip install wget
if errorlevel 1 echo WARNING: wget failed to install.

if exist "CosyVoice\requirements.txt" (
    "%CV_PY%" -m pip install --no-deps -r CosyVoice\requirements.txt
)

if exist "cosyvoice_rest_service.py" (
    copy /Y "cosyvoice_rest_service.py" "CosyVoice\cosyvoice_rest_service.py" >nul
)

rem NOTE: the voice cloning engine used by cosyvoice_rest_service.py was
rem switched from CosyVoice2 first to XTTS-v2, then to F5-TTS with a
rem Russian-specific finetuned checkpoint (F5-TTS-Russian) - CosyVoice2 is
rem not trained on Russian at all and produced unintelligible output no
rem matter how it was called; XTTS-v2 is officially multilingual (incl.
rem Russian) but not specialized for it, so pronunciation/stress was often
rem off; F5-TTS-Russian was actually finetuned on real Russian speech
rem (Common Voice, SOVA RUDevices, SberDevices Golos), so it should sound
rem noticeably more natural. The service tries F5-TTS first and falls back
rem to XTTS automatically if F5-TTS fails to load - both are installed
rem below so that fallback actually works. The CosyVoice2 model download
rem step that used to be here was removed - no longer used.
echo.
echo Installing f5-tts ^(F5-TTS-Russian voice cloning engine - primary^)...
"%CV_PY%" -m pip install f5-tts huggingface_hub
if errorlevel 1 (
    echo WARNING: f5-tts failed to install - this can happen because one of
    echo its dependencies ^(bitsandbytes^) does not always have a ready-made
    echo Windows wheel. The service will fall back to the XTTS-v2 engine
    echo below if this happens, so this is not fatal - just means the
    echo Russian-specialized voice will not be available until this is
    echo fixed. You can try running install.bat again, or report the exact
    echo error if it keeps failing.
) else (
    echo.
    echo Pre-downloading the F5-TTS-Russian checkpoint ^(~1.3 GB, only needs
    echo to happen once - if this step is skipped or fails, the service will
    echo just download it the first time voice cloning is used instead^)...
    "%CV_PY%" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hotstone228/F5-TTS-Russian', filename='model_last.safetensors', local_dir='pretrained_models/f5-tts-russian'); hf_hub_download(repo_id='hotstone228/F5-TTS-Russian', filename='vocab.txt', local_dir='pretrained_models/f5-tts-russian')"
    if errorlevel 1 (
        echo WARNING: could not pre-download the F5-TTS-Russian checkpoint -
        echo it will be downloaded automatically the first time you use voice
        echo cloning instead ^(just makes the very first use slower^).
    )
)

echo.
echo Installing coqui-tts ^(XTTS-v2 voice cloning engine - fallback, used if
echo F5-TTS above failed to install or load^)...
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

echo.
echo CosyVoice setup finished ^(see any WARNING lines above for details^).
echo If everything above succeeded, the "CosyVoice" voice mode will be
echo available in the program, and its background service will start
echo automatically when you select that mode - no separate window needed.

:cosyvoice_done
echo.
goto :done

:warn_partial
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
echo   All done. You can now run run_gui.bat.
echo ===============================================
echo.
pause
