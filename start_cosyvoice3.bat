@echo off
rem Manually starts the CosyVoice3 service in a visible console window (for
rem troubleshooting - the program itself starts/stops this automatically
rem when you select the "CosyVoice 3" engine, no separate window). Useful
rem to see errors directly instead of digging through the log file, or to
rem pre-warm the model weights download (~1-2 GB, first run only) before
rem using the program.
rem Requires install.bat to have set up .venv_cosyvoice3 and CosyVoice3\
rem first (see the "CosyVoice3 (experimental)" section of install.bat).

set "CV3_PY=%~dp0.venv_cosyvoice3\Scripts\python.exe"
if not exist "%CV3_PY%" (
    echo .venv_cosyvoice3 not found - run install.bat first.
    pause
    exit /b 1
)
if not exist "%~dp0CosyVoice3\cosyvoice3_rest_service.py" (
    echo CosyVoice3\cosyvoice3_rest_service.py not found - run install.bat first.
    pause
    exit /b 1
)

cd /d "%~dp0CosyVoice3"
"%CV3_PY%" cosyvoice3_rest_service.py
pause
