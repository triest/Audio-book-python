@echo off
rem OPTIONAL, manual way to start the CosyVoice service. You normally do
rem NOT need this file at all - the program (run_gui.bat) starts and stops
rem this service by itself automatically as soon as you pick the
rem "CosyVoice" mode. Use this file only if you want to watch the service's
rem own console window directly, or start it ahead of time.
rem Requires install.bat to have been run first (it sets up CosyVoice too).
rem Leave this window open while using CosyVoice mode - closing it stops
rem the service. The very first request after starting can take a while
rem (the neural model is loaded into GPU memory at that point).
rem (This file's own messages are in English on purpose - see install.bat
rem for why. The program itself works in Russian as usual.)

cd /d "%~dp0"

if not exist ".venv_cosyvoice\Scripts\python.exe" (
    echo CosyVoice has not been set up yet on this computer.
    echo Run install.bat first ^(it now sets up CosyVoice too - one time,
    echo optional, large download^), then run this file again.
    pause
    exit /b 1
)

if not exist "CosyVoice\cosyvoice_rest_service.py" (
    echo cosyvoice_rest_service.py was not found inside the CosyVoice
    echo folder. Run install.bat again to fix this.
    pause
    exit /b 1
)

cd CosyVoice

echo Starting the CosyVoice service on http://localhost:5011 ...
echo ^(the first request will be slow - the model loads into the GPU then^)
echo Leave this window open. Press Ctrl+C or close this window to stop it.
echo.

"%~dp0.venv_cosyvoice\Scripts\python.exe" cosyvoice_rest_service.py
if errorlevel 1 pause
