@echo off
rem Runs silero_rest_service.py and automatically restarts it if the
rem process crashes (e.g. a native failure inside torch that Python
rem cannot catch and log). A normal Ctrl+C still stops this wrapper too
rem if pressed twice (or the window is closed).
rem
rem Usually you do not need to run this separately: if the GUI's
rem "Start Silero REST service automatically" checkbox is on, the
rem program does this itself in the background. This file is for
rem running the service manually / continuously.
rem (This file's own messages are in English on purpose - see
rem install.bat for why. The program itself works in Russian as usual.)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The .venv environment is not installed yet - run install.bat first.
    pause
    exit /b 1
)

:loop
echo [%date% %time%] Starting silero_rest_service.py...
".venv\Scripts\python.exe" silero_rest_service.py
echo [%date% %time%] Service exited (code %ERRORLEVEL%), restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto loop
