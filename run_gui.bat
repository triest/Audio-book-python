@echo off
rem One click to run: on first run it installs everything needed itself
rem (via install.bat, in a .venv folder next to this file), after that it
rem just opens the program.
rem (This file's own messages are in English on purpose - see
rem install.bat for why. The program itself works in Russian as usual.)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call install.bat
    if not exist ".venv\Scripts\python.exe" (
        echo Setup did not finish - the .venv environment was not created.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" fb2_reader_gui.py %*
if errorlevel 1 pause
