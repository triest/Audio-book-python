@echo off
rem Command-line voicing. You can drag & drop a .fb2 file onto this file.
rem (This file's own messages are in English on purpose - see install.bat
rem for why. The program itself works in Russian as usual.)
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_cli.bat book.fb2 [extra fb2_reader.py arguments]
  echo Example: run_cli.bat book.fb2 --mode silero --speaker xenia --outdir audiobook
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo The .venv environment is not installed yet - run install.bat first
    echo (or run_gui.bat, which installs dependencies itself on first run).
    pause
    exit /b 1
)

".venv\Scripts\python.exe" fb2_reader.py %*
if errorlevel 1 pause
