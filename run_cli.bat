@echo off
REM Озвучка из командной строки. Можно перетащить .fb2 на этот файл.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%~1"=="" (
  echo Использование: run_cli.bat book.fb2 [доп. аргументы fb2_reader.py]
  echo Пример: run_cli.bat book.fb2 --mode silero --speaker xenia --outdir audiobook
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Окружение .venv ещё не установлено — сначала запустите install.bat
    echo (или run_gui.bat, он ставит зависимости сам при первом запуске).
    pause
    exit /b 1
)

".venv\Scripts\python.exe" fb2_reader.py %*
if errorlevel 1 pause
