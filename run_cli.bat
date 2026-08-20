@echo off
REM Озвучка из командной строки. Можно перетащить .fb2 на этот файл.
cd /d "%~dp0"
if "%~1"=="" (
  echo Использование: run_cli.bat book.fb2 [доп. аргументы fb2_reader.py]
  echo Пример: run_cli.bat book.fb2 --mode silero --speaker xenia --outdir audiobook
  pause
  exit /b 1
)
python fb2_reader.py %*
if errorlevel 1 pause
