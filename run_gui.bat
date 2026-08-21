@echo off
rem Один файл на клик: первый запуск сам ставит всё нужное (через
rem install.bat, окружение .venv рядом), дальше просто открывает программу.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call install.bat
    if not exist ".venv\Scripts\python.exe" (
        echo Установка не завершилась — окружение .venv не создано.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" fb2_reader_gui.py %*
if errorlevel 1 pause
