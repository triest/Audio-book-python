@echo off
REM Запускает silero_rest_service.py и автоматически перезапускает его,
REM если процесс упадёт (например, из-за нативного сбоя в torch, который
REM Python не может поймать и залогировать). Обычный Ctrl+C всё равно
REM останавливает и эту обёртку, если нажать его дважды (или закрыть окно).
REM
REM Обычно отдельно запускать не нужно: если в GUI включена галочка
REM "Запускать сервис Silero REST автоматически", программа делает это
REM сама в фоне. Этот файл — для ручного/постоянно работающего сервиса.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Окружение .venv ещё не установлено — сначала запустите install.bat.
    pause
    exit /b 1
)

:loop
echo [%date% %time%] Запускаю silero_rest_service.py...
".venv\Scripts\python.exe" silero_rest_service.py
echo [%date% %time%] Сервис завершился (код %ERRORLEVEL%), перезапуск через 3 секунды...
timeout /t 3 /nobreak >nul
goto loop
