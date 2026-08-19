@echo off
REM Запускает silero_rest_service.py и автоматически перезапускает его,
REM если процесс упадёт (например, из-за нативного сбоя в torch, который
REM Python не может поймать и залогировать). Обычный Ctrl+C всё равно
REM останавливает и эту обёртку, если нажать его дважды (или закрыть окно).
cd /d "%~dp0"
:loop
echo [%date% %time%] Запускаю silero_rest_service.py...
python3 silero_rest_service.py
echo [%date% %time%] Сервис завершился (код %ERRORLEVEL%), перезапуск через 3 секунды...
timeout /t 3 /nobreak >nul
goto loop
