@echo off
rem Force-stops whatever process is listening on the CosyVoice3 port
rem (5012) - useful if the program was closed abnormally and left the
rem service running in the background, or start_cosyvoice3.bat's window
rem was closed without Ctrl+C.
setlocal enabledelayedexpansion
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":5012" ^| findstr "LISTENING"') do (
    echo Stopping process %%P (listening on port 5012)...
    taskkill /PID %%P /F
    set "FOUND=1"
)
if not defined FOUND echo Nothing is listening on port 5012 - nothing to stop.
pause
