@echo off
cd /d "%~dp0"
python fb2_reader.py --gui %*
if errorlevel 1 pause
