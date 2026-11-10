@echo off
rem One click to run: on first run it installs everything needed itself
rem (via install.bat, in a .venv folder next to this file), after that it
rem just opens the program.
rem (This file's own messages are in English on purpose - see
rem install.bat for why. The program itself works in Russian as usual.)
cd /d "%~dp0"

rem Python 3.14 on this machine has Tcl/Tk installed, but tkinter does not
rem discover the libraries from a venv reliably. Set the official runtime
rem paths explicitly when they are available.
if exist "%~dp0tcl_runtime\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=%~dp0tcl_runtime\tcl8.6"
    set "TK_LIBRARY=%~dp0tcl_runtime\tk8.6"
) else if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\tcl\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\tcl\tcl8.6"
    set "TK_LIBRARY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\tcl\tk8.6"
)

rem Grammar gate (case/number stress homographs) is ON by default; to switch it off for a test:
rem set AUDIOBOOK_GRAMMAR_GATE=0

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
