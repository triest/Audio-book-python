@echo off
rem Experimentally builds the program into a single FB2AudiobookReader.exe
rem file (via PyInstaller), so it can run without Python installed at all -
rem just double-click the .exe.
rem
rem NOTE: this method has only been checked theoretically (the environment
rem this project was built in cannot run Windows programs directly) - if
rem the build fails on the first try, send the error text. The more
rem reliable "just click it" option is run_gui.bat, which installs
rem dependencies via pip into a regular Python and runs the program;
rem building the exe is worth trying only if, for some reason, you need a
rem single file with no Python on the computer at all.
rem
rem Because of torch, the resulting .exe will be large (roughly 400 MB to
rem 1+ GB) and the first build can take several minutes.
rem
rem Automatic startup of the Silero REST service (the "silero_rest" mode
rem with auto-start) will NOT work in the built .exe - use the local
rem "Silero" mode instead, it needs nothing extra.
rem (This file's own messages are in English on purpose - see install.bat
rem for why. The program itself works in Russian as usual.)

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Run install.bat (or run_gui.bat) at least once first -
    echo the .venv environment has not been created yet.
    pause
    exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo Installing PyInstaller...
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo Failed to install PyInstaller - see the message above.
    pause
    exit /b 1
)

echo.
echo Building FB2AudiobookReader.exe - please wait, this is not fast...
echo.

"%VENV_PY%" -m PyInstaller --noconfirm --onefile --windowed ^
    --name FB2AudiobookReader ^
    --hidden-import=silero ^
    --hidden-import=torch ^
    --hidden-import=torchaudio ^
    --hidden-import=omegaconf ^
    --hidden-import=pyttsx3 ^
    --hidden-import=gtts ^
    --hidden-import=requests ^
    --hidden-import=pydub ^
    --hidden-import=pygame ^
    fb2_reader_gui.py

if errorlevel 1 (
    echo.
    echo Build failed - see the error messages above.
    echo The most reliable way to run without building this is run_gui.bat.
    pause
    exit /b 1
)

echo.
echo Done: dist\FB2AudiobookReader.exe
echo You can copy this single file anywhere on the computer and run it by
echo double-clicking. On first run it will still download the Silero model
echo itself (~140 MB) into the folder it is run from - that is separate
echo from the .exe and happens once.
echo.
pause
