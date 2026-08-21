@echo off
rem Экспериментально собирает программу в один файл FB2AudiobookReader.exe
rem (через PyInstaller), чтобы её можно было запускать без установленного
rem Python вообще — просто по клику на .exe.
rem
rem ВАЖНО: этот способ проверялся только теоретически (среда, где
rem собирался этот проект, не может запускать Windows-программы напрямую) —
rem если сборка не пройдёт с первого раза, пришлите текст ошибки. Более
rem надёжный вариант "просто по клику" — run_gui.bat, он ставит зависимости
rem через pip в обычный Python и запускает программу; экспериментировать с
rem exe стоит, только если по каким-то причинам нужен именно один файл без
rem Python на компьютере вообще.
rem
rem Из-за torch итоговый .exe будет большим (по грубой оценке от 400 МБ до
rem 1+ ГБ) и первая сборка может занять несколько минут.
rem
rem В собранном .exe НЕ будет работать автозапуск сервиса Silero REST
rem (режим "silero_rest" с автостартом) — используйте локальный режим
rem "Silero", он ничего дополнительного не требует.

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите install.bat (или run_gui.bat) хотя бы один раз —
    echo окружение .venv ещё не создано.
    pause
    exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo Устанавливаю PyInstaller...
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo Не удалось установить PyInstaller — см. сообщение выше.
    pause
    exit /b 1
)

echo.
echo Собираю FB2AudiobookReader.exe — подождите, это не быстро...
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
    fb2_reader_gui.py

if errorlevel 1 (
    echo.
    echo Сборка не удалась — см. сообщения об ошибках выше.
    echo Самый надёжный способ запуска без такой сборки — run_gui.bat.
    pause
    exit /b 1
)

echo.
echo Готово: dist\FB2AudiobookReader.exe
echo Можно скопировать этот один файл куда угодно на компьютере и запускать
echo двойным кликом. При первом запуске он всё равно скачает саму модель
echo Silero (~140 МБ) в ту папку, откуда его запустят, — это отдельно от
echo самого .exe и происходит один раз.
echo.
pause
