@echo off
rem Устанавливает Python-окружение и все зависимости этой программы в
rem папку .venv рядом с этим файлом. Ничего не трогает в системном Python
rem компьютера — все пакеты ставятся только в эту локальную папку.
rem Можно запускать повторно (например, после обновления файлов) — уже
rem установленное просто пропускается.

rem setlocal — чтобы переменные окружения (PYTHONUTF8 и т.п.), которые этот
rem файл ставит ниже, не "утекали" в run_gui.bat, когда он вызывает
rem install.bat через call (иначе они могли повлиять на последующий запуск
rem GUI из того же файла).
setlocal

rem Переключаем консоль на UTF-8, иначе русский текст (в том числе имена
rem глав книги) может отображаться "кракозябрами" — вопросиками или
rem нечитаемыми символами вместо кириллицы.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"

echo ===============================================
echo   FB2 Audiobook Reader — установка зависимостей
echo ===============================================
echo.

rem --- найти Python ---
set "PYLAUNCHER="
where py >nul 2>nul
if not errorlevel 1 set "PYLAUNCHER=py -3"
if not defined PYLAUNCHER (
    where python >nul 2>nul
    if not errorlevel 1 set "PYLAUNCHER=python"
)

if not defined PYLAUNCHER (
    echo Python не найден на этом компьютере.
    echo.
    echo Установите Python 3.10 или новее с сайта:
    echo   https://www.python.org/downloads/
    echo При установке обязательно поставьте галочку внизу первого экрана:
    echo   "Add python.exe to PATH"
    echo Затем запустите этот файл ещё раз.
    echo.
    pause
    exit /b 1
)

echo Найден Python:
%PYLAUNCHER% --version
echo.

rem --- создать виртуальное окружение, если его ещё нет ---
if not exist ".venv\Scripts\python.exe" (
    echo Создаю окружение .venv ...
    %PYLAUNCHER% -m venv .venv
    if errorlevel 1 (
        echo Не удалось создать виртуальное окружение — см. сообщение выше.
        pause
        exit /b 1
    )
    echo Готово.
    echo.
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo Устанавливаю зависимости — при первом запуске это может занять
echo несколько минут (особенно torch, обычно 200-700 МБ). Идёт закачка...
echo.

"%VENV_PY%" -m pip install --upgrade pip --quiet

rem Базовые зависимости — нужны почти всегда (чтение fb2, GUI, скачивание)
"%VENV_PY%" -m pip install lxml numpy tqdm requests
if errorlevel 1 goto :warn_partial

rem Локальный синтез Silero (основной, рекомендуемый режим)
"%VENV_PY%" -m pip install silero torch torchaudio omegaconf
if errorlevel 1 goto :warn_partial

rem Silero через отдельный REST-сервис (SSML-паузы, необязательно)
"%VENV_PY%" -m pip install fastapi uvicorn ruaccent num2words
if errorlevel 1 goto :warn_partial

rem Некоторые из пакетов выше могут по пути поставить старую версию click
rem (используется сторонними зависимостями типа huggingface-hub) — pip иногда
rem пишет про это предупреждение "dependency resolver does not... conflicts".
rem Само по себе это НЕ ошибка (пакеты всё равно ставятся и программа
rem работает) — но подтягиваем совместимую версию явно, чтобы предупреждение
rem не появлялось и на всякий случай.
"%VENV_PY%" -m pip install "click>=8.4.2" --upgrade --quiet

rem Google TTS и системный TTS (необязательные режимы)
"%VENV_PY%" -m pip install gTTS pyttsx3 pydub
if errorlevel 1 goto :warn_partial

echo.
echo Все зависимости установлены успешно.
echo (Если выше видно предупреждение вида "dependency resolver does not
echo currently take into account..." про click или другой пакет — это не
echo ошибка, а информационное предупреждение pip; раз выше написано
echo "Successfully installed", всё встало нормально и программа будет
echo работать.)
goto :done

:warn_partial
echo.
echo ВНИМАНИЕ: часть пакетов не установилась (см. сообщения об ошибках выше).
echo Программа всё равно запустится — то, что не установилось, будет
echo недоступно как отдельный режим озвучки (например, если не встал torch,
echo не будет работать локальный Silero; если не встал pyttsx3 — offline-режим).
echo Можно попробовать запустить install.bat ещё раз.

:done
echo.
pause
