import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from fb2_reader import COSYVOICE_DEFAULT_REST_URL
from gui_shared import (
    _app_dir,
    _log_exception_to_file,
    COSYVOICE_DEFAULT_ENGINE,
    COSYVOICE_ENGINE_CODES,
    COSYVOICE_EXPECTED_SERVICE_VERSION,
    COSYVOICE3_DEFAULT_REST_URL,
    COSYVOICE3_EXPECTED_SERVICE_VERSION,
)


class ServiceManagementMixin:
    @staticmethod
    def _rest_url_is_local(url: str) -> bool:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1", "")
    @staticmethod
    def _ping_rest_service(url: str, timeout: float = 1.5) -> bool:
        """Проверяет, отвечает ли уже сервис Silero REST по этому адресу.
        Любой ответ сервера (даже 404 на несуществующий путь) значит, что
        сервис поднят — ошибка нас интересует только на уровне соединения
        (сервис не запущен / порт никто не слушает)."""
        try:
            urllib.request.urlopen(url.rstrip("/") + "/docs", timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False
    @staticmethod
    def _ping_http_ok(url: str, timeout: float = 1.5) -> bool:
        """Как _ping_rest_service, но без привязки к /docs (FastAPI) — тут
        просто проверяем, отвечает ли вообще что-то по корневому адресу.
        Используется для локального Gradio-сервера Qwen3-TTS, у которого
        по корню отдаётся веб-страница интерфейса, а не /docs."""
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def _ensure_qwen_local_service_running(self, server_url: str, start_cmd: str,
                                            ready_timeout: float = 180.0) -> bool:
        """Аналог _ensure_rest_service_running/_ensure_cosyvoice_service_running,
        но для произвольного стороннего сервера (Gradio-демо Qwen3-TTS,
        обычно поставленного через Pinokio) — вместо фиксированного пути к
        своему скрипту здесь запускается команда, которую один раз указал
        сам пользователь (путь к .bat/.exe). Если сервер уже отвечает —
        ничего не запускаем и помечаем self._qwen_local_started_by_us =
        False, чтобы потом (после озвучки) НЕ гасить чужой/ранее
        запущенный вручную процесс — глушим только то, что подняли сами."""
        self._qwen_local_started_by_us = False
        if self._ping_http_ok(server_url):
            self.after(0, lambda: self.log(
                f"Локальный сервер Qwen3-TTS уже отвечает на {server_url} — использую его."
            ))
            return True

        if not start_cmd:
            self.after(0, lambda: self.log(
                f"Локальный сервер Qwen3-TTS не отвечает на {server_url}, а команда "
                "автозапуска не указана — запустите сервер вручную (см. блок "
                "«Qwen3-TTS» справа) и повторите озвучку."
            ))
            return False

        if self._qwen_local_proc is None or self._qwen_local_proc.poll() is not None:
            try:
                popen_kwargs = dict(
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._qwen_local_proc = subprocess.Popen(start_cmd, shell=True, **popen_kwargs)
            except Exception as e:
                err = str(e)
                self.after(0, lambda t=err: self.log(
                    f"Не удалось запустить команду автозапуска Qwen3-TTS: {t}"
                ))
                return False
            self._qwen_local_started_by_us = True

            def pump_output():
                proc = self._qwen_local_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [Qwen3-TTS локально] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        self.after(0, lambda: self.log(
            "…запускаю локальный сервер Qwen3-TTS (при первом запуске может загружать "
            "модель, это может занять несколько минут)…"
        ))
        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            if self._ping_http_ok(server_url):
                self.after(0, lambda: self.log("Локальный сервер Qwen3-TTS запущен и готов."))
                return True
            if self._qwen_local_proc is not None and self._qwen_local_proc.poll() is not None:
                self.after(0, lambda: self.log(
                    f"Команда автозапуска Qwen3-TTS завершилась сама собой "
                    f"(код {self._qwen_local_proc.returncode}) во время запуска — "
                    "см. сообщения [Qwen3-TTS локально] выше."
                ))
                return False
            if time.time() - last_notice > 15:
                last_notice = time.time()
                self.after(0, lambda: self.log("…сервер Qwen3-TTS ещё запускается…"))
            time.sleep(1.0)

        self.after(0, lambda: self.log("Локальный сервер Qwen3-TTS не успел запуститься за отведённое время."))
        return False

    def _stop_qwen_local_service(self):
        """Гасит сервер, только если его запустила именно эта программа
        (см. self._qwen_local_started_by_us в _ensure_qwen_local_service_running) —
        сервер, который пользователь уже держал запущенным сам, не трогаем."""
        if not self._qwen_local_started_by_us:
            return
        if self._qwen_local_proc is not None and self._qwen_local_proc.poll() is None:
            try:
                self._qwen_local_proc.terminate()
                self.after(0, lambda: self.log("Локальный сервер Qwen3-TTS остановлен."))
            except Exception:
                pass
        self._qwen_local_proc = None
        self._qwen_local_started_by_us = False

    @staticmethod
    def _service_python() -> str:
        """Interpreter for the service: the project's own .venv (where install.bat put fastapi,
        silero...) when it exists, otherwise the interpreter running the GUI."""
        venv_py = _app_dir() / ".venv" / "Scripts" / "python.exe"
        return str(venv_py) if venv_py.exists() else sys.executable

    def _ensure_rest_service_running(self, rest_url: str, ready_timeout: float = 600.0) -> bool:
        """Thread-safe wrapper: the startup recycle (see _startup_recycle_services) and a
        "Start" click must never spawn two Silero REST processes at once."""
        with self._rest_svc_lock:
            return self._ensure_rest_service_running_impl(rest_url, ready_timeout)

    def _ensure_rest_service_running_impl(self, rest_url: str, ready_timeout: float = 600.0) -> bool:
        """Запускает silero_rest_service.py в фоне (без консоли пользователя)
        и ждёт, пока он не начнёт отвечать на rest_url. При первом запуске
        сервис может несколько минут скачивать модель — поэтому таймаут
        большой, а в журнал периодически пишется, что процесс ещё идёт."""
        if self._rest_proc is None or self._rest_proc.poll() is not None:
            service_path = _app_dir() / "silero_rest_service.py"
            frozen = bool(getattr(sys, "frozen", False))
            if not frozen and not service_path.exists():
                self.after(0, lambda: self.log(f"Не найден файл сервиса: {service_path}"))
                return False
            try:
                popen_kwargs = dict(
                    cwd=str(_app_dir()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if sys.platform == "win32":
                    # без отдельного окна консоли для сервиса
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._rest_proc = subprocess.Popen(
                    ([sys.executable, "serve-silero"] if frozen
                     else [self._service_python(), str(service_path)]), **popen_kwargs
                )
            except Exception as e:
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис Silero REST: {t}"))
                return False

            def pump_output():
                proc = self._rest_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис Silero REST] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            if self._ping_rest_service(rest_url):
                self.after(0, lambda: self.log("Сервис Silero REST запущен и готов принимать запросы."))
                return True
            if self._rest_proc is not None and self._rest_proc.poll() is not None:
                self.after(0, lambda: self.log(
                    f"Сервис Silero REST завершился сам собой (код {self._rest_proc.returncode}) "
                    "во время запуска — см. сообщения [сервис Silero REST] выше."
                ))
                return False
            if time.time() - last_notice > 15:
                last_notice = time.time()
                self.after(0, lambda: self.log(
                    "…сервис Silero REST ещё запускается (при первом запуске может скачивать "
                    "модель, это может занять несколько минут)…"
                ))
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис Silero REST не успел запуститься за отведённое время."))
        return False
    @staticmethod
    def _cosyvoice_model_loaded(url: str, timeout: float = 3.0, expected_version: str = None) -> bool:
        """В отличие от _ping_rest_service (который считает сервис рабочим
        по любому HTTP-ответу), здесь мы проверяем именно то, что модель
        CosyVoice реально загрузилась И что запущенный сервис - это
        актуальная версия кода (через отдельный эндпоинт /health).

        Без первой проверки зависший с прошлого раза процесс с ошибкой при
        загрузке модели выглядел бы для программы как "уже запущен и всё
        хорошо", и она бы просто продолжала слать в него запросы, каждый
        раз получая HTTP 500 без единой новой попытки перезапуска.

        Без второй проверки (версия) уже РАБОТАЮЩИЙ (с успешно загруженной
        моделью) сервис, оставшийся с прошлого запуска программы, считался
        бы полностью готовым даже после того, как сам файл
        cosyvoice_rest_service.py обновился на диске — из-за чего новые
        эндпоинты и исправления молча не действовали бы, пока пользователь
        не перезапустит компьютер вручную (именно так один раз сломалось
        удаление голосов: старый процесс просто не знал про /voices/{name}).

        expected_version - какую версию сервиса считать актуальной; по
        умолчанию COSYVOICE_EXPECTED_SERVICE_VERSION (обычный сервис в
        .venv_cosyvoice), но для CosyVoice3 (см. _ensure_cosyvoice3_service_running)
        передаётся COSYVOICE3_EXPECTED_SERVICE_VERSION - у него отдельный
        файл сервиса и своя нумерация версий."""
        if expected_version is None:
            expected_version = COSYVOICE_EXPECTED_SERVICE_VERSION
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("model_loaded"):
                    return False
                # Сверяем строго, включая случай "поля вообще нет" (значит,
                # это ещё более старая версия сервиса, до появления самой
                # этой проверки, - тем более пора перезапустить).
                if data.get("service_version") != expected_version:
                    return False
                return True
        except Exception:
            return False
    @staticmethod
    def _pids_listening_on_port(port: int) -> set:
        no_window = subprocess.CREATE_NO_WINDOW
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, creationflags=no_window,
        )
        pids = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == "TCP":
                local_addr = parts[1]
                if local_addr.endswith(f":{port}") and "LISTENING" in line.upper():
                    pid = parts[-1]
                    if pid.isdigit() and pid != "0":
                        pids.add(pid)
        return pids
    @classmethod
    def _kill_process_on_port(cls, port: int, wait_free_timeout: float = 8.0):
        """Убивает процесс(ы), слушающие данный TCP-порт на localhost —
        используется, чтобы наверняка избавиться от зависшего с прошлого
        раза сервиса CosyVoice (например, оставшегося после того, как
        программу закрыли не тем способом, или процесс пережил закрытие
        родителя). Работает через стандартные системные утилиты Windows
        (netstat/taskkill), без сторонних зависимостей.

        После taskkill Windows не всегда освобождает порт мгновенно —
        поэтому дальше ждём (опрашивая netstat), пока порт реально не
        освободится, вместо того чтобы полагаться на фиксированную паузу
        (слишком короткая пауза раньше приводила к тому, что новый процесс
        сразу падал с "address already in use", т.к. старый ещё не успел
        полностью закрыть сокет)."""
        if sys.platform != "win32":
            return
        try:
            pids = cls._pids_listening_on_port(port)
            no_window = subprocess.CREATE_NO_WINDOW
            for pid in pids:
                subprocess.run(
                    ["taskkill", "/PID", pid, "/F"],
                    creationflags=no_window, capture_output=True,
                )
            if not pids:
                return
            deadline = time.time() + wait_free_timeout
            while time.time() < deadline:
                if not cls._pids_listening_on_port(port):
                    return
                time.sleep(0.5)
        except Exception:
            _log_exception_to_file(f"попытка остановить процесс на порту {port}")
    def _cosyvoice_engine_code(self) -> str:
        """Внутренний код выбранного движка ("f5"/"xtts") по подписи,
        выбранной в выпадающем списке — уходит в переменную окружения
        TTS_ENGINE процесса сервиса."""
        return COSYVOICE_ENGINE_CODES.get(self.cosyvoice_engine_var.get(), COSYVOICE_DEFAULT_ENGINE)
    def _on_cosyvoice_engine_changed(self, event=None):
        """При смене движка в выпадающем списке уже запущенный сервис
        обслуживает запросы старым движком, пока его не перезапустить -
        останавливаем его здесь же, чтобы следующий запрос сам поднял
        сервис заново с новым TTS_ENGINE (см. _ensure_cosyvoice_service_running
        и сверку engine в _cosyvoice_model_loaded).

        CosyVoice3 - отдельный случай: у него свой подпроцесс/окружение и
        свой порт 5012 (см. _ensure_cosyvoice3_service_running), не
        связанный с движками f5/espeech/f5winter/xtts на порту 5011 -
        здесь просто переключаем REST URL на его порт, чтобы не пришлось
        делать это вручную (сам процесс на 5011, если запущен, трогать не
        нужно - при выборе cosyvoice3 к нему никто и не обращается). При
        переключении ОБРАТНО с cosyvoice3 на любой другой движок -
        возвращаем URL по умолчанию, если пользователь не менял его сам
        вручную на что-то своё."""
        engine = self._cosyvoice_engine_code()
        current_url = self.cosyvoice_rest_url_var.get().strip()
        if engine == "cosyvoice3":
            if current_url in ("", COSYVOICE_DEFAULT_REST_URL):
                self.cosyvoice_rest_url_var.set(COSYVOICE3_DEFAULT_REST_URL)
            return
        if current_url == COSYVOICE3_DEFAULT_REST_URL:
            self.cosyvoice_rest_url_var.set(COSYVOICE_DEFAULT_REST_URL)

        if self._cosyvoice_proc is not None and self._cosyvoice_proc.poll() is None:
            self.log(f"Движок CosyVoice изменён на «{self.cosyvoice_engine_var.get()}» — "
                      "перезапускаю сервис…")
            try:
                rest_url = self.cosyvoice_rest_url_var.get().strip() or COSYVOICE_DEFAULT_REST_URL
                from urllib.parse import urlparse
                port = urlparse(rest_url).port or 5011
            except Exception:
                port = 5011
            self._cosyvoice_proc = None
            threading.Thread(target=lambda: self._kill_process_on_port(port), daemon=True).start()
    def _ensure_cosyvoice3_service_running(self, rest_url: str, ready_timeout: float = 2700.0) -> bool:
        """Аналог _ensure_cosyvoice_service_running, но для CosyVoice3 -
        собственный подпроцесс в СВОЁМ окружении .venv_cosyvoice3 (у
        CosyVoice3 несовместимый со всем остальным набор фиксированных
        версий зависимостей - torch==2.3.1 и т.д., см. install.bat,
        секция CosyVoice3), и свой порт (COSYVOICE3_DEFAULT_REST_URL).

        (Изначально это планировалось запускать в Docker-контейнере (см.
        docker/cosyvoice3/) для полной изоляции - но Docker Desktop/WSL2 у
        пользователя оказался слишком капризным, поэтому вместо контейнера -
        обычный подпроцесс, как у остальных движков.)"""
        if getattr(sys, "frozen", False):
            self.after(0, lambda: self.log(
                "Автозапуск сервиса CosyVoice3 недоступен в собранной .exe-версии "
                "программы. Запустите программу из исходников (run_gui.bat)."
            ))
            return False

        cosyvoice3_py = _app_dir() / ".venv_cosyvoice3" / "Scripts" / "python.exe"
        if not cosyvoice3_py.exists():
            alt = _app_dir() / ".venv_cosyvoice3" / "bin" / "python"
            cosyvoice3_py = alt if alt.exists() else cosyvoice3_py
        service_dir = _app_dir() / "CosyVoice3"
        service_path = service_dir / "cosyvoice3_rest_service.py"

        if not cosyvoice3_py.exists() or not service_path.exists():
            self.after(0, lambda: self.log(
                "Движок CosyVoice3 ещё не установлен на этом компьютере. Запустите "
                "install.bat ещё раз (секция CosyVoice3 — отдельная, большая загрузка, "
                "может занять время)."
            ))
            return False

        try:
            from urllib.parse import urlparse
            port = urlparse(rest_url).port or 5012
        except Exception:
            port = 5012

        need_start = self._cosyvoice3_proc is None or self._cosyvoice3_proc.poll() is not None
        if need_start and self._ping_rest_service(rest_url):
            # self._cosyvoice3_proc only tracks a process THIS instance of
            # the GUI started - it's None right after the program (re)starts
            # even if a service from a PREVIOUS run is still alive and
            # holding the port (its parent GUI process can exit without
            # killing it, since it isn't spawned as a detached/job-linked
            # child). Without this check we would spawn a second copy here,
            # which immediately crashes with "[Errno 10048] address already
            # in use" while the old copy just keeps answering /health - so
            # the GUI would loop "model not loaded... service exited...
            # model not loaded..." forever instead of just using it or
            # replacing it once.
            if self._cosyvoice_model_loaded(rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION):
                self.after(0, lambda: self.log(
                    "Сервис CosyVoice3 уже запущен (остался от предыдущего запуска "
                    "программы) и готов принимать запросы."
                ))
                return True
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice3 (видимо, от предыдущего "
                "запуска программы), но версия/модель не подходят — останавливаю его "
                "и запускаю заново…"
            ))
            self._kill_process_on_port(port)
            time.sleep(1.0)
        elif not need_start and self._ping_rest_service(rest_url) and not self._cosyvoice_model_loaded(
            rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION
        ):
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice3, у которого не загружена "
                "модель — останавливаю его и запускаю заново…"
            ))
            self._cosyvoice3_proc = None
            self._kill_process_on_port(port)
            time.sleep(1.0)
            need_start = True
        if need_start:
            try:
                popen_kwargs = dict(
                    cwd=str(service_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=dict(os.environ, COSYVOICE3_PORT=str(port)),
                )
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._cosyvoice3_proc = subprocess.Popen(
                    [str(cosyvoice3_py), str(service_path)], **popen_kwargs
                )
            except Exception as e:
                _log_exception_to_file("запуск сервиса CosyVoice3")
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice3: {t}"))
                return False

            def pump_output():
                proc = self._cosyvoice3_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис CosyVoice3] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            just_exited = self._cosyvoice3_proc is not None and self._cosyvoice3_proc.poll() is not None
            if not just_exited and self._ping_rest_service(rest_url):
                if self._cosyvoice_model_loaded(rest_url, expected_version=COSYVOICE3_EXPECTED_SERVICE_VERSION):
                    self.after(0, lambda: self.log("Сервис CosyVoice3 запущен и готов принимать запросы."))
                    return True
                if time.time() - last_notice > 15:
                    last_notice = time.time()
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice3 отвечает, но модель ещё не загрузилась (первый "
                        "запуск также скачивает веса, ~1-2 ГБ) — жду… Подробности в "
                        "CosyVoice3\\data\\cosyvoice3_rest_service.log."
                    ))
            if just_exited:
                self.after(0, lambda: self.log(
                    f"Сервис CosyVoice3 завершился сам собой (код {self._cosyvoice3_proc.returncode}) "
                    "— см. CosyVoice3\\data\\cosyvoice3_rest_service.log."
                ))
                return False
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис CosyVoice3 не успел запуститься за отведённое время."))
        return False
    def _ensure_cosyvoice_service_running(self, rest_url: str, ready_timeout: float = 900.0) -> bool:
        """Аналог _ensure_rest_service_running для CosyVoice: запускает
        cosyvoice_rest_service.py тем интерпретатором, что установлен
        install_cosyvoice.bat в .venv_cosyvoice (отдельно от основного
        .venv, т.к. у CosyVoice несовместимый со всем остальным набор
        зависимостей), без отдельного окна консоли. Первый запуск может
        занять несколько минут — модель загружается в память GPU.

        Движок CosyVoice3 обрабатывается отдельно (см.
        _ensure_cosyvoice3_service_running) - у него своё окружение
        (.venv_cosyvoice3) и свой порт."""
        if self._cosyvoice_engine_code() == "cosyvoice3":
            # CosyVoice3 при первом запуске ещё и скачивает веса модели
            # (~1-2 ГБ с HuggingFace) поверх обычной загрузки в память GPU,
            # так что 900 сек (15 мин) не хватало на медленном соединении -
            # сервис реально работал (что подтверждали логи), просто GUI
            # переставал ждать раньше, чем он успевал подняться. Даём
            # заметно больше времени именно для cosyvoice3, если вызывающий
            # код не передал явный ready_timeout.
            cv3_timeout = ready_timeout if ready_timeout != 900.0 else 2700.0
            return self._ensure_cosyvoice3_service_running(rest_url, ready_timeout=cv3_timeout)

        if getattr(sys, "frozen", False):
            self.after(0, lambda: self.log(
                "Автозапуск сервиса CosyVoice недоступен в собранной .exe-версии "
                "программы. Запустите программу из исходников (run_gui.bat)."
            ))
            return False

        cosyvoice_py = _app_dir() / ".venv_cosyvoice" / "Scripts" / "python.exe"
        if not cosyvoice_py.exists():
            # на всякий случай также проверяем "не-Windows" раскладку venv
            alt = _app_dir() / ".venv_cosyvoice" / "bin" / "python"
            cosyvoice_py = alt if alt.exists() else cosyvoice_py
        service_dir = _app_dir() / "CosyVoice"
        service_path = service_dir / "cosyvoice_rest_service.py"

        if not cosyvoice_py.exists() or not service_path.exists():
            self.after(0, lambda: self.log(
                "Режим CosyVoice ещё не установлен на этом компьютере. Запустите "
                "install.bat ещё раз (он теперь заодно ставит и CosyVoice — это "
                "большая, отдельная от остального загрузка, может занять время)."
            ))
            return False

        try:
            from urllib.parse import urlparse
            port = urlparse(rest_url).port or 5011
        except Exception:
            port = 5011
        port_retry_done = False

        need_start = self._cosyvoice_proc is None or self._cosyvoice_proc.poll() is not None
        if not need_start and self._ping_rest_service(rest_url) and not self._cosyvoice_model_loaded(rest_url):
            # Сервис отвечает (например, остался запущенным с прошлого
            # раза), но модель CosyVoice у него не загрузилась — такой
            # "зомби"-процесс нужно принудительно остановить и поднять
            # заново, иначе программа будет считать его рабочим и получать
            # HTTP 500 на каждый запрос без единой новой попытки перезапуска.
            self.after(0, lambda: self.log(
                "Обнаружен уже запущенный сервис CosyVoice, у которого не загружена "
                "модель (скорее всего остался с прошлого раза, до недавнего "
                "исправления) — останавливаю его и запускаю заново…"
            ))
            self._cosyvoice_proc = None
            self._kill_process_on_port(port)
            time.sleep(1.0)
            need_start = True
        if need_start:
            try:
                popen_env = dict(os.environ)
                popen_env["TTS_ENGINE"] = self._cosyvoice_engine_code()
                popen_kwargs = dict(
                    cwd=str(service_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=popen_env,
                )
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._cosyvoice_proc = subprocess.Popen(
                    [str(cosyvoice_py), str(service_path)], **popen_kwargs
                )
            except Exception as e:
                _log_exception_to_file("запуск сервиса CosyVoice")
                err = str(e)
                self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice: {t}"))
                return False

            def pump_output():
                proc = self._cosyvoice_proc
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.after(0, lambda l=line: self.log("  [сервис CosyVoice] " + l))

            threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.time() + ready_timeout
        last_notice = 0.0
        while time.time() < deadline:
            # ВАЖНО: сначала проверяем, не завершился ли только что нами
            # же запущенный процесс, и только потом - пинг. Если проверять
            # в обратном порядке, то в ситуации "старый зомби-процесс всё
            # ещё держит порт" пинг будет успешно отвечать (это ответ
            # зомби, а не нашего свежего процесса), и мы никогда не
            # заметим, что наш новый процесс уже упал с "порт занят".
            just_exited = self._cosyvoice_proc is not None and self._cosyvoice_proc.poll() is not None
            if not just_exited and self._ping_rest_service(rest_url):
                # Сервис отвечает - но это ещё не значит, что модель
                # загрузилась (uvicorn начинает принимать запросы даже
                # если startup_event поймал исключение при загрузке модели
                # - см. _cosyvoice_model_loaded). Дожидаемся именно этого,
                # а не просто первого ответа сервера.
                if self._cosyvoice_model_loaded(rest_url):
                    self.after(0, lambda: self.log("Сервис CosyVoice запущен и готов принимать запросы."))
                    return True
                if time.time() - last_notice > 15:
                    last_notice = time.time()
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice отвечает, но модель ещё не загрузилась (или не "
                        "смогла загрузиться) — жду… Если это долго не проходит, смотрите "
                        "подробности в CosyVoice\\cosyvoice_rest_service.log."
                    ))
                time.sleep(1.0)
                continue
            if just_exited:
                if not port_retry_done:
                    # Частый случай: старый ("зомби") процесс ещё не успел
                    # полностью освободить порт к моменту, когда новый уже
                    # пытался на него забиндиться ("address already in
                    # use") - добираем ещё раз, уже с более долгим
                    # ожиданием освобождения порта, и пробуем один раз ещё.
                    port_retry_done = True
                    self.after(0, lambda: self.log(
                        "Сервис CosyVoice завершился сразу после запуска — возможно, порт "
                        "ещё не освободился от предыдущего процесса. Пробую ещё раз…"
                    ))
                    self._cosyvoice_proc = None
                    self._kill_process_on_port(port, wait_free_timeout=12.0)
                    time.sleep(1.0)
                    try:
                        popen_env2 = dict(os.environ)
                        popen_env2["TTS_ENGINE"] = self._cosyvoice_engine_code()
                        popen_kwargs = dict(
                            cwd=str(service_dir),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=popen_env2,
                        )
                        if sys.platform == "win32":
                            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                        self._cosyvoice_proc = subprocess.Popen(
                            [str(cosyvoice_py), str(service_path)], **popen_kwargs
                        )

                        def pump_output2():
                            proc = self._cosyvoice_proc
                            if proc is None or proc.stdout is None:
                                return
                            for line in proc.stdout:
                                line = line.rstrip()
                                if line:
                                    self.after(0, lambda l=line: self.log("  [сервис CosyVoice] " + l))

                        threading.Thread(target=pump_output2, daemon=True).start()
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        _log_exception_to_file("повторный запуск сервиса CosyVoice")
                        err = str(e)
                        self.after(0, lambda t=err: self.log(f"Не удалось запустить сервис CosyVoice: {t}"))
                        return False
                self.after(0, lambda: self.log(
                    f"Сервис CosyVoice завершился сам собой (код {self._cosyvoice_proc.returncode}) "
                    "во время запуска — см. сообщения [сервис CosyVoice] выше."
                ))
                return False
            if time.time() - last_notice > 15:
                last_notice = time.time()
                self.after(0, lambda: self.log(
                    "…сервис CosyVoice ещё запускается (модель грузится в память GPU, "
                    "при первом запуске это может занять несколько минут)…"
                ))
            time.sleep(1.0)

        self.after(0, lambda: self.log("Сервис CosyVoice не успел запуститься за отведённое время."))
        return False

    # ------------------------------------------------------------------
    # Startup recycle of the Silero REST service
    # ------------------------------------------------------------------
    # On launch: find the old silero_rest_service.py process(es) (by command line AND by the
    # Python process that listens on the service port), kill them with their child trees,
    # wait for the port to be released, then start a fresh service with the same
    # interpreter / working folder the normal auto-start uses. Every step is logged to the
    # GUI log (which is also written to the log file). Nothing here may raise into the GUI.

    @staticmethod
    def _run_hidden(cmd: list, timeout: float = 20.0):
        """Runs a console tool without a window. Returns (returncode, stdout) or (None, "")."""
        try:
            kw = dict(capture_output=True, text=True, errors="replace", timeout=timeout)
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            r = subprocess.run(cmd, **kw)
            return r.returncode, (r.stdout or "")
        except Exception:
            return None, ""

    @staticmethod
    def _parse_process_json(text: str) -> list:
        """PowerShell ConvertTo-Json gives one object, a list, or nothing -> [{pid, cmd}]."""
        text = (text or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except ValueError:
            return []
        if isinstance(data, dict):
            data = [data]
        out = []
        for item in data if isinstance(data, list) else []:
            try:
                out.append({"pid": int(item.get("ProcessId")), "cmd": str(item.get("CommandLine") or ""),
                            "name": str(item.get("Name") or "").lower()})
            except (TypeError, ValueError, AttributeError):
                continue
        return out

    @classmethod
    def _list_python_processes(cls) -> list:
        rc, out = cls._run_hidden([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe' "
            "OR Name='cmd.exe'\" | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ])
        return cls._parse_process_json(out) if rc == 0 else []

    @classmethod
    def _pid_alive(cls, pid: int) -> bool:
        rc, out = cls._run_hidden(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], timeout=10.0)
        return rc is not None and f'"{pid}"' in out

    @classmethod
    def _safe_port_holders(cls, port: int) -> set:
        try:
            return {int(x) for x in cls._pids_listening_on_port(port)}
        except Exception:
            return set()

    @classmethod
    def _find_service_pids(cls, script_name: str, port: int) -> dict:
        """{pid: reason} of processes to stop: any python process whose command line contains
        script_name, plus any PYTHON process listening on `port`. A non-python process that
        holds the port is deliberately left alone (reported by the caller)."""
        own = os.getpid()
        procs = cls._list_python_processes()
        by_pid = {p["pid"]: p["cmd"] for p in procs if p.get("name", "python.exe").startswith("python")}
        found = {}
        # the run_service_forever.bat wrapper restarts the service 3 s after it dies - it must go first
        for p in procs:
            if p.get("name") == "cmd.exe" and "run_service_forever" in p["cmd"].lower():
                found[p["pid"]] = "run_service_forever.bat wrapper (would respawn the service)"
        for p in procs:
            if p.get("name", "python.exe").startswith("python") \
                    and p["pid"] != own and script_name.lower() in p["cmd"].lower():
                found[p["pid"]] = f"runs {script_name}"
        holders = cls._safe_port_holders(port)
        for pid in holders:
            if pid == own or pid in found:
                continue
            if pid in by_pid:
                found[pid] = f"python process listening on port {port}"
            else:
                found[-pid] = f"NOT a python process (pid {pid}) holds port {port} - left alone"
        return found

    @classmethod
    def _kill_pid_tree(cls, pid: int) -> str:
        """Returns "killed", "already gone" or "denied" (still alive after taskkill)."""
        if not cls._pid_alive(pid):
            return "already gone"
        cls._run_hidden(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=20.0)
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if not cls._pid_alive(pid):
                return "killed"
            time.sleep(0.4)
        return "denied"

    def _startup_recycle_services(self):
        """Entry point, called once from the GUI initialisation. Runs in a worker thread."""
        try:
            if not self.recycle_services_var.get():
                return
        except Exception:
            return
        if sys.platform != "win32":
            self.log("Service recycle: skipped (Windows-only: uses taskkill/tasklist).")
            return
        rest_url = self.rest_url_var.get().strip()
        auto_start = bool(self.auto_start_rest_var.get())
        threading.Thread(target=self._recycle_rest_service, args=(rest_url, auto_start),
                         daemon=True).start()

    def _recycle_rest_service(self, rest_url: str, auto_start: bool):
        def log(msg: str):
            self.after(0, lambda m=msg: self.log("[service recycle] " + m))

        try:
            if getattr(sys, "frozen", False):
                log("skipped: the frozen .exe build cannot restart silero_rest_service.py.")
                return
            if not self._rest_url_is_local(rest_url):
                log(f"skipped: {rest_url} is not a local address.")
                return
            from urllib.parse import urlparse
            port = urlparse(rest_url).port or 5010
            with self._rest_svc_lock:
                log(f"looking for old Silero REST service processes (script or port {port})...")
                found = self._find_service_pids("silero_rest_service.py", port)
                to_kill = {p: why for p, why in found.items() if p > 0}
                for p, why in found.items():
                    if p < 0:
                        log("WARNING: " + why)
                if not to_kill:
                    log("no old service process found.")
                denied = []
                for pid, why in to_kill.items():
                    res = self._kill_pid_tree(pid)
                    log(f"PID {pid} ({why}): {res}.")
                    if res == "denied":
                        denied.append(pid)
                if denied:
                    log(f"WARNING: could not stop PID(s) {denied} (access denied?). Close them manually or "
                        f"start this program as administrator. A fresh service was NOT started.")
                    return
                if to_kill:
                    deadline = time.time() + 10.0
                    while time.time() < deadline and self._safe_port_holders(port):
                        time.sleep(0.5)
                if self._safe_port_holders(port):
                    log(f"WARNING: port {port} is still busy - a fresh service was NOT started.")
                    return
                if not auto_start:
                    log("old processes cleaned up; auto-start of the service is switched off, so no "
                        "new one is started (it starts on demand when you press the start button).")
                    return
                log("starting a fresh service...")
                ok = self._ensure_rest_service_running_impl(rest_url)
                if ok:
                    pid = self._rest_proc.pid if self._rest_proc is not None else "?"
                    log(f"fresh Silero REST service is ACTIVE (PID {pid}, {rest_url}).")
                else:
                    log("the fresh service did NOT come up - see the [сервис Silero REST] lines above.")
        except Exception as e:
            _log_exception_to_file("startup recycle of the Silero REST service")
            log(f"failed with {type(e).__name__}: {e} (the program keeps working; the service starts on demand).")
