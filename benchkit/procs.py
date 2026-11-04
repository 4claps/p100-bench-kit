"""Child process helpers: every external command is launched from an argv
list (never a shell string), tracked by PID from launch, logged line by line
and killed with its whole process group on abort or timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable


class Proc:
    def __init__(self, argv: list[str], env: dict | None = None, cwd: str | None = None,
                 on_line: Callable[[str], None] | None = None, preexec_uid: int | None = None):
        self.argv = argv
        kw: dict = {}
        if preexec_uid is not None and os.geteuid() == 0:
            kw["user"] = preexec_uid
            kw["group"] = preexec_uid
            kw["extra_groups"] = []
        self.p = subprocess.Popen(
            argv, env={**os.environ, **(env or {})}, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            start_new_session=True, **kw)
        self.pid = self.p.pid
        self.lines: list[str] = []
        self._on_line = on_line
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        assert self.p.stdout
        for line in self.p.stdout:
            line = line.rstrip("\n")
            self.lines.append(line)
            if len(self.lines) > 20000:
                del self.lines[:10000]
            if self._on_line:
                try:
                    self._on_line(line)
                except Exception:  # noqa: BLE001
                    pass

    def alive(self) -> bool:
        return self.p.poll() is None

    def wait(self, timeout: float | None = None, abort: threading.Event | None = None) -> int | None:
        """Wait for exit; returns the exit code, or None on timeout/abort (the
        process group is killed in that case)."""
        t0 = time.time()
        while self.alive():
            if abort is not None and abort.is_set():
                self.kill()
                return None
            if timeout is not None and time.time() - t0 > timeout:
                self.kill()
                return None
            time.sleep(0.2)
        self._reader.join(timeout=5)
        return self.p.returncode

    def kill(self, grace: float = 10) -> None:
        if not self.alive():
            return
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        t0 = time.time()
        while self.alive() and time.time() - t0 < grace:
            time.sleep(0.2)
        if self.alive():
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.p.wait(timeout=10)
