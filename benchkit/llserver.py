"""Start, health-check and stop one llama-server owned by a run."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .engines import env_for, split_args
from .llama_log import parse_print_timing
from .procs import Proc


class LlamaServer:
    def __init__(self, ctx, extra_args: list[str], ctx_size: int, mtp: bool = False):
        self.ctx = ctx
        e = ctx.engine
        self.port = ctx.cfg.server_port
        self.argv = [f"{e['bin_dir']}/llama-server", "-m", str(ctx.model_path),
                     "--host", "127.0.0.1", "--port", str(self.port),
                     "-ngl", "99", "-fa", "on", "-c", str(ctx_size), "-np", "1", "--jinja",
                     *split_args(e, ctx.gpus), *extra_args]
        if mtp:
            if not e.get("mtp_args"):
                raise RuntimeError(f"engine {e['name']} has no MTP support configured")
            self.argv += e["mtp_args"]
        self.proc: Proc | None = None
        self.base = f"http://127.0.0.1:{self.port}"

    def _on_line(self, line: str) -> None:
        self.ctx.log("server", line)
        ev = parse_print_timing(line)
        if ev:
            self.ctx.server_timing(ev)

    def start(self, timeout: float = 1200) -> None:
        self.ctx.log("runner", "starting llama-server: " + " ".join(self.argv))
        self.proc = Proc(self.argv, env=env_for(self.ctx.engine, self.ctx.gpus), on_line=self._on_line)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ctx.abort.is_set():
                raise RuntimeError("aborted")
            if not self.proc.alive():
                tail = "\n".join(self.proc.lines[-15:])
                raise RuntimeError(f"llama-server exited during startup (rc={self.proc.p.returncode})\n{tail}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                    if r.status == 200:
                        self.ctx.log("runner", f"llama-server ready after {time.time() - t0:.0f}s")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become healthy in time")

    def chat(self, body: dict, timeout: float = 1800) -> dict:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        if self.proc:
            self.proc.kill()
            self.ctx.log("runner", "llama-server stopped")
            self.proc = None

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()  # __exit__ is not called when __enter__ raises
            raise
        return self

    def __exit__(self, *exc):
        self.stop()
