"""App entry point: `python -m benchkit.main` (or uvicorn benchkit.main:app)."""

from __future__ import annotations

import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Config
from .db import DB
from .models import ModelStore
from .runner import KIT_VERSION, Runner

FRONTEND = Path(__file__).resolve().parent / "frontend"
PUBLIC = {"/", "/healthz", "/api/login", "/favicon.ico"}


class Sessions:
    """Login sessions, kept in the data dir so a container restart doesn't log everyone out."""

    TTL = 30 * 86400

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            self._s = {k: v for k, v in json.loads(path.read_text()).items() if v > time.time()}
        except (OSError, ValueError):
            self._s = {}

    def _save(self):
        self.path.write_text(json.dumps(self._s))
        self.path.chmod(0o600)

    def create(self) -> str:
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._s[sid] = time.time() + self.TTL
            self._save()
        return sid

    def valid(self, sid: str) -> bool:
        exp = self._s.get(sid)
        return bool(exp and exp > time.time())

    def drop(self, sid: str) -> None:
        with self._lock:
            if self._s.pop(sid, None):
                self._save()


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.load_token()
        db = DB(cfg.db_path)
        for rid in db.mark_unfinished_interrupted():
            db.add_event(rid, "error", "interrupted", "app restarted during the run")
        app.state.cfg = cfg
        app.state.db = db
        app.state.store = ModelStore(cfg.models_dir, cfg.data_dir, cfg.manifest)
        app.state.runner = Runner(cfg, db, app.state.store)
        app.state.sessions = Sessions(cfg.data_dir / "sessions.json")
        app.state.kit_version = KIT_VERSION
        print(f"benchkit {KIT_VERSION} on http://{cfg.host}:{cfg.port}  "
              f"(login token: {cfg.data_dir / 'token'} or BENCHKIT_TOKEN)", flush=True)
        yield
        if app.state.runner.active:
            app.state.runner.active.abort.set()

    app = FastAPI(title="P100 bench kit", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        path = request.url.path
        if path in PUBLIC or path.startswith("/static/"):
            return await call_next(request)
        st = request.app.state
        hdr = request.headers.get("authorization", "")
        ok = (hdr.startswith("Bearer ") and secrets.compare_digest(hdr[7:].strip().encode(), st.cfg.token.encode())) \
            or st.sessions.valid(request.cookies.get("benchkit_session", ""))
        if not ok:
            return JSONResponse({"detail": "login required"}, status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    app.include_router(router, prefix="/api")
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND / "index.html")

    return app


app = create_app()


def main():
    import uvicorn
    cfg = Config()
    uvicorn.run("benchkit.main:app", host=cfg.host, port=cfg.port, log_level="info", proxy_headers=False)


if __name__ == "__main__":
    main()
