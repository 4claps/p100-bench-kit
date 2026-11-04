"""SQLite persistence: runs, module results, agent task results, telemetry,
logs, events, chart layouts and presets.

Adapted from the bakeoff app's db.py: same WAL + short-transaction pattern,
with a generic results table so every module (speed, server, agent, offload)
stores its numbers the same way.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    status TEXT NOT NULL,        -- created, running, done, failed, aborted, interrupted, imported
    spec_json TEXT NOT NULL,     -- what was asked for (model, engine, gpus, modules + params)
    snapshot_json TEXT,          -- hardware/software snapshot at start
    estimate_s INTEGER,
    error TEXT,
    note TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    invalid INTEGER NOT NULL DEFAULT 0,
    invalid_reason TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    module TEXT NOT NULL,        -- speed | server | agent | offload
    rep INTEGER NOT NULL,
    variant TEXT NOT NULL,       -- e.g. "kv=q4_0" or "mtp=on"
    metric TEXT NOT NULL,        -- e.g. "pp512@0", "tg128@16384", "code.tg"
    value REAL,
    unit TEXT NOT NULL DEFAULT '',
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS results_run ON results(run_id);

CREATE TABLE IF NOT EXISTS agent_tasks (
    run_id TEXT NOT NULL,
    task TEXT NOT NULL,
    rep INTEGER NOT NULL,
    wall_s REAL,
    exit_code INTEGER,
    success INTEGER,
    tail TEXT,
    PRIMARY KEY (run_id, task, rep)
);

CREATE TABLE IF NOT EXISTS telemetry (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    t REAL NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS logs (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    t REAL NOT NULL,
    stream TEXT NOT NULL,        -- runner | server | bench | agent
    line TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    t REAL NOT NULL,
    level TEXT NOT NULL,         -- info | warn | error
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS layouts (
    name TEXT PRIMARY KEY,
    charts_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    name TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

RUN_FIELDS = {"status", "started_at", "finished_at", "snapshot_json", "estimate_s",
              "error", "note", "tags", "invalid", "invalid_reason"}


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as c:
            c.executescript(SCHEMA)
            c.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                      (str(SCHEMA_VERSION),))
        self._seq: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def tx(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- runs ----------------
    def create_run(self, run_id: str, spec: dict, estimate_s: int, status: str = "created") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO runs (id, created_at, status, spec_json, estimate_s) "
                      "VALUES (?,?,?,?,?)", (run_id, time.time(), status, json.dumps(spec), estimate_s))

    def update_run(self, run_id: str, **fields: Any) -> None:
        bad = set(fields) - RUN_FIELDS
        if bad:
            raise ValueError(f"unknown run fields: {bad}")
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self.tx() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return _run_row(r) if r else None

    def list_runs(self, limit: int = 500) -> list[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_run_row(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self.tx() as c:
            for t in ("results", "agent_tasks", "telemetry", "logs", "events"):
                c.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
            c.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def mark_unfinished_interrupted(self) -> list[str]:
        """On startup: any run still marked running died with the previous
        process (everything runs inside this container, so its children are
        gone too)."""
        with self.tx() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM runs WHERE status IN ('running','starting')")]
            c.execute("UPDATE runs SET status='interrupted', finished_at=?, "
                      "error='app restarted during the run' WHERE status IN ('running','starting')",
                      (time.time(),))
        return ids

    # ---------------- results ----------------
    def add_result(self, run_id: str, module: str, rep: int, variant: str, metric: str,
                   value: float | None, unit: str = "", extra: dict | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                      (run_id, module, rep, variant, metric, value, unit,
                       json.dumps(extra) if extra else None))

    def get_results(self, run_id: str) -> list[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT * FROM results WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()
        return [_with_json(dict(r), "extra_json", "extra") for r in rows]

    def add_agent_task(self, run_id: str, task: str, rep: int, wall_s: float,
                       exit_code: int, success: bool, tail: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO agent_tasks VALUES (?,?,?,?,?,?,?)",
                      (run_id, task, rep, wall_s, exit_code, int(success), tail))

    def get_agent_tasks(self, run_id: str) -> list[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT * FROM agent_tasks WHERE run_id=? ORDER BY rep, task",
                             (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["success"] = bool(d["success"])
            out.append(d)
        return out

    # ---------------- telemetry / logs / events ----------------
    def _next(self, run_id: str, table: str, c: sqlite3.Connection) -> int:
        key = (run_id, table)
        if key not in self._seq:
            r = c.execute(f"SELECT COALESCE(MAX(seq), -1) FROM {table} WHERE run_id=?", (run_id,)).fetchone()
            self._seq[key] = r[0]
        self._seq[key] += 1
        return self._seq[key]

    def add_telemetry(self, run_id: str, samples: Iterable[dict]) -> None:
        with self._lock, self.tx() as c:
            for s in samples:
                c.execute("INSERT INTO telemetry VALUES (?,?,?,?)",
                          (run_id, self._next(run_id, "telemetry", c), s["t"], json.dumps(s)))

    def get_telemetry(self, run_id: str, max_points: int = 2000) -> list[dict]:
        with self.tx() as c:
            n = c.execute("SELECT COUNT(*) FROM telemetry WHERE run_id=?", (run_id,)).fetchone()[0]
            step = max(1, -(-n // max_points))
            rows = c.execute("SELECT payload FROM telemetry WHERE run_id=? AND seq % ? = 0 ORDER BY seq",
                             (run_id, step)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def add_log(self, run_id: str, stream: str, lines: Iterable[str]) -> None:
        now = time.time()
        with self._lock, self.tx() as c:
            for line in lines:
                c.execute("INSERT INTO logs VALUES (?,?,?,?,?)",
                          (run_id, self._next(run_id, "logs", c), now, stream, line))

    def get_logs(self, run_id: str, stream: str | None = None, tail: int = 500) -> list[dict]:
        q = "SELECT seq, t, stream, line FROM logs WHERE run_id=?"
        args: list = [run_id]
        if stream:
            q += " AND stream=?"
            args.append(stream)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(tail)
        with self.tx() as c:
            rows = c.execute(q, args).fetchall()
        return [dict(r) for r in reversed(rows)]

    def add_event(self, run_id: str | None, level: str, kind: str, message: str,
                  payload: dict | None = None) -> dict:
        ev = {"run_id": run_id, "t": time.time(), "level": level, "kind": kind,
              "message": message, "payload": payload}
        with self.tx() as c:
            c.execute("INSERT INTO events (run_id, t, level, kind, message, payload) VALUES (?,?,?,?,?,?)",
                      (run_id, ev["t"], level, kind, message, json.dumps(payload) if payload else None))
        return ev

    def get_events(self, run_id: str, limit: int = 500) -> list[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT * FROM events WHERE run_id=? ORDER BY id LIMIT ?",
                             (run_id, limit)).fetchall()
        return [_with_json(dict(r), "payload", "payload") for r in rows]

    # ---------------- layouts / presets ----------------
    def get_layout(self, name: str) -> list | None:
        with self.tx() as c:
            r = c.execute("SELECT charts_json FROM layouts WHERE name=?", (name,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_layout(self, name: str, charts: list) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO layouts VALUES (?,?,?)", (name, json.dumps(charts), time.time()))

    def list_presets(self) -> list[dict]:
        with self.tx() as c:
            rows = c.execute("SELECT name, spec_json FROM presets ORDER BY name").fetchall()
        return [{"name": r["name"], "spec": json.loads(r["spec_json"])} for r in rows]

    def put_preset(self, name: str, spec: dict) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO presets VALUES (?,?,?)", (name, json.dumps(spec), time.time()))

    def delete_preset(self, name: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM presets WHERE name=?", (name,))


def _with_json(d: dict, src: str, dst: str) -> dict:
    raw = d.pop(src, None)
    d[dst] = json.loads(raw) if raw else None
    return d


def _run_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["spec"] = json.loads(d.pop("spec_json"))
    raw = d.pop("snapshot_json")
    d["snapshot"] = json.loads(raw) if raw else None
    d["invalid"] = bool(d["invalid"])
    d["tags"] = [t for t in d["tags"].split(",") if t]
    return d
