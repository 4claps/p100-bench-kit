"""Run lifecycle: validate a spec, preflight, execute the chosen modules in
a worker thread while a sampler thread records telemetry and evaluates
guardrails, then finish with a status. One run at a time.

Everything runs inside this container, so there is no remote service to
stop or restore: a run owns the GPUs it was given for its duration, and if
the app restarts mid-run the children are gone with it (the run is marked
interrupted on the next start).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import engines as engines_mod
from . import hw
from .config import Config
from .db import DB
from .guardrails import Guardrails
from .models import ModelStore
from .modules import MODULE_ORDER, MODULES
from .telemetry import Sampler

KIT_VERSION = "0.1.0"


class RunContext:
    """What a module sees. All methods are safe to call from the worker thread."""

    def __init__(self, runner: "Runner", run_id: str, spec: dict, engine: dict, model_path: Path,
                 model_meta: dict | None, snapshot: dict):
        self.runner = runner
        self.cfg = runner.cfg
        self.db = runner.db
        self.run_id = run_id
        self.spec = spec
        self.engine = engine
        self.model_path = model_path
        self.model_meta = model_meta
        self.gpus: list[int] = spec["gpus"]
        self.snapshot = snapshot
        self.abort = threading.Event()
        self.abort_reason = ""
        self.server_live: dict = {}
        self._log_buf: list[tuple[str, str]] = []
        self._log_lock = threading.Lock()

    def log(self, stream: str, line: str) -> None:
        with self._log_lock:
            self._log_buf.append((stream, line))
        self.runner.emit({"type": "log", "run_id": self.run_id, "stream": stream, "line": line})

    def flush_logs(self) -> None:
        with self._log_lock:
            buf, self._log_buf = self._log_buf, []
        by: dict[str, list[str]] = {}
        for s, l in buf:
            by.setdefault(s, []).append(l)
        for s, lines in by.items():
            self.db.add_log(self.run_id, s, lines)

    def event(self, level: str, kind: str, message: str, payload: dict | None = None) -> None:
        ev = self.db.add_event(self.run_id, level, kind, message, payload)
        self.runner.emit({"type": "event", **ev})

    def result(self, module, rep, variant, metric, value, unit="", extra=None) -> None:
        self.db.add_result(self.run_id, module, rep, variant, metric, value, unit, extra)
        self.runner.emit({"type": "result", "run_id": self.run_id, "module": module, "rep": rep,
                          "variant": variant, "metric": metric, "value": value, "unit": unit})

    def agent_task(self, task, rep, wall_s, exit_code, success, tail) -> None:
        self.db.add_agent_task(self.run_id, task, rep, wall_s, exit_code, success, tail)
        self.runner.emit({"type": "agent_task", "run_id": self.run_id, "task": task, "rep": rep,
                          "wall_s": wall_s, "exit_code": exit_code, "success": success})

    def progress(self, module: str, step: int, steps: int, label: str) -> None:
        self.runner.progress[self.run_id] = {"module": module, "step": step, "steps": steps,
                                             "label": label, "t": time.time()}
        self.runner.emit({"type": "progress", "run_id": self.run_id, **self.runner.progress[self.run_id]})
        self.log("runner", f"[{module}] {step}/{steps} {label}")

    def server_timing(self, ev) -> None:
        if ev.kind == "prompt":
            self.server_live["pp_tps"] = ev.tokens_per_s
        elif ev.kind == "eval":
            self.server_live["tg_tps"] = ev.tokens_per_s
        elif ev.kind == "draft":
            self.server_live["draft_accept"] = ev.acceptance

    def cool_down(self) -> None:
        """Wait until the run's GPUs are at or below the cool-down temperature."""
        t0 = time.time()
        sampler = Sampler()
        while not self.abort.is_set():
            temp = sampler.max_gpu_temp(self.gpus)
            if temp is None or temp <= self.cfg.cooldown_c:
                if time.time() - t0 > 5:
                    self.log("runner", f"cooled to {temp} °C in {time.time() - t0:.0f}s")
                return
            if time.time() - t0 > self.cfg.cooldown_timeout_s:
                self.event("warn", "cooldown_timeout",
                           f"GPUs still at {temp} °C after {self.cfg.cooldown_timeout_s:.0f}s; continuing")
                return
            time.sleep(5)


class Runner:
    def __init__(self, cfg: Config, db: DB, store: ModelStore):
        self.cfg = cfg
        self.db = db
        self.store = store
        self.engines = engines_mod.discover(cfg.engines_dir)
        self.active: RunContext | None = None
        self.progress: dict[str, dict] = {}
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        self.snapshot_cache: dict | None = None

    # ---------------- pub/sub for SSE ----------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs = [(l, s) for l, s in self._subs if s is not q]

    def emit(self, ev: dict) -> None:
        for loop, q in list(self._subs):
            def put(q=q, ev=ev):
                if not q.full():
                    q.put_nowait(ev)
            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:
                pass

    # ---------------- system ----------------
    def snapshot(self, refresh: bool = False) -> dict:
        if refresh or self.snapshot_cache is None:
            self.snapshot_cache = hw.snapshot(engines=[
                {k: e.get(k) for k in ("name", "label", "repo", "commit", "cuda_archs", "multi_gpu_split")}
                for e in self.engines])
        return self.snapshot_cache

    def engine(self, name: str) -> dict | None:
        return next((e for e in self.engines if e["name"] == name and e["available"]), None)

    # ---------------- validation / estimate / preflight ----------------
    def validate(self, spec: dict) -> dict:
        """Normalize a run spec from the UI/API. Raises ValueError on bad input."""
        if not isinstance(spec, dict):
            raise ValueError("spec must be an object")
        model_id = str(spec.get("model") or "")
        engine = self.engine(str(spec.get("engine") or ""))
        if not engine:
            raise ValueError("pick an available engine")
        n_gpus = len(self.snapshot().get("gpus") or [])
        gpus = sorted({int(g) for g in spec.get("gpus") or []})
        if not gpus:
            raise ValueError("pick at least one GPU")
        if n_gpus and any(g < 0 or g >= n_gpus for g in gpus):
            raise ValueError(f"GPU index out of range (0..{n_gpus - 1})")
        mods = spec.get("modules") or {}
        if not isinstance(mods, dict) or not mods:
            raise ValueError("pick at least one module")
        norm = {}
        for name in MODULE_ORDER:
            if name in mods:
                norm[name] = MODULES[name].normalize(mods[name])
        unknown = set(mods) - set(MODULES)
        if unknown:
            raise ValueError(f"unknown modules: {sorted(unknown)}")
        path, comparable = self.store.model_path(model_id)
        if path is None:
            raise ValueError("model is not present on this machine (download it first)")
        split = str(spec.get("split") or "")
        if split and split not in engines_mod.SPLIT_MODES:
            raise ValueError(f"split must be one of {engines_mod.SPLIT_MODES} (or empty for the engine default)")
        # both guardrails abort by default; an explicit [] turns them off
        abort_on = [a for a in spec.get("abort_on", ["low_clock", "thermal"]) or [] if a in ("low_clock", "thermal")]
        return {"model": model_id, "engine": engine["name"], "gpus": gpus, "modules": norm,
                "split": split or engine.get("multi_gpu_split", "layer"), "abort_on": abort_on,
                "label": str(spec.get("label") or "")[:120], "comparable": comparable}

    def estimate(self, spec: dict) -> int:
        path, _ = self.store.model_path(spec["model"])
        gb = path.stat().st_size / 1e9 if path else 10
        return sum(MODULES[n].estimate_s(p, gb) for n, p in spec["modules"].items())

    def preflight(self, spec: dict) -> dict:
        """Checks before starting. 'block' items stop the run; 'warn' items are shown."""
        block, warn = [], []
        snap = self.snapshot(refresh=True)
        engine = self.engine(spec["engine"])
        model_meta = self.store.get(spec["model"])
        status = self.store.status(model_meta) if model_meta else None
        if self.active:
            block.append(f"run {self.active.run_id} is still running")
        if model_meta and status and not status["verified"]:
            warn.append("model checksum not verified yet (Models page → Verify); results may not be comparable")
        if not spec["comparable"]:
            warn.append("custom model: results are marked not comparable with other machines")
        for name, p in spec["modules"].items():
            for prob in MODULES[name].check(p, model_meta, engine, snap):
                block.append(f"{name}: {prob}")
        ok, out = hw.run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid",
                              "--format=csv,noheader"])
        if ok and out.strip():
            uuids = {g["uuid"] for g in snap.get("gpus", []) if g.get("index") in spec["gpus"]}
            busy = [l for l in out.strip().splitlines() if any(u in l for u in uuids)]
            if busy:
                block.append("selected GPUs are in use by other processes: " + "; ".join(busy)[:300])
        if not snap.get("gpus"):
            block.append("no NVIDIA GPUs visible (start the container with --gpus all)")
        free = shutil.disk_usage(self.cfg.data_dir).free
        if free < 5e9:
            warn.append(f"only {free / 1e9:.1f} GB free in the data dir")
        vram = sum((g.get("memory_total") or 0) for g in snap.get("gpus", []) if g.get("index") in spec["gpus"]) / 1024
        if model_meta and vram and model_meta["vram_gb"] > vram and set(spec["modules"]) - {"offload"}:
            warn.append(f"model needs about {model_meta['vram_gb']} GB VRAM; selected GPUs have {vram:.0f} GB")
        for g in snap.get("gpus", []):
            if g.get("index") in spec["gpus"] and g.get("pcie_link_width_max") and \
                    g.get("pcie_link_width_current") and g["pcie_link_width_current"] < g["pcie_link_width_max"]:
                warn.append(f"GPU {g['index']} PCIe link is x{g['pcie_link_width_current']} "
                            f"(card supports x{g['pcie_link_width_max']})")
        return {"block": block, "warn": warn}

    # ---------------- run ----------------
    def create(self, spec_in: dict) -> dict:
        spec = self.validate(spec_in)
        run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        est = self.estimate(spec)
        self.db.create_run(run_id, spec, est)
        return {"id": run_id, "spec": spec, "estimate_s": est, "preflight": self.preflight(spec)}

    def start(self, run_id: str) -> None:
        with self._lock:
            run = self.db.get_run(run_id)
            if not run or run["status"] != "created":
                raise ValueError("run not found or already started")
            if self.active:
                raise ValueError("another run is active")
            spec = run["spec"]
            pf = self.preflight(spec)
            if pf["block"]:
                raise ValueError("preflight blocked: " + "; ".join(pf["block"]))
            snap = self.snapshot()
            path, _ = self.store.model_path(spec["model"])
            ctx = RunContext(self, run_id, spec, self.engine(spec["engine"]), path,
                             self.store.get(spec["model"]), snap)
            self.active = ctx
            self.db.update_run(run_id, status="running", started_at=time.time(),
                               snapshot_json=json.dumps(snap))
        threading.Thread(target=self._sample_loop, args=(ctx,), daemon=True, name="sampler").start()
        threading.Thread(target=self._work, args=(ctx,), daemon=True, name="worker").start()
        self.emit({"type": "status", "run_id": run_id, "status": "running"})

    def abort(self, run_id: str, reason: str = "aborted by user") -> None:
        ctx = self.active
        if not ctx or ctx.run_id != run_id:
            raise ValueError("run is not active")
        ctx.abort_reason = reason
        ctx.abort.set()
        ctx.event("warn", "abort", reason)

    def _work(self, ctx: RunContext) -> None:
        status, error = "done", None
        try:
            ctx.log("runner", f"run {ctx.run_id}: engine {ctx.engine['name']}, model {ctx.model_path.name}, "
                              f"GPUs {ctx.gpus}, modules {list(ctx.spec['modules'])}")
            for name, params in ctx.spec["modules"].items():
                if ctx.abort.is_set():
                    break
                ctx.event("info", "module_start", f"{name} started")
                MODULES[name].run(ctx, params)
                ctx.event("info", "module_done", f"{name} finished")
            if ctx.abort.is_set():
                status, error = "aborted", ctx.abort_reason
        except Exception as e:  # noqa: BLE001
            status, error = ("aborted", ctx.abort_reason) if ctx.abort.is_set() else ("failed", str(e))
            ctx.log("runner", traceback.format_exc())
        finally:
            ctx.abort.set()  # stops the sampler
            ctx.flush_logs()
            self.db.update_run(ctx.run_id, status=status, finished_at=time.time(), error=error)
            self.active = None
            self.progress.pop(ctx.run_id, None)
            self.emit({"type": "status", "run_id": ctx.run_id, "status": status, "error": error})

    def _sample_loop(self, ctx: RunContext) -> None:
        sampler = Sampler()
        guard = Guardrails(self.cfg.busy_util_pct, self.cfg.low_clock_mhz, self.cfg.thermal_warn_c,
                           self.cfg.guardrail_sustain_s)
        abort_on = ctx.spec.get("abort_on") or []
        batch: list[dict] = []
        last_flush = time.time()
        while not ctx.abort.is_set():
            t0 = time.time()
            s = sampler.sample()
            s["gpus"] = {k: v for k, v in s["gpus"].items() if int(k) in ctx.gpus}
            s["server"] = dict(ctx.server_live)
            batch.append(s)
            self.emit({"type": "telemetry", "run_id": ctx.run_id, "sample": s})
            for a in guard.check(s):
                level = "info" if a.kind == "recovered" else "warn"
                ctx.event(level, a.kind, a.message, {"gpu": a.gpu, "value": a.value})
                if a.kind in abort_on:
                    ctx.abort_reason = f"guardrail: {a.message}"
                    ctx.abort.set()
            if time.time() - last_flush >= 5:
                self.db.add_telemetry(ctx.run_id, batch)
                ctx.flush_logs()
                batch, last_flush = [], time.time()
            time.sleep(max(0.2, self.cfg.sample_interval_s - (time.time() - t0)))
        if batch:
            self.db.add_telemetry(ctx.run_id, batch)
