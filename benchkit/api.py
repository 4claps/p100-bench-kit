"""HTTP API. Every route except login, health and the static UI needs auth:
either the session cookie set by /api/login or `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import export
from .modules import MODULE_ORDER, MODULES
from .telemetry import DEFAULT_CHARTS, METRICS

router = APIRouter()


def S(request: Request):
    return request.app.state


async def body(request: Request) -> dict:
    try:
        b = await request.json()
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    if not isinstance(b, dict):
        raise HTTPException(400, "body must be a JSON object")
    return b


# ---------------- auth ----------------
@router.post("/login")
async def login(request: Request):
    b = await body(request)
    st = S(request)
    if not secrets.compare_digest(str(b.get("token", "")).encode(), st.cfg.token.encode()):
        await asyncio.sleep(1)  # slow down guessing
        raise HTTPException(401, "wrong token")
    sid = st.sessions.create()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("benchkit_session", sid, httponly=True, samesite="strict", max_age=30 * 86400,
                    secure=request.url.scheme == "https")
    return resp


@router.post("/logout")
async def logout(request: Request):
    S(request).sessions.drop(request.cookies.get("benchkit_session", ""))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("benchkit_session")
    return resp


# ---------------- system ----------------
@router.get("/system")
async def system(request: Request, refresh: int = 0):
    st = S(request)
    snap = await asyncio.to_thread(st.runner.snapshot, bool(refresh))
    return {
        "kit_version": st.kit_version,
        "snapshot": snap,
        "engines": [{k: e.get(k) for k in ("name", "label", "commit", "cuda_archs", "multi_gpu_split",
                                            "available", "notes")} | {"mtp": bool(e.get("mtp_args"))}
                    for e in st.runner.engines],
        "modules": [MODULES[n].describe() for n in MODULE_ORDER],
        "metrics": METRICS,
        "active_run": st.runner.active.run_id if st.runner.active else None,
        "guardrails": {"low_clock_mhz": st.cfg.low_clock_mhz, "thermal_warn_c": st.cfg.thermal_warn_c,
                       "sustain_s": st.cfg.guardrail_sustain_s, "cooldown_c": st.cfg.cooldown_c},
    }


# ---------------- models ----------------
@router.get("/models")
async def models(request: Request):
    return await asyncio.to_thread(S(request).store.list)


@router.post("/models/{model_id}/download")
async def model_download(request: Request, model_id: str):
    try:
        return S(request).store.download(model_id).view()
    except KeyError:
        raise HTTPException(404, "unknown model")
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/models/{model_id}/verify")
async def model_verify(request: Request, model_id: str):
    try:
        return S(request).store.verify(model_id).view()
    except KeyError:
        raise HTTPException(404, "unknown model")
    except RuntimeError as e:
        raise HTTPException(409, str(e))


# ---------------- runs ----------------
@router.post("/runs")
async def create_run(request: Request):
    spec = await body(request)
    try:
        return await asyncio.to_thread(S(request).runner.create, spec)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/runs/{run_id}/start")
async def start_run(request: Request, run_id: str):
    try:
        await asyncio.to_thread(S(request).runner.start, run_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@router.post("/runs/{run_id}/abort")
async def abort_run(request: Request, run_id: str):
    try:
        S(request).runner.abort(run_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@router.get("/runs")
async def list_runs(request: Request):
    st = S(request)
    out = []
    for r in st.db.list_runs():
        spec = r["spec"]
        out.append({"id": r["id"], "status": r["status"], "created_at": r["created_at"],
                    "finished_at": r["finished_at"], "label": spec.get("label", ""), "model": spec.get("model"),
                    "engine": spec.get("engine"), "gpus": spec.get("gpus"),
                    "modules": list((spec.get("modules") or {}).keys()), "invalid": r["invalid"],
                    "tags": r["tags"], "comparable": spec.get("comparable"),
                    "hardware": export.hardware_line(r["snapshot"] if r["status"] != "imported"
                                                     else spec["bundle"].get("snapshot"))})
    return out


def _run_or_404(st, run_id: str) -> dict:
    r = st.db.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    st = S(request)
    r = _run_or_404(st, run_id)
    if r["status"] == "imported":
        b = r["spec"]["bundle"]
        return {"run": r | {"snapshot": b.get("snapshot")}, "results": b.get("results", []),
                "summary": b.get("summary", {}), "agent_tasks": b.get("agent_tasks", []),
                "events": b.get("events", []), "progress": None, "imported": True}
    results = st.db.get_results(run_id)
    return {"run": r, "results": results, "summary": export.summarize_results(results),
            "agent_tasks": st.db.get_agent_tasks(run_id), "events": st.db.get_events(run_id),
            "progress": st.runner.progress.get(run_id), "imported": False}


@router.patch("/runs/{run_id}")
async def patch_run(request: Request, run_id: str):
    st = S(request)
    _run_or_404(st, run_id)
    b = await body(request)
    fields = {}
    if "note" in b:
        fields["note"] = str(b["note"])[:4000]
    if "tags" in b:
        fields["tags"] = ",".join(str(t).strip()[:40] for t in b["tags"] if str(t).strip())
    if "invalid" in b:
        fields["invalid"] = int(bool(b["invalid"]))
        fields["invalid_reason"] = str(b.get("invalid_reason") or "")[:500] if b["invalid"] else None
        if b["invalid"] and not fields["invalid_reason"]:
            raise HTTPException(400, "give a reason when marking a run invalid")
    st.db.update_run(run_id, **fields)
    return {"ok": True}


@router.delete("/runs/{run_id}")
async def delete_run(request: Request, run_id: str):
    st = S(request)
    if st.runner.active and st.runner.active.run_id == run_id:
        raise HTTPException(409, "abort the run before deleting it")
    _run_or_404(st, run_id)
    st.db.delete_run(run_id)
    return {"ok": True}


@router.get("/runs/{run_id}/telemetry")
async def run_telemetry(request: Request, run_id: str, max_points: int = 1500):
    st = S(request)
    _run_or_404(st, run_id)
    return await asyncio.to_thread(st.db.get_telemetry, run_id, min(max(max_points, 10), 20000))


@router.get("/runs/{run_id}/logs")
async def run_logs(request: Request, run_id: str, stream: str | None = None, tail: int = 500):
    st = S(request)
    _run_or_404(st, run_id)
    return st.db.get_logs(run_id, stream, min(max(tail, 1), 5000))


@router.get("/runs/{run_id}/export.json")
async def export_json(request: Request, run_id: str):
    st = S(request)
    _run_or_404(st, run_id)
    b = await asyncio.to_thread(export.bundle, st.db, run_id, st.kit_version)
    return JSONResponse(b, headers={"Content-Disposition": f'attachment; filename="benchkit-{run_id}.json"'})


@router.get("/runs/{run_id}/export.md")
async def export_md(request: Request, run_id: str):
    st = S(request)
    _run_or_404(st, run_id)
    b = await asyncio.to_thread(export.bundle, st.db, run_id, st.kit_version)
    return PlainTextResponse(export.markdown(b))


@router.post("/import")
async def import_bundle(request: Request):
    """Load someone else's result bundle so it can be compared with local runs."""
    st = S(request)
    b = await body(request)
    try:
        export.validate_bundle(b)
    except ValueError as e:
        raise HTTPException(400, str(e))
    rid = "imp-" + str(b["run"].get("id", "x"))[:40] + "-" + secrets.token_hex(2)
    spec = {"imported": True, "bundle": b, "label": b["run"].get("label") or "imported",
            "model": b["spec"].get("model"), "engine": b["spec"].get("engine"), "gpus": b["spec"].get("gpus"),
            "modules": b["spec"].get("modules") or {}, "comparable": b["spec"].get("comparable")}
    st.db.create_run(rid, spec, 0, status="imported")
    return {"id": rid}


@router.get("/compare")
async def compare(request: Request, ids: str):
    st = S(request)
    out = []
    for rid in [i for i in ids.split(",") if i][:12]:
        _run_or_404(st, rid)
        b = await asyncio.to_thread(export.bundle, st.db, rid, st.kit_version)
        out.append({"id": rid, "label": b["run"].get("label"), "hardware": export.hardware_line(b.get("snapshot")),
                    "engine": b["spec"].get("engine"), "model": b["spec"].get("model"), "gpus": b["spec"].get("gpus"),
                    "invalid": b["run"].get("invalid"), "comparable": b["spec"].get("comparable"),
                    "summary": b["summary"]})
    return out


# ---------------- layouts / presets ----------------
@router.get("/layouts/{name}")
async def get_layout(request: Request, name: str):
    return {"charts": S(request).db.get_layout(name[:40]) or DEFAULT_CHARTS}


@router.put("/layouts/{name}")
async def put_layout(request: Request, name: str):
    b = await body(request)
    valid = {m["id"] for m in METRICS}
    charts = [c for c in b.get("charts", []) if c in valid][:24]
    S(request).db.put_layout(name[:40], charts)
    return {"charts": charts}


@router.get("/presets")
async def presets(request: Request):
    return S(request).db.list_presets()


@router.put("/presets/{name}")
async def put_preset(request: Request, name: str):
    b = await body(request)
    S(request).db.put_preset(name[:60], b.get("spec") or {})
    return {"ok": True}


@router.delete("/presets/{name}")
async def delete_preset(request: Request, name: str):
    S(request).db.delete_preset(name)
    return {"ok": True}


# ---------------- live stream ----------------
@router.get("/stream")
async def stream(request: Request):
    runner = S(request).runner
    q = runner.subscribe()

    async def gen():
        try:
            yield "retry: 3000\n\n"
            last = time.time()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except asyncio.TimeoutError:
                    pass
                if time.time() - last > 15:
                    yield ": ping\n\n"
                    last = time.time()
        finally:
            runner.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
