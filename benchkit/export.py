"""Shareable result bundle (one JSON per run) and a Markdown write-up.

Bundle schema 1:
  kit_version, exported_at, run{id, status, label, created/started/finished,
  invalid, invalid_reason, note, tags}, spec, snapshot, results[],
  summary{module: {variant: {metric: {median, min, max, n, unit}}}},
  agent_tasks[], telemetry_summary{}, events[] (warn/error only).
The snapshot is anonymized: GPU UUIDs and PCI bus ids are replaced by slot
labels; no hostnames, IPs or usernames are collected in the first place.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict

from .db import DB
from .telemetry import summarize

BUNDLE_SCHEMA = 1


def anonymize_snapshot(snap: dict | None) -> dict | None:
    if not snap:
        return snap
    snap = {**snap, "gpus": [dict(g) for g in snap.get("gpus") or []]}
    for g in snap["gpus"]:
        g.pop("uuid", None)
        g["pci_bus_id"] = f"slot{g.get('index')}"
    mem = dict(snap.get("memory") or {})
    mem["dimms"] = [{k: d.get(k) for k in ("size", "type", "speed", "configured_speed")}
                    for d in mem.get("dimms") or []]
    snap["memory"] = mem
    topo = snap.get("gpu_topology")
    if topo:
        snap["gpu_topology"] = "\n".join(l for l in topo.splitlines() if not l.lstrip().startswith("NIC"))
    return snap


def summarize_results(results: list[dict]) -> dict:
    groups: dict = defaultdict(list)
    units = {}
    for r in results:
        if r["value"] is None:
            continue
        key = (r["module"], r["variant"], r["metric"])
        groups[key].append(r["value"])
        units[key] = r["unit"]
    out: dict = {}
    for (mod, var, met), vals in sorted(groups.items()):
        out.setdefault(mod, {}).setdefault(var, {})[met] = {
            "median": round(statistics.median(vals), 3), "min": min(vals), "max": max(vals),
            "n": len(vals), "unit": units[(mod, var, met)]}
    return out


def bundle(db: DB, run_id: str, kit_version: str) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise KeyError(run_id)
    if run["status"] == "imported":
        return run["spec"]["bundle"]
    results = db.get_results(run_id)
    return {
        "bundle_schema": BUNDLE_SCHEMA,
        "kit_version": kit_version,
        "exported_at": time.time(),
        "run": {k: run.get(k) for k in ("id", "status", "created_at", "started_at", "finished_at",
                                         "invalid", "invalid_reason", "note", "tags", "error")}
               | {"label": run["spec"].get("label", "")},
        "spec": run["spec"],
        "snapshot": anonymize_snapshot(run["snapshot"]),
        "results": [{k: r[k] for k in ("module", "rep", "variant", "metric", "value", "unit")} for r in results],
        "summary": summarize_results(results),
        "agent_tasks": [{k: t[k] for k in ("task", "rep", "wall_s", "exit_code", "success")}
                        for t in db.get_agent_tasks(run_id)],
        "telemetry_summary": summarize(db.get_telemetry(run_id, max_points=100000)),
        "events": [e for e in db.get_events(run_id) if e["level"] in ("warn", "error")],
    }


def validate_bundle(b: dict) -> None:
    if not isinstance(b, dict) or b.get("bundle_schema") != BUNDLE_SCHEMA:
        raise ValueError("not a benchkit result bundle (bundle_schema 1)")
    for k in ("run", "spec", "summary"):
        if k not in b:
            raise ValueError(f"bundle is missing {k!r}")


def hardware_line(snap: dict | None) -> str:
    if not snap:
        return "unknown hardware"
    gpus = snap.get("gpus") or []
    names: dict = defaultdict(int)
    for g in gpus:
        names[g.get("name")] += 1
    gtxt = ", ".join(f"{n}x {name}" for name, n in names.items()) or "no GPU"
    mem = snap.get("memory") or {}
    cpu = (snap.get("cpu") or {}).get("model") or "?"
    ram = f"{mem.get('total_gb')} GB {mem.get('type') or ''} {mem.get('configured_speed') or ''}".strip()
    return f"{gtxt}; {cpu}; {ram}"


def markdown(b: dict) -> str:
    run, spec = b["run"], b["spec"]
    lines = [f"# Bench run {run['id']}", "",
             f"- Hardware: {hardware_line(b.get('snapshot'))}",
             f"- Engine: {spec.get('engine')}; model: {spec.get('model')}; GPUs: {spec.get('gpus')}",
             f"- Status: {run.get('status')}" + (" (marked invalid: " + str(run.get('invalid_reason')) + ")"
                                                 if run.get("invalid") else ""),
             f"- Comparable: {'yes' if spec.get('comparable') else 'no (custom model)'}", ""]
    for mod, variants in b["summary"].items():
        lines += [f"## {mod}", "", "| variant | metric | median | min | max | n |", "|---|---|---|---|---|---|"]
        for var, mets in variants.items():
            for met, s in mets.items():
                u = f" {s['unit']}" if s["unit"] else ""
                lines.append(f"| {var} | {met} | {s['median']}{u} | {s['min']:.4g} | {s['max']:.4g} | {s['n']} |")
        lines.append("")
    tel = b.get("telemetry_summary") or {}
    keys = [k for k in tel if k.endswith((".temp_c", ".sm_mhz", ".power_w"))]
    if keys:
        lines += ["## Telemetry", "", "| metric | min | mean | max |", "|---|---|---|---|"]
        lines += [f"| {k} | {tel[k]['min']} | {tel[k]['mean']} | {tel[k]['max']} |" for k in keys]
        lines.append("")
    if b.get("events"):
        lines += ["## Warnings", ""] + [f"- {e['kind']}: {e['message']}" for e in b["events"]] + [""]
    return "\n".join(lines)
