"""Agent: the Hermes tool-use battery against the run's own llama-server.

Tasks, sandboxes and pass/fail checks come from hermes-agent's
scripts/toolperf_abeval/ab_eval.py (MIT, Nous Research), imported as-is so
grading stays identical to the published harness. Each task is a trap for a
specific tool-use mistake. The kit drives `hermes chat` itself instead of
ab_eval's runner: no bwrap (the container is the sandbox), agent processes
run as an unprivileged user, and each task-rep is tracked and timed here.

Pass rate mostly measures the model; wall time is what changes with hardware.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import time
from pathlib import Path

import yaml

from ..llserver import LlamaServer
from ..procs import Proc
from .base import Module, load_s

NO_TOOL_TASK = ("Without using any tool, calculate 17 multiplied by 19. Reply with exactly "
                "RESULT= followed by the integer and nothing else.")

# err_big_file_read hit the 600 s cap on ~30% of past runs and no_tool_required
# is a sanity check, so neither is in the default set.
DEFAULT_OFF = {"err_big_file_read", "no_tool_required"}


def load_abeval(hermes_source: Path):
    path = Path(hermes_source) / "scripts" / "toolperf_abeval" / "ab_eval.py"
    spec = importlib.util.spec_from_file_location("benchkit_abeval", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.TASKS["no_tool_required"] = NO_TOOL_TASK
    mod.SUCCESS["no_tool_required"] = lambda text, work: text.strip() == "RESULT=323"
    return mod


def task_names(hermes_source: Path) -> list[str]:
    try:
        return list(load_abeval(hermes_source).TASKS)
    except (OSError, ImportError, AttributeError):
        return []


def hermes_config(base_url: str, model_label: str, ctx_size: int, max_tokens: int, timeout_s: int) -> dict:
    return {
        "model": {"default": model_label, "provider": "custom", "base_url": base_url,
                  "api_key": "local-benchmark-only", "context_length": ctx_size, "max_tokens": max_tokens},
        "providers": {"custom": {"request_timeout_seconds": timeout_s}},
        "compression": {"enabled": False},
        "agent": {"reasoning_effort": "medium", "tool_use_enforcement": "auto"},
        "terminal": {"backend": "local"},
        "approvals": {"single_query_mode": "approve"},
        # The original harness never had the tirith binary, so it ran on pattern
        # matching only. Disabling it is the same behaviour without the stdout
        # warning (which breaks exact-match grading) or a download attempt.
        "security": {"tirith_enabled": False},
    }


def _chown_tree(path: Path, uid: int) -> None:
    if os.geteuid() != 0:
        return
    for root, dirs, files in os.walk(path):
        for n in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
            os.lchown(n, uid, uid)


class AgentModule(Module):
    name = "agent"
    label = "Agent battery (Hermes)"
    description = "Hermes agent tool-use trap tasks; reports pass rate and wall time per task."

    def __init__(self, hermes_source: Path | None = None):
        src = hermes_source or Path(os.environ.get("BENCHKIT_HERMES_SOURCE", "/opt/hermes-agent"))
        names = task_names(src) or ["err_python_env", "err_replay_patch", "err_ambiguous_edit",
                                    "err_case_search", "err_hidden_search", "err_big_output",
                                    "err_multi_dir", "err_inline_script", "err_big_file_read",
                                    "no_tool_required"]
        if "no_tool_required" not in names:
            names.append("no_tool_required")
        self.params = {
            "reps": {"type": "int", "default": 1, "min": 1, "max": 10, "label": "Runs per task"},
            "tasks": {"type": "multi", "default": [n for n in names if n not in DEFAULT_OFF],
                      "choices": names, "label": "Tasks"},
            "task_timeout_s": {"type": "int", "default": 600, "min": 60, "max": 3600, "label": "Per-task time cap (s)"},
            "max_tokens": {"type": "int", "default": 8192, "min": 512, "max": 32768, "label": "Max tokens per reply"},
            "ctx": {"type": "int", "default": 65536, "min": 65536, "max": 262144, "label": "Context size"},
            "mtp": {"type": "bool", "default": True, "label": "Use MTP if available"},
            "kv": {"type": "choice", "default": "q8_0", "choices": ["f16", "q8_0", "q4_0"], "label": "KV cache type"},
        }

    def estimate_s(self, p, model_gb):
        return int(load_s(model_gb) + p["reps"] * len(p["tasks"]) * 120)

    def run(self, ctx, p):
        ab = load_abeval(ctx.cfg.hermes_source)
        use_mtp = bool(p["mtp"] and ctx.model_meta and ctx.model_meta.get("mtp") and ctx.engine.get("mtp_args"))
        base = ctx.cfg.data_dir / "agent" / ctx.run_id
        home = base / "home"
        home.mkdir(parents=True, exist_ok=True)
        model_label = f"benchkit-{ctx.run_id}"
        ctx.cool_down()
        with LlamaServer(ctx, ["-ctk", p["kv"], "-ctv", p["kv"]], p["ctx"], mtp=use_mtp,
                         record=(self.name, f"mtp={'on' if use_mtp else 'off'}")) as srv:
            (home / "config.yaml").write_text(yaml.safe_dump(hermes_config(
                srv.base + "/v1", model_label, p["ctx"], p["max_tokens"], p["task_timeout_s"])))
            steps = p["reps"] * len(p["tasks"])
            step = 0
            for rep in range(p["reps"]):
                for task in p["tasks"]:
                    if ctx.abort.is_set():
                        return
                    step += 1
                    ctx.progress(self.name, step, steps, f"{task} run {rep + 1}")
                    work = base / "work" / f"{task}-r{rep}"
                    if work.exists():
                        shutil.rmtree(work)
                    work.mkdir(parents=True)
                    ab.make_sandbox(work)
                    _chown_tree(base, ctx.cfg.agent_uid)
                    query = ab.TASKS[task].replace("{WORK}", str(work))
                    env = {"HOME": str(home), "HERMES_HOME": str(home), "PYTHONPATH": str(ctx.cfg.hermes_source),
                           "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                           "OPENAI_API_KEY": "local-benchmark-only"}
                    argv = [ctx.cfg.hermes_python, "-m", "hermes_cli.main", "chat", "--query", query,
                            "--quiet", "--max-turns", "30", "--accept-hooks", "--model", model_label]
                    t0 = time.time()
                    # graded on stdout only, like ab_eval (stderr carries warnings)
                    proc = Proc(argv, env=env, cwd=str(work), preexec_uid=ctx.cfg.agent_uid, split_stderr=True,
                                on_line=lambda l, tk=task: ctx.log("agent", f"[{tk}] {l}"))
                    rc = proc.wait(timeout=p["task_timeout_s"], abort=ctx.abort)
                    wall = round(time.time() - t0, 1)
                    if ctx.abort.is_set():
                        return
                    out = "\n".join(proc.lines).strip()
                    if rc is None:
                        out, rc = f"MODEL_TIMEOUT_{p['task_timeout_s']}S", -9
                    elif rc != 0 and not out:
                        ctx.event("warn", "agent_infra", f"{task} run {rep + 1}: hermes exited {rc} with no output")
                    tail = "\n".join(out.splitlines()[-12:])
                    try:
                        ok = bool(ab.SUCCESS[task](tail, work))
                    except Exception:  # noqa: BLE001
                        ok = False
                    ctx.agent_task(task, rep, wall, rc, ok, tail)
        # summary rows so agent numbers sit beside the other modules in exports
        rows = ctx.db.get_agent_tasks(ctx.run_id)
        for task in p["tasks"]:
            tr = [r for r in rows if r["task"] == task]
            if tr:
                walls = sorted(r["wall_s"] for r in tr)
                ctx.result(self.name, 0, "all", f"{task}.pass_rate", sum(r["success"] for r in tr) / len(tr), "")
                ctx.result(self.name, 0, "all", f"{task}.median_wall", walls[len(walls) // 2], "s")
        if rows:
            ctx.result(self.name, 0, "all", "total.pass_rate", sum(r["success"] for r in rows) / len(rows), "")
            ctx.result(self.name, 0, "all", "total.wall", round(sum(r["wall_s"] for r in rows), 1), "s")
