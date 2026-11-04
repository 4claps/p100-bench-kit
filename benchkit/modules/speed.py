"""Speed: llama-bench prompt processing and decode at several context depths.

One llama-bench call per (rep, KV type), each preceded by a cool-down, so
reps are spread over time like the Houston A/B protocol instead of being
three back-to-back samples of the same warm state.
"""

from __future__ import annotations

import json

from ..engines import env_for, split_args
from ..procs import Proc
from .base import Module, load_s

KV_TYPES = ["f16", "q8_0", "q4_0"]


def parse_bench_jsonl(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "avg_ts" in d:
            out.append(d)
    return out


def metric_name(d: dict) -> str:
    depth = d.get("n_depth", 0)
    if d.get("n_gen", 0) > 0 and d.get("n_prompt", 0) == 0:
        return f"tg{d['n_gen']}@{depth}"
    return f"pp{d['n_prompt']}@{depth}"


class SpeedModule(Module):
    name = "speed"
    label = "Speed (llama-bench)"
    description = "Prompt processing and decode tokens/s at empty and filled context."
    params = {
        "reps": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Runs"},
        "kv": {"type": "multi", "default": ["q4_0"], "choices": KV_TYPES, "label": "KV cache types"},
        "depths": {"type": "ints", "default": [0, 16384], "label": "Context depths"},
        "pp": {"type": "int", "default": 512, "min": 16, "max": 8192, "label": "Prompt tokens"},
        "tg": {"type": "int", "default": 128, "min": 16, "max": 2048, "label": "Generated tokens"},
        "samples": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "llama-bench -r per run"},
    }

    def estimate_s(self, p: dict, model_gb: float) -> int:
        per = load_s(model_gb) + p["samples"] * sum(15 + d / 300 for d in p["depths"])
        return int(p["reps"] * len(p["kv"]) * per)

    def run(self, ctx, p: dict) -> None:
        e = ctx.engine
        steps = p["reps"] * len(p["kv"])
        step = 0
        for rep in range(p["reps"]):
            for kv in p["kv"]:
                step += 1
                ctx.progress(self.name, step, steps, f"run {rep + 1}, KV {kv}")
                ctx.cool_down()
                argv = [f"{e['bin_dir']}/llama-bench", "-m", str(ctx.model_path), "-ngl", "99",
                        "-fa", "1", "-ctk", kv, "-ctv", kv, "-p", str(p["pp"]), "-n", str(p["tg"]),
                        "-d", ",".join(str(d) for d in p["depths"]), "-r", str(p["samples"]),
                        "-o", "jsonl", *split_args(e, ctx.gpus)]
                ctx.log("runner", "llama-bench: " + " ".join(argv))
                proc = Proc(argv, env=env_for(e, ctx.gpus), on_line=lambda l: ctx.log("bench", l))
                rc = proc.wait(timeout=3600, abort=ctx.abort)
                if ctx.abort.is_set():
                    return
                rows = parse_bench_jsonl(proc.lines)
                if rc != 0 or not rows:
                    ctx.event("error", "bench_failed", f"llama-bench failed (rc={rc}) for KV {kv}, run {rep + 1}",
                              {"tail": proc.lines[-15:]})
                    continue
                for d in rows:
                    ctx.result(self.name, rep, f"kv={kv}", metric_name(d), d["avg_ts"], "t/s",
                               {"stddev": d.get("stddev_ts"), "samples": d.get("samples_ts"),
                                "build": d.get("build_commit")})
