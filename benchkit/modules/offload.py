"""Offload: MoE expert weights in system RAM (-ot exps=CPU), attention on GPU.

With the experts on the CPU side, decode speed is bound by system RAM
bandwidth, so this is the module that separates DDR3 from DDR4 boxes. The
snapshot records RAM type, speed and populated slots next to the numbers.
"""

from __future__ import annotations

from ..engines import env_for, split_args
from ..procs import Proc
from .base import Module, load_s
from .speed import metric_name, parse_bench_jsonl


class OffloadModule(Module):
    name = "offload"
    label = "RAM offload (MoE experts on CPU)"
    description = "llama-bench with MoE expert layers in system RAM; shows RAM bandwidth (DDR3 vs DDR4)."
    params = {
        "reps": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Runs"},
        "pp": {"type": "int", "default": 512, "min": 16, "max": 8192, "label": "Prompt tokens"},
        "tg": {"type": "int", "default": 128, "min": 16, "max": 2048, "label": "Generated tokens"},
        "threads": {"type": "int", "default": 0, "min": 0, "max": 256, "label": "CPU threads (0 = physical cores)"},
        "samples": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "llama-bench -r per run"},
    }

    def check(self, p, model, engine, snapshot):
        probs = []
        if model is not None and model.get("kind") != "moe":
            probs.append("offload needs a MoE model (for example Qwen3.6-35B-A3B)")
        total = (snapshot.get("memory") or {}).get("total_gb")
        if model is not None and total and model["size"] / 1e9 > total * 0.9:
            probs.append(f"model is {model['size'] / 1e9:.0f} GB but the box has {total:.0f} GB RAM")
        return probs

    def estimate_s(self, p, model_gb):
        return int(p["reps"] * (load_s(model_gb) + p["samples"] * 60))

    def run(self, ctx, p):
        e = ctx.engine
        threads = p["threads"] or (ctx.snapshot.get("cpu") or {}).get("cores") or 4
        for rep in range(p["reps"]):
            ctx.progress(self.name, rep + 1, p["reps"], f"run {rep + 1}")
            ctx.cool_down()
            argv = [f"{e['bin_dir']}/llama-bench", "-m", str(ctx.model_path), "-ngl", "99", "-fa", "1",
                    "-ot", "exps=CPU", "-t", str(threads), "-p", str(p["pp"]), "-n", str(p["tg"]),
                    "-r", str(p["samples"]), "-o", "jsonl", *split_args(e, ctx.gpus)]
            ctx.log("runner", "llama-bench: " + " ".join(argv))
            proc = Proc(argv, env=env_for(e, ctx.gpus), on_line=lambda l: ctx.log("bench", l))
            rc = proc.wait(timeout=3600, abort=ctx.abort)
            if ctx.abort.is_set():
                return
            rows = parse_bench_jsonl(proc.lines)
            if rc != 0 or not rows:
                ctx.event("error", "bench_failed", f"offload llama-bench failed (rc={rc}), run {rep + 1}",
                          {"tail": proc.lines[-15:]})
                continue
            mem = ctx.snapshot.get("memory") or {}
            for d in rows:
                ctx.result(self.name, rep, f"threads={threads}", metric_name(d), d["avg_ts"], "t/s",
                           {"stddev": d.get("stddev_ts"), "ram_type": mem.get("type"),
                            "ram_speed": mem.get("configured_speed")})
