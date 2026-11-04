"""Server: real llama-server requests, the way an agent or chat client uses it.

Fixed prompts at temperature 0 with a fixed seed and a fixed output length
(ignore_eos), so every machine generates the same number of tokens. MTP
draft acceptance still varies a little run to run; report medians.
"""

from __future__ import annotations

from pathlib import Path

from ..llserver import LlamaServer
from .base import Module, load_s

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

CODE_PROMPT = ("Write a complete, well-commented Python module that implements an LRU cache with "
               "TTL expiry, thread safety, statistics counters and a small CLI demo. Include type "
               "hints and unit tests.")


def prompts() -> dict[str, str]:
    prose = (PROMPTS_DIR / "prose.txt").read_text(encoding="utf-8")
    return {"prose": "Summarize the following article in detail, section by section:\n\n" + prose,
            "code": CODE_PROMPT}


class ServerModule(Module):
    name = "server"
    label = "Server (real requests)"
    description = "llama-server prompt processing and generation on a ~2k-token prose prompt and a code prompt."
    params = {
        "reps": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Runs"},
        "mtp": {"type": "multi", "default": ["off", "on"], "choices": ["off", "on"], "label": "MTP"},
        "kv": {"type": "choice", "default": "q4_0", "choices": ["f16", "q8_0", "q4_0"], "label": "KV cache type"},
        "max_tokens": {"type": "int", "default": 512, "min": 64, "max": 4096, "label": "Tokens generated"},
        "ctx": {"type": "int", "default": 32768, "min": 4096, "max": 262144, "label": "Context size"},
    }

    def check(self, p, model, engine, snapshot):
        probs = []
        if "on" in p["mtp"] and model is not None and not model.get("mtp"):
            probs.append("MTP 'on' needs a model with MTP heads")
        if "on" in p["mtp"] and not engine.get("mtp_args"):
            probs.append(f"engine {engine['name']} has no MTP support")
        return probs

    def estimate_s(self, p, model_gb):
        return int(len(p["mtp"]) * (load_s(model_gb) + 30 + p["reps"] * 2 * (10 + p["max_tokens"] / 25)))

    def run(self, ctx, p):
        P = prompts()
        steps = len(p["mtp"]) * p["reps"]
        step = 0
        for mtp in p["mtp"]:
            ctx.cool_down()
            with LlamaServer(ctx, ["-ctk", p["kv"], "-ctv", p["kv"]], p["ctx"], mtp=(mtp == "on")) as srv:
                srv.chat({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16,
                          "cache_prompt": False})  # warmup, discarded
                for rep in range(p["reps"]):
                    step += 1
                    ctx.progress(self.name, step, steps, f"MTP {mtp}, run {rep + 1}")
                    if rep:
                        ctx.cool_down()
                    for pname, text in P.items():
                        if ctx.abort.is_set():
                            return
                        j = srv.chat({"messages": [{"role": "user", "content": text}],
                                      "max_tokens": p["max_tokens"], "temperature": 0, "seed": 42,
                                      "ignore_eos": True, "cache_prompt": False,
                                      "chat_template_kwargs": {"enable_thinking": False}})
                        t = j.get("timings") or {}
                        var = f"mtp={mtp}"
                        ctx.result(self.name, rep, var, f"{pname}.pp", t.get("prompt_per_second"), "t/s",
                                   {"tokens": t.get("prompt_n")})
                        ctx.result(self.name, rep, var, f"{pname}.tg", t.get("predicted_per_second"), "t/s",
                                   {"tokens": t.get("predicted_n")})
                        if t.get("draft_n"):
                            ctx.result(self.name, rep, var, f"{pname}.draft_accept",
                                       round(t["draft_n_accepted"] / t["draft_n"], 4), "",
                                       {"draft_n": t["draft_n"], "accepted": t["draft_n_accepted"]})
