"""Parsers for llama-server log lines (the print_timing family).

Real sample lines (from runs/*/hermes-eval/server.log):

    1.40.320.240 I slot print_timing: id  0 | task 0 | prompt eval time =    1275.81 ms /   425 tokens (    3.00 ms per token,   333.12 tokens per second)
    1.40.320.254 I slot print_timing: id  0 | task 0 |        eval time =   14358.06 ms /   646 tokens (   22.26 ms per token,    44.92 tokens per second)
    2.32.041.818 I slot print_timing: id  0 | task 0 | draft acceptance = 0.63137 (  161 accepted /   255 generated), mean len =  2.89

These may be prefixed by syslog-style headers
("Aug 24 00:16:36 bowie llama-server[1501336]: ") or written directly by the
server. We anchor on the "slot print_timing:" token so either form parses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# prompt eval time = 1275.81 ms / 425 tokens ( 3.00 ms per token, 333.12 tokens per second)
PROMPT_RE = re.compile(
    r"print_timing:\s+id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)\s*\|\s*"
    r"prompt eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens\s*\("
    r"\s*(?P<ms_per_token>[\d.]+)\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second"
)

#        eval time = 14358.06 ms / 646 tokens (22.26 ms per token, 44.92 tokens per second)
EVAL_RE = re.compile(
    r"print_timing:\s+id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)\s*\|\s*"
    r"eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens\s*\("
    r"\s*(?P<ms_per_token>[\d.]+)\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second"
)

# draft acceptance = 0.63137 (161 accepted / 255 generated), mean len = 2.89
DRAFT_RE = re.compile(
    r"print_timing:\s+id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>\d+)\s*\|\s*"
    r"draft acceptance\s*=\s*(?P<accept>[\d.]+)\s*\(\s*(?P<accepted>\d+)\s*accepted\s*/\s*"
    r"(?P<generated>\d+)\s*generated\)\s*,\s*mean len\s*=\s*(?P<mean_len>[\d.]+)"
)


@dataclass
class TimingEvent:
    slot: int
    task: int
    kind: str            # "prompt" | "eval" | "draft"
    tokens: int = 0
    ms: float = 0.0
    ms_per_token: float = 0.0
    tokens_per_s: float = 0.0
    # draft-only
    acceptance: float | None = None
    accepted: int = 0
    generated: int = 0
    mean_len: float = 0.0


def parse_print_timing(line: str) -> TimingEvent | None:
    line = line.rstrip("\n")
    m = PROMPT_RE.search(line)
    if m:
        return TimingEvent(slot=int(m["slot"]), task=int(m["task"]), kind="prompt",
                           tokens=int(m["tokens"]), ms=float(m["ms"]),
                           ms_per_token=float(m["ms_per_token"]),
                           tokens_per_s=float(m["tps"]))
    m = DRAFT_RE.search(line)
    if m:
        return TimingEvent(slot=int(m["slot"]), task=int(m["task"]), kind="draft",
                           acceptance=float(m["accept"]), accepted=int(m["accepted"]),
                           generated=int(m["generated"]), mean_len=float(m["mean_len"]))
    m = EVAL_RE.search(line)
    if m:
        return TimingEvent(slot=int(m["slot"]), task=int(m["task"]), kind="eval",
                           tokens=int(m["tokens"]), ms=float(m["ms"]),
                           ms_per_token=float(m["ms_per_token"]),
                           tokens_per_s=float(m["tps"]))
    return None


def parse_log_lines(lines) -> Iterator[TimingEvent]:
    """Yield a TimingEvent for each line that carries one; skip the rest."""
    for line in lines:
        ev = parse_print_timing(line)
        if ev is not None:
            yield ev
