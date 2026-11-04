"""Live telemetry sampled locally: nvidia-smi for GPUs, /proc and /sys for
CPU, memory, fans. One sample per interval, shaped as

    {"t": epoch, "gpus": {"0": {"temp_c": .., "sm_mhz": ..}, ...},
     "cpu": {"util_pct": .., "mhz": .., "temp_c": .., "power_w": ..},
     "mem": {"avail_gb": ..}, "fans": {"hwmon2/fan1": rpm, ...}}

METRICS is the catalog the UI offers for charts. Adding a metric means
adding one entry here and filling its key in sample(); no UI change.
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path

from .hw import run_cmd, _num

# id, label, unit, scope ("gpu" = one series per GPU), key inside the scope
METRICS = [
    {"id": "gpu.temp_c", "label": "GPU temperature", "unit": "°C", "scope": "gpu", "key": "temp_c"},
    {"id": "gpu.sm_mhz", "label": "GPU SM clock", "unit": "MHz", "scope": "gpu", "key": "sm_mhz"},
    {"id": "gpu.mem_mhz", "label": "GPU memory clock", "unit": "MHz", "scope": "gpu", "key": "mem_mhz"},
    {"id": "gpu.power_w", "label": "GPU power", "unit": "W", "scope": "gpu", "key": "power_w"},
    {"id": "gpu.util_pct", "label": "GPU utilization", "unit": "%", "scope": "gpu", "key": "util_pct"},
    {"id": "gpu.mem_util_pct", "label": "GPU memory-bus utilization", "unit": "%", "scope": "gpu", "key": "mem_util_pct"},
    {"id": "gpu.vram_mb", "label": "GPU VRAM used", "unit": "MiB", "scope": "gpu", "key": "vram_mb"},
    {"id": "gpu.fan_pct", "label": "GPU fan", "unit": "%", "scope": "gpu", "key": "fan_pct"},
    {"id": "gpu.pcie_gen", "label": "PCIe link gen", "unit": "", "scope": "gpu", "key": "pcie_gen"},
    {"id": "gpu.pcie_width", "label": "PCIe link width", "unit": "x", "scope": "gpu", "key": "pcie_width"},
    {"id": "cpu.util_pct", "label": "CPU utilization", "unit": "%", "scope": "cpu", "key": "util_pct"},
    {"id": "cpu.mhz", "label": "CPU clock (avg)", "unit": "MHz", "scope": "cpu", "key": "mhz"},
    {"id": "cpu.temp_c", "label": "CPU temperature", "unit": "°C", "scope": "cpu", "key": "temp_c"},
    {"id": "cpu.power_w", "label": "CPU package power", "unit": "W", "scope": "cpu", "key": "power_w"},
    {"id": "mem.avail_gb", "label": "RAM available", "unit": "GB", "scope": "mem", "key": "avail_gb"},
    {"id": "server.pp_tps", "label": "Prompt processing", "unit": "t/s", "scope": "server", "key": "pp_tps"},
    {"id": "server.tg_tps", "label": "Generation", "unit": "t/s", "scope": "server", "key": "tg_tps"},
    {"id": "server.draft_accept", "label": "MTP draft acceptance", "unit": "", "scope": "server", "key": "draft_accept"},
]

DEFAULT_CHARTS = ["gpu.temp_c", "gpu.sm_mhz", "gpu.power_w", "gpu.util_pct", "server.tg_tps"]

GPU_QUERY = [("index", None), ("temperature.gpu", "temp_c"), ("clocks.sm", "sm_mhz"),
             ("clocks.mem", "mem_mhz"), ("power.draw", "power_w"), ("utilization.gpu", "util_pct"),
             ("utilization.memory", "mem_util_pct"), ("memory.used", "vram_mb"),
             ("fan.speed", "fan_pct"), ("pcie.link.gen.current", "pcie_gen"),
             ("pcie.link.width.current", "pcie_width"), ("pstate", "pstate"),
             ("clocks_event_reasons.active", "throttle")]


def parse_gpu_sample(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_QUERY):
            continue
        g = {}
        for (_, key), v in zip(GPU_QUERY[1:], parts[1:]):
            g[key] = v if key in ("pstate", "throttle") else _num(v)
        out[parts[0]] = g
    return out


def parse_proc_stat(text: str) -> tuple[int, int] | None:
    """(busy, total) jiffies from the aggregate cpu line."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            vals = [int(x) for x in line.split()[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            return sum(vals) - idle, sum(vals)
    return None


class Sampler:
    """Stateful: CPU util and RAPL power need deltas between samples."""

    def __init__(self):
        self._stat = None
        self._rapl = None
        self._gpu_query = GPU_QUERY
        self._rapl_path = next(iter(glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj")), None)

    def _gpus(self) -> dict:
        fields = ",".join(f for f, _ in self._gpu_query)
        ok, out = run_cmd(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=10)
        if not ok and "clocks_event_reasons" in fields:
            # older drivers only know the pre-rename field
            self._gpu_query = [(("clocks_throttle_reasons.active" if f.startswith("clocks_event") else f), k)
                               for f, k in self._gpu_query]
            return self._gpus()
        return parse_gpu_sample(out) if ok else {}

    def _cpu(self) -> dict:
        cpu: dict = {}
        try:
            cur = parse_proc_stat(Path("/proc/stat").read_text())
        except OSError:
            cur = None
        if cur and self._stat and cur[1] > self._stat[1]:
            cpu["util_pct"] = round(100 * (cur[0] - self._stat[0]) / (cur[1] - self._stat[1]), 1)
        self._stat = cur
        try:
            mhz = [float(x) for x in re.findall(r"cpu MHz\s*:\s*([\d.]+)", Path("/proc/cpuinfo").read_text())]
            if mhz:
                cpu["mhz"] = round(sum(mhz) / len(mhz))
        except OSError:
            pass
        for name_file in glob.glob("/sys/class/hwmon/hwmon*/name"):
            try:
                if Path(name_file).read_text().strip() in ("coretemp", "k10temp", "zenpower"):
                    t = Path(os.path.dirname(name_file), "temp1_input").read_text()
                    cpu["temp_c"] = int(t) / 1000
                    break
            except OSError:
                continue
        if self._rapl_path:
            try:
                e, now = int(Path(self._rapl_path).read_text()), time.time()
                if self._rapl and e >= self._rapl[0]:
                    cpu["power_w"] = round((e - self._rapl[0]) / 1e6 / (now - self._rapl[1]), 1)
                self._rapl = (e, now)
            except (OSError, ValueError):
                self._rapl_path = None
        return cpu

    @staticmethod
    def _fans() -> dict:
        fans = {}
        for f in glob.glob("/sys/class/hwmon/hwmon*/fan*_input"):
            try:
                rpm = int(Path(f).read_text())
            except (OSError, ValueError):
                continue
            hw = os.path.basename(os.path.dirname(f))
            fans[f"{hw}/{os.path.basename(f)[:-6]}"] = rpm
        return fans

    def sample(self) -> dict:
        mem = {}
        try:
            m = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
            if m:
                mem["avail_gb"] = round(int(m.group(1)) / 1024 / 1024, 2)
        except OSError:
            pass
        return {"t": time.time(), "gpus": self._gpus(), "cpu": self._cpu(), "mem": mem, "fans": self._fans()}

    def max_gpu_temp(self, gpus: list[int] | None = None) -> float | None:
        temps = [g.get("temp_c") for i, g in self._gpus().items()
                 if (gpus is None or int(i) in gpus) and isinstance(g.get("temp_c"), (int, float))]
        return max(temps) if temps else None


def summarize(samples: list[dict]) -> dict:
    """min/max/mean per metric (per GPU for gpu metrics) for the export bundle."""
    acc: dict[str, list[float]] = {}
    for s in samples:
        for gi, g in (s.get("gpus") or {}).items():
            for k, v in g.items():
                if isinstance(v, (int, float)):
                    acc.setdefault(f"gpu{gi}.{k}", []).append(v)
        for scope in ("cpu", "mem", "server"):
            for k, v in (s.get(scope) or {}).items():
                if isinstance(v, (int, float)):
                    acc.setdefault(f"{scope}.{k}", []).append(v)
    return {k: {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 2), "n": len(v)}
            for k, v in sorted(acc.items())}
