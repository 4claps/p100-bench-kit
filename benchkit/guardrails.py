"""Guardrails evaluated on every telemetry sample.

Two checks carried over from the Houston lessons:
  - low_clock: a busy GPU (util >= busy_util_pct) running below low_clock_mhz.
    Cards stuck at their idle clock invalidated a whole early result.
  - thermal: a GPU above thermal_warn_c (throttling territory).

Unlike the bakeoff version, a condition must hold for sustain_s before it
fires, fires once per episode per GPU, and clears when the GPU recovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Alert:
    kind: str          # low_clock | thermal | recovered
    gpu: str
    message: str
    value: float


@dataclass
class Guardrails:
    busy_util_pct: float = 50
    low_clock_mhz: float = 1000
    thermal_warn_c: float = 80
    sustain_s: float = 10
    _since: dict = field(default_factory=dict)    # (kind, gpu) -> first t the condition held
    _active: set = field(default_factory=set)     # (kind, gpu) currently alerted
    fired: list = field(default_factory=list)     # every Alert, for the run record

    def _conditions(self, g: dict) -> dict[str, float | None]:
        util, sm, temp = g.get("util_pct"), g.get("sm_mhz"), g.get("temp_c")
        num = lambda v: isinstance(v, (int, float))  # noqa: E731
        return {
            "low_clock": sm if (num(util) and num(sm) and util >= self.busy_util_pct
                                and sm < self.low_clock_mhz) else None,
            "thermal": temp if (num(temp) and temp > self.thermal_warn_c) else None,
        }

    def check(self, sample: dict) -> list[Alert]:
        t = sample.get("t", 0)
        out: list[Alert] = []
        for gpu, g in (sample.get("gpus") or {}).items():
            for kind, value in self._conditions(g).items():
                key = (kind, gpu)
                if value is None:
                    self._since.pop(key, None)
                    if key in self._active:
                        self._active.discard(key)
                        out.append(Alert("recovered", gpu, f"GPU {gpu}: {kind} cleared", 0))
                    continue
                start = self._since.setdefault(key, t)
                if key not in self._active and t - start >= self.sustain_s:
                    self._active.add(key)
                    if kind == "low_clock":
                        msg = (f"GPU {gpu}: busy but SM clock {value:.0f} MHz < "
                               f"{self.low_clock_mhz:.0f} MHz for {t - start:.0f}s; results may be invalid")
                    else:
                        msg = (f"GPU {gpu}: {value:.0f} °C > {self.thermal_warn_c:.0f} °C for "
                               f"{t - start:.0f}s; likely throttling")
                    out.append(Alert(kind, gpu, msg, value))
        self.fired.extend(a for a in out if a.kind != "recovered")
        return out
