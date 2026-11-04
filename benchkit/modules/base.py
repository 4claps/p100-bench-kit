from __future__ import annotations


class Module:
    name = ""
    label = ""
    description = ""
    # name -> {"type": int|bool|choice|multi, "default": .., "min"/"max"/"choices": .., "label": ..}
    params: dict = {}

    def defaults(self) -> dict:
        return {k: v["default"] for k, v in self.params.items()}

    def normalize(self, given: dict | None) -> dict:
        """Merge user params over defaults and validate types and ranges."""
        out = self.defaults()
        for k, v in (given or {}).items():
            spec = self.params.get(k)
            if spec is None:
                raise ValueError(f"{self.name}: unknown parameter {k!r}")
            t = spec["type"]
            if t == "int":
                v = int(v)
                if not spec.get("min", v) <= v <= spec.get("max", v):
                    raise ValueError(f"{self.name}.{k} must be {spec.get('min')}..{spec.get('max')}")
            elif t == "bool":
                v = bool(v)
            elif t == "choice":
                if v not in spec["choices"]:
                    raise ValueError(f"{self.name}.{k} must be one of {spec['choices']}")
            elif t == "multi":
                v = list(v)
                bad = [x for x in v if x not in spec["choices"]]
                if bad or not v:
                    raise ValueError(f"{self.name}.{k}: pick at least one of {spec['choices']}")
            elif t == "ints":
                v = sorted({int(x) for x in v})
                if not v or any(x < 0 for x in v):
                    raise ValueError(f"{self.name}.{k} must be a list of non-negative integers")
            out[k] = v
        return out

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description,
                "params": self.params}

    def check(self, params: dict, model: dict | None, engine: dict, snapshot: dict) -> list[str]:
        """Problems that block this module (empty list = ok)."""
        return []

    def estimate_s(self, params: dict, model_gb: float) -> int:
        return 0

    def run(self, ctx, params: dict) -> None:
        raise NotImplementedError


def load_s(model_gb: float) -> float:
    """Rough model load time (first read from disk dominates)."""
    return 20 + model_gb * 3
