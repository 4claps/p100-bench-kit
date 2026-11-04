"""llama.cpp builds shipped in the image.

Each engine is a directory under the engines dir with bin/llama-server,
bin/llama-bench and an engine.json written at image build time:

    {"name": "kmic", "label": "...", "repo": "...", "commit": "...",
     "cuda_archs": "60;61;70", "multi_gpu_split": "tensor",
     "env": {"GGML_CUDA_P2P": "1"}, "notes": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path


def discover(engines_dir: Path) -> list[dict]:
    out = []
    for d in sorted(Path(engines_dir).glob("*/engine.json")):
        try:
            e = json.loads(d.read_text())
        except ValueError:
            continue
        root = d.parent
        e["bin_dir"] = str(root / "bin")
        e["available"] = all((root / "bin" / b).exists() for b in ("llama-server", "llama-bench"))
        e.setdefault("env", {})
        e.setdefault("multi_gpu_split", "layer")
        out.append(e)
    return out


def split_args(engine: dict, gpus: list[int]) -> list[str]:
    """Split mode and an even tensor split for the selected cards."""
    if len(gpus) <= 1:
        return ["-sm", "none"]
    return ["-sm", engine.get("multi_gpu_split", "layer"), "-ts", "/".join("1" for _ in gpus)]


def env_for(engine: dict, gpus: list[int]) -> dict:
    return {**engine.get("env", {}), "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)}
