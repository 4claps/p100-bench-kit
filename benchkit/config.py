"""Runtime configuration, all from environment variables.

Nothing host-specific is baked in: the image runs on whatever box it is
started on, with models and data mounted in.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    host: str = field(default_factory=lambda: _env("BENCHKIT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("BENCHKIT_PORT", "8090")))
    data_dir: Path = field(default_factory=lambda: Path(_env("BENCHKIT_DATA_DIR", "/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("BENCHKIT_MODELS_DIR", "/models")))
    engines_dir: Path = field(default_factory=lambda: Path(_env("BENCHKIT_ENGINES_DIR", "/opt/engines")))
    hermes_python: str = field(default_factory=lambda: _env("BENCHKIT_HERMES_PYTHON", "/opt/hermes-venv/bin/python"))
    hermes_source: Path = field(default_factory=lambda: Path(_env("BENCHKIT_HERMES_SOURCE", "/opt/hermes-agent")))
    manifest: Path = field(default_factory=lambda: Path(_env(
        "BENCHKIT_MANIFEST", str(Path(__file__).resolve().parents[1] / "models" / "manifest.yaml"))))
    # agent tasks run as this unprivileged uid when the app itself runs as root
    agent_uid: int = field(default_factory=lambda: int(_env("BENCHKIT_AGENT_UID", "1000")))
    server_port: int = field(default_factory=lambda: int(_env("BENCHKIT_SERVER_PORT", "8099")))

    # guardrails
    busy_util_pct: float = field(default_factory=lambda: float(_env("BENCHKIT_BUSY_UTIL_PCT", "50")))
    low_clock_mhz: float = field(default_factory=lambda: float(_env("BENCHKIT_LOW_CLOCK_MHZ", "1000")))
    thermal_warn_c: float = field(default_factory=lambda: float(_env("BENCHKIT_THERMAL_WARN_C", "80")))
    guardrail_sustain_s: float = field(default_factory=lambda: float(_env("BENCHKIT_GUARDRAIL_SUSTAIN_S", "10")))
    cooldown_c: float = field(default_factory=lambda: float(_env("BENCHKIT_COOLDOWN_C", "48")))
    cooldown_timeout_s: float = field(default_factory=lambda: float(_env("BENCHKIT_COOLDOWN_TIMEOUT_S", "600")))
    sample_interval_s: float = field(default_factory=lambda: float(_env("BENCHKIT_SAMPLE_INTERVAL_S", "1")))

    token: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "benchkit.sqlite"

    def load_token(self) -> str:
        """BENCHKIT_TOKEN if set; otherwise a random token generated on first
        start and kept in the data dir. The app never runs without one."""
        tok = os.environ.get("BENCHKIT_TOKEN", "").strip()
        if not tok:
            path = self.data_dir / "token"
            if path.exists():
                tok = path.read_text().strip()
            if not tok:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                tok = secrets.token_urlsafe(24)
                path.write_text(tok + "\n")
                path.chmod(0o600)
        self.token = tok
        return tok
