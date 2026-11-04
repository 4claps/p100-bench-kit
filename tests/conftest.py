"""Test setup: fake nvidia-smi on PATH, fake engine, temp data/models dirs.

Set BENCHKIT_TEST_HERMES_SOURCE to a hermes-agent checkout to exercise the
agent module against the real task definitions (skipped otherwise).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "fakes"
sys.path.insert(0, str(ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, models = tmp_path / "data", tmp_path / "models"
    models.mkdir()
    (models / "tiny-custom.gguf").write_bytes(b"GGUF" + b"\0" * 1024)
    monkeypatch.setenv("PATH", f"{FAKES / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("BENCHKIT_DATA_DIR", str(data))
    monkeypatch.setenv("BENCHKIT_MODELS_DIR", str(models))
    monkeypatch.setenv("BENCHKIT_ENGINES_DIR", str(FAKES / "engines"))
    monkeypatch.setenv("BENCHKIT_TOKEN", "test-token")
    monkeypatch.setenv("BENCHKIT_COOLDOWN_C", "90")
    monkeypatch.setenv("BENCHKIT_SERVER_PORT", str(18000 + os.getpid() % 1000))
    monkeypatch.setenv("BENCHKIT_HERMES_PYTHON", str(FAKES / "bin" / "fake-hermes-python"))
    hs = os.environ.get("BENCHKIT_TEST_HERMES_SOURCE")
    if hs:
        monkeypatch.setenv("BENCHKIT_HERMES_SOURCE", hs)
    return {"data": data, "models": models, "tmp": tmp_path}
