import hashlib
import json
from pathlib import Path

import pytest
import yaml

from benchkit import export, hw, telemetry
from benchkit.db import DB
from benchkit.guardrails import Guardrails
from benchkit.llama_log import parse_print_timing
from benchkit.models import ModelStore, load_manifest
from benchkit.modules.base import Module
from benchkit.modules.speed import metric_name, parse_bench_jsonl

ROOT = Path(__file__).resolve().parents[1]


# ---------------- parsers ----------------
def test_print_timing_real_kmic_lines():
    lines = [
        "1.41.518.006 I slot print_timing: id  0 | task 0 | prompt eval time =    1262.49 ms /    55 tokens (   22.95 ms per token,    43.56 tokens per second)",
        "1.41.518.014 I slot print_timing: id  0 | task 0 |        eval time =     620.85 ms /    28 tokens (   22.99 ms per token,    43.49 tokens per second)",
        "1.41.518.032 I slot print_timing: id  0 | task 0 | draft acceptance = 0.66667 (   18 accepted /    27 generated), mean len =  3.00",
        "1.49.336.621 I slot print_timing: id  0 | task 13 | n_gen =    122, tg =  39.80 t/s, tg_3s =  40.12 t/s",
    ]
    evs = [parse_print_timing(l) for l in lines]
    assert [e.kind if e else None for e in evs] == ["prompt", "eval", "draft", None]
    assert evs[0].tokens == 55 and evs[0].tokens_per_s == 43.56
    assert evs[2].acceptance == 0.66667 and evs[2].accepted == 18


def test_bench_jsonl_and_metric_names():
    lines = ["noise", json.dumps({"n_prompt": 512, "n_gen": 0, "n_depth": 16384, "avg_ts": 349.9}),
             json.dumps({"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 26.2}), "{bad json"]
    rows = parse_bench_jsonl(lines)
    assert [metric_name(r) for r in rows] == ["pp512@16384", "tg128@0"]


def test_gpu_csv_and_sample():
    g = hw.parse_gpu_csv("0, Tesla P100-PCIE-16GB, GPU-x, 00000000:03:00.0, 580.1, 86.0, 16384, 250.00, 250.00, 3, 3, 16, 16, 1328, 715")
    assert g[0]["name"] == "Tesla P100-PCIE-16GB" and g[0]["memory_total"] == 16384 and g[0]["power_limit"] == 250.0
    s = telemetry.parse_gpu_sample("1, 55, 1328, 715, 140.2, 98, 60, 9000, [N/A], 3, 16, P0, 0x0")
    assert s["1"]["sm_mhz"] == 1328 and s["1"]["fan_pct"] is None and s["1"]["pstate"] == "P0"


def test_dmidecode_and_cpuinfo():
    text = """Handle 0x0040, DMI type 17, 40 bytes
Memory Device
\tSize: 8 GB
\tLocator: DIMM_A1
\tType: DDR3
\tSpeed: 1600 MT/s
\tConfigured Memory Speed: 1333 MT/s

Handle 0x0041, DMI type 17, 40 bytes
Memory Device
\tSize: No Module Installed
\tLocator: DIMM_A2
"""
    d = hw.parse_dmidecode_memory(text)
    assert len(d) == 1 and d[0]["type"] == "DDR3" and d[0]["configured_speed"] == "1333 MT/s"
    c = hw.parse_cpuinfo("processor : 0\nmodel name : Xeon\nphysical id : 0\ncore id : 0\n\n"
                         "processor : 1\nmodel name : Xeon\nphysical id : 0\ncore id : 0\n")
    assert c == {"model": "Xeon", "sockets": 1, "cores": 1, "threads": 2}


# ---------------- guardrails ----------------
def test_guardrails_sustain_once_and_recover():
    g = Guardrails(busy_util_pct=50, low_clock_mhz=1000, thermal_warn_c=80, sustain_s=10)
    busy_low = {"util_pct": 99, "sm_mhz": 405, "temp_c": 50}
    fired = []
    for t in range(0, 30, 2):
        fired += g.check({"t": t, "gpus": {"0": busy_low}})
    assert [a.kind for a in fired] == ["low_clock"]  # once, not every sample
    rec = g.check({"t": 31, "gpus": {"0": {"util_pct": 99, "sm_mhz": 1328, "temp_c": 50}}})
    assert [a.kind for a in rec] == ["recovered"]
    # short spike below the sustain window never fires
    g2 = Guardrails(sustain_s=10)
    assert not g2.check({"t": 0, "gpus": {"0": {"temp_c": 85}}})
    assert not g2.check({"t": 5, "gpus": {"0": {"temp_c": 85}}})
    assert not g2.check({"t": 6, "gpus": {"0": {"temp_c": 60}}})
    # idle GPU at idle clock is fine
    assert not Guardrails(sustain_s=0).check({"t": 0, "gpus": {"0": {"util_pct": 0, "sm_mhz": 405}}})


# ---------------- module params ----------------
def test_module_param_validation():
    from benchkit.modules import MODULES
    sp = MODULES["speed"]
    p = sp.normalize({"reps": 2, "kv": ["f16"], "depths": [16384, 0, 0]})
    assert p["reps"] == 2 and p["depths"] == [0, 16384] and p["pp"] == 512
    with pytest.raises(ValueError):
        sp.normalize({"reps": 0})
    with pytest.raises(ValueError):
        sp.normalize({"kv": ["q3_0"]})
    with pytest.raises(ValueError):
        sp.normalize({"bogus": 1})
    ag = MODULES["agent"].defaults()
    assert "err_big_file_read" not in ag["tasks"] and "no_tool_required" not in ag["tasks"]


# ---------------- models ----------------
def test_manifest_is_valid():
    ms = load_manifest(ROOT / "models" / "manifest.yaml")
    ids = [m["id"] for m in ms]
    assert len(ids) == len(set(ids))
    for m in ms:
        assert m["kind"] in ("dense", "moe")
        for f in m["files"]:
            assert len(f["sha256"]) == 64 and f["size"] > 0


def test_model_store_adopts_renamed_file_and_verifies(tmp_path):
    content = b"GGUF" + b"x" * 5000
    sha = hashlib.sha256(content).hexdigest()
    man = tmp_path / "m.yaml"
    man.write_text(yaml.safe_dump({"schema": 1, "models": [{
        "id": "t", "label": "T", "repo": "x/y", "kind": "dense", "mtp": False, "vram_gb": 1,
        "files": [{"name": "official.gguf", "size": len(content), "sha256": sha}]}]}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "renamed-by-user.gguf").write_bytes(content)
    (models / "other.gguf").write_bytes(b"GGUF")
    st = ModelStore(models, tmp_path / "data", man)
    s = st.status(st.get("t"))
    assert s["present"] and not s["verified"]
    st.verify("t")
    import time
    for _ in range(50):
        if st.jobs["t"].status != "running":
            break
        time.sleep(0.05)
    assert st.jobs["t"].status == "done"
    assert st.status(st.get("t"))["verified"]
    lst = st.list()
    assert [c["label"] for c in lst["custom"]] == ["other.gguf"]
    p, comparable = st.model_path("custom:other.gguf")
    assert p.name == "other.gguf" and not comparable
    assert st.model_path("custom:../../etc/passwd")[0] is None


# ---------------- db + export ----------------
def test_db_results_and_markdown(tmp_path):
    db = DB(tmp_path / "x.sqlite")
    db.create_run("r1", {"model": "m", "engine": "e", "gpus": [0], "modules": {}, "comparable": True}, 60)
    for rep, v in enumerate([10.0, 12.0, 11.0]):
        db.add_result("r1", "speed", rep, "kv=q4_0", "tg128@0", v, "t/s")
    db.add_telemetry("r1", [{"t": 1, "gpus": {"0": {"temp_c": 50, "sm_mhz": 1328}}},
                            {"t": 2, "gpus": {"0": {"temp_c": 60, "sm_mhz": 1300}}}])
    db.update_run("r1", status="done", snapshot_json=json.dumps(
        {"gpus": [{"index": 0, "name": "P100", "uuid": "GPU-secret", "pci_bus_id": "0000:03:00.0"}],
         "memory": {"total_gb": 16, "type": "DDR3", "dimms": [{"size": "8 GB", "part_number": "SERIAL1"}]}}))
    b = export.bundle(db, "r1", "test")
    export.validate_bundle(b)
    assert b["summary"]["speed"]["kv=q4_0"]["tg128@0"]["median"] == 11.0
    assert "GPU-secret" not in json.dumps(b) and "SERIAL1" not in json.dumps(b)
    assert b["telemetry_summary"]["gpu0.temp_c"]["max"] == 60
    md = export.markdown(b)
    assert "| kv=q4_0 | tg128@0 | 11.0 t/s |" in md and "1x P100" in md
    with pytest.raises(ValueError):
        db.update_run("r1", bogus=1)
    assert db.mark_unfinished_interrupted() == []


def test_build_tables_skips_custom_and_invalid(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import build_tables

    def b(rid, comparable=True, invalid=False, tg=26.0):
        return {"bundle_schema": 1, "run": {"id": rid, "invalid": invalid, "invalid_reason": "x" if invalid else None},
                "spec": {"model": "qwen38-27b-q6kxl", "engine": "kmic", "gpus": [0, 1], "comparable": comparable},
                "snapshot": {"gpus": [{"name": "Tesla P100"}] * 2, "memory": {"total_gb": 16, "type": "DDR3"}},
                "summary": {"speed": {"kv=q4_0": {"tg128@0": {"median": tg, "min": tg, "max": tg, "n": 3, "unit": "t/s"}}}}}
    (tmp_path / "p100x2").mkdir()
    for name, bb in {"a": b("a"), "b": b("b", tg=38.0), "c": b("c", comparable=False), "d": b("d", invalid=True)}.items():
        (tmp_path / "p100x2" / f"{name}.json").write_text(json.dumps(bb))
    (tmp_path / "junk.json").write_text("{}")
    md = build_tables.build(tmp_path)
    assert "2 runs." in md
    assert md.index("| 38.0 |") < md.index("| 26.0 |")  # best first
    assert "custom model" in md and "marked invalid" in md and "junk.json" in md


def test_split_args():
    from benchkit.engines import split_args
    e = {"multi_gpu_split": "tensor"}
    assert split_args(e, [0]) == ["-sm", "none"]
    assert split_args(e, [0, 1]) == ["-sm", "tensor", "-ts", "1/1"]
    assert split_args(e, [0, 1, 2, 3], "layer") == ["-sm", "layer", "-ts", "1/1/1/1"]
    assert split_args(e, [0, 1], "bogus") == ["-sm", "tensor", "-ts", "1/1"]
