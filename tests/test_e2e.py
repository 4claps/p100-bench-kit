"""End to end through the HTTP API with fake GPU tools: login, system,
create (validation + preflight), start, wait, results, export, import,
compare, invalid marking, delete."""

import os
import time

import pytest
from fastapi.testclient import TestClient


def make_client():
    from benchkit.config import Config
    from benchkit.main import create_app
    return TestClient(create_app(Config()))


def wait_done(c, rid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/runs/{rid}").json()["run"]["status"]
        if st not in ("created", "running"):
            return st
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def login(c):
    assert c.post("/api/login", json={"token": "wrong"}).status_code == 401
    assert c.get("/api/system").status_code == 401
    assert c.post("/api/login", json={"token": "test-token"}).status_code == 200


def test_auth_required_and_bearer(env):
    with make_client() as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/").status_code == 200
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer test-token"}).status_code == 200


def test_speed_and_server_run(env):
    with make_client() as c:
        login(c)
        sysinfo = c.get("/api/system").json()
        assert len(sysinfo["snapshot"]["gpus"]) == 2
        assert [e["name"] for e in sysinfo["engines"]] == ["fake"]
        models = c.get("/api/models").json()
        mid = models["custom"][0]["id"]

        bad = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [5], "modules": {"speed": {}}})
        assert bad.status_code == 400

        spec = {"model": mid, "engine": "fake", "gpus": [0, 1], "label": "e2e",
                "modules": {"speed": {"reps": 2, "kv": ["q4_0", "f16"], "depths": [0, 4096]},
                            "server": {"reps": 2, "mtp": ["off", "on"]}}}
        r = c.post("/api/runs", json=spec).json()
        assert r["estimate_s"] > 0
        assert not r["preflight"]["block"]
        assert any("custom model" in w for w in r["preflight"]["warn"])
        assert c.post(f"/api/runs/{r['id']}/start").status_code == 200
        assert wait_done(c, r["id"]) == "done"

        d = c.get(f"/api/runs/{r['id']}").json()
        sp = d["summary"]["speed"]
        assert sp["kv=q4_0"]["pp512@0"]["n"] == 2 and sp["kv=f16"]["tg128@4096"]["median"] > 0
        sv = d["summary"]["server"]
        assert sv["mtp=on"]["code.tg"]["median"] == 50.0 and sv["mtp=off"]["code.tg"]["median"] == 26.0
        assert "code.draft_accept" in sv["mtp=on"] and "code.draft_accept" not in sv["mtp=off"]
        assert d["run"]["snapshot"]["gpus"][0]["name"].startswith("Tesla")
        assert len(c.get(f"/api/runs/{r['id']}/telemetry").json()) >= 1
        assert any("llama-bench" in l["line"] for l in c.get(f"/api/runs/{r['id']}/logs").json())

        b = c.get(f"/api/runs/{r['id']}/export.json").json()
        assert b["bundle_schema"] == 1 and "GPU-fake" not in str(b)
        md = c.get(f"/api/runs/{r['id']}/export.md").text
        assert "## speed" in md and "2x Tesla P100-PCIE-16GB" in md

        imp = c.post("/api/import", json=b).json()["id"]
        cmp = c.get(f"/api/compare?ids={r['id']},{imp}").json()
        assert len(cmp) == 2 and cmp[0]["summary"] == cmp[1]["summary"]
        assert c.post("/api/import", json={"nope": 1}).status_code == 400

        assert c.patch(f"/api/runs/{r['id']}", json={"invalid": True}).status_code == 400
        assert c.patch(f"/api/runs/{r['id']}", json={"invalid": True, "invalid_reason": "test",
                                                     "tags": ["a", "b"], "note": "n"}).status_code == 200
        lst = c.get("/api/runs").json()
        assert {x["id"] for x in lst} == {r["id"], imp}
        assert next(x for x in lst if x["id"] == r["id"])["invalid"]
        assert c.delete(f"/api/runs/{imp}").status_code == 200

        lay = c.put("/api/layouts/default", json={"charts": ["gpu.temp_c", "nope"]}).json()
        assert lay["charts"] == ["gpu.temp_c"]


def test_abort(env):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0],
                                      "modules": {"speed": {"reps": 10, "kv": ["f16", "q8_0", "q4_0"]}}}).json()
        c.post(f"/api/runs/{r['id']}/start")
        assert c.post(f"/api/runs/{r['id']}/start").status_code == 409
        assert c.post(f"/api/runs/{r['id']}/abort").status_code == 200
        assert wait_done(c, r["id"]) == "aborted"


@pytest.mark.skipif(not os.environ.get("BENCHKIT_TEST_HERMES_SOURCE"), reason="needs a hermes-agent checkout")
def test_agent_module_flow(env):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0],
                                      "modules": {"agent": {"tasks": ["no_tool_required", "err_case_search"],
                                                            "reps": 2, "mtp": False}}}).json()
        assert not r["preflight"]["block"], r["preflight"]
        c.post(f"/api/runs/{r['id']}/start")
        assert wait_done(c, r["id"]) == "done"
        d = c.get(f"/api/runs/{r['id']}").json()
        tasks = {(t["task"], t["rep"]): t["success"] for t in d["agent_tasks"]}
        assert tasks == {("no_tool_required", 0): True, ("no_tool_required", 1): True,
                         ("err_case_search", 0): False, ("err_case_search", 1): False}
        assert d["summary"]["agent"]["all"]["total.pass_rate"]["median"] == 0.5
