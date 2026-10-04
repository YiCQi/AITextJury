import pytest
from fastapi.testclient import TestClient

from aitextjury import __version__
from aitextjury.main import World, build_app
import aitextjury.config as cfg


def make_client(tmp_path):
    # point the data dir at a scratch area so tests never touch real data
    cfg.data_dir = lambda: tmp_path
    import aitextjury.providers.manager as pm
    world = World()
    world.providers.path = tmp_path / "providers.json"
    world.providers._providers = []
    world.calibration.root = tmp_path / "calibration"
    world.history.root = tmp_path / "history"
    world.rebuild_engine()
    app = build_app(world)
    return TestClient(app)


TEXT = ("In today's fast-paced world, effective communication is crucial. "
        "Furthermore, it is important to note that listening matters.")


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"] == __version__


def test_detectors_endpoint(client):
    r = client.get("/api/detectors")
    assert r.status_code == 200
    dets = r.json()["detectors"]
    ids = {d["id"] for d in dets}
    assert {"stylometry", "llm_judge", "binoculars"} <= ids
    styl = next(d for d in dets if d["id"] == "stylometry")
    assert styl["available"] is True


def test_analyze_stylometry(client):
    r = client.post("/api/analyze",
                    json={"text": TEXT, "detector_ids": ["stylometry"]})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["detector_id"] == "stylometry"
    assert "consensus" in body
    assert body["consensus"]["contributors"]


def test_analyze_unknown_detector_422(client):
    r = client.post("/api/analyze", json={"text": TEXT,
                                          "detector_ids": ["ghost"]})
    assert r.status_code == 422


def test_history_roundtrip(client):
    client.post("/api/analyze", json={"text": TEXT,
                                      "detector_ids": ["stylometry"]})
    lst = client.get("/api/history").json()["entries"]
    assert lst, "history list must contain the run"
    rid = lst[0]["id"]
    full = client.get(f"/api/history/{rid}")
    assert full.status_code == 200
    assert full.json()["text"] == TEXT
    assert client.delete(f"/api/history/{rid}").status_code == 200
    assert client.get(f"/api/history/{rid}").status_code == 404


def test_provider_crud_and_masking(client):
    r = client.put("/api/keys", json={
        "id": "openai", "kind": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test-1234567890abcdef", "default_model": "gpt-4o-mini"})
    assert r.status_code == 200
    providers = r.json()["providers"]
    assert providers[0]["id"] == "openai"
    assert "1234567890abcdef" not in str(providers)   # never leak full key
    assert providers[0]["has_key"] is True

    # overwrite without resending key -> old key kept
    r = client.put("/api/keys", json={
        "id": "openai", "kind": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "", "default_model": "gpt-4o"})
    assert r.json()["providers"][0]["has_key"] is True

    assert client.delete("/api/keys/openai").status_code == 200


def test_calibration_endpoints(client):
    r = client.get("/api/calibration")
    assert r.status_code == 200
    c = r.json()["calibration"]
    assert c["stylometry"]["status"] in ("calibrated", "uncalibrated")

    r = client.post("/api/calibration/run",
                    json={"detector_ids": ["stylometry"], "dataset": "demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_samples"] >= 20
    assert body["results"]["stylometry"]["ok"] is True
    assert body["results"]["stylometry"]["auc"] is not None

    # after the run the fit should be visible
    r = client.get("/api/calibration")
    assert r.json()["calibration"]["stylometry"]["status"] == "calibrated"


def test_bench_endpoint(client):
    r = client.get("/api/bench")
    assert r.status_code == 200
    datasets = r.json()["datasets"]
    assert any(d["name"] == "demo" for d in datasets)


def test_settings_roundtrip(client):
    r = client.put("/api/settings", json={
        "detectors": {"binoculars": {"observer_model": "gpt2"}}})
    assert r.status_code == 200
    assert (r.json()["settings"]["detectors"]["binoculars"]
            ["observer_model"] == "gpt2")
    r = client.get("/api/settings")
    assert r.status_code == 200
