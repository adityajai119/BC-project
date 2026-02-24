import json

from app import app


def test_health_endpoint():
  client = app.test_client()
  resp = client.get("/api/health")
  assert resp.status_code == 200
  data = json.loads(resp.data)
  assert data["status"] == "ok"


def test_feature_metadata():
  client = app.test_client()
  resp = client.get("/api/feature-metadata")
  assert resp.status_code == 200
  data = json.loads(resp.data)
  assert "features" in data
  assert len(data["features"]) > 0


def test_single_prediction_happy_path():
  client = app.test_client()
  meta_resp = client.get("/api/feature-metadata")
  features_meta = json.loads(meta_resp.data)["features"]
  payload = {
    "features": {f["name"]: f["default"] for f in features_meta},
  }
  resp = client.post(
    "/api/predict",
    data=json.dumps(payload),
    content_type="application/json",
  )
  assert resp.status_code == 200
  data = json.loads(resp.data)
  assert "prediction" in data
  assert "label" in data

