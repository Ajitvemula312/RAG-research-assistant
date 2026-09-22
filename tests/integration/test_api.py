from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import create_app


def test_health_endpoint(tmp_path):
    settings = Settings(sqlite_path=str(tmp_path / "db.sqlite"), chroma_path=str(tmp_path / "chroma"))
    client = TestClient(create_app(settings))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"