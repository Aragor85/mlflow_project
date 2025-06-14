from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API d'analyse des sentiments (modèle USE)"}

def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "Air Paradis is absolutely fantastic!"}
    )
    result = response.json()
    print("\n[POSITIVE TEST] →", result)

    assert response.status_code == 200
    assert "prediction" in result
    assert result["prediction"] in [0, 1]


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "This airline is the worst I've ever seen."}
    )
    result = response.json()
    print("\n[NEGATIVE TEST] →", result)

    assert response.status_code == 200
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
