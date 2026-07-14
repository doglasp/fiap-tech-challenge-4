from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


class FakePredictor:
    symbol = "TEST"
    window = 2
    min_prices = 3
    scaler = object()

    def warm_up(self):
        return None

    def predict(self, prices, horizon=1):
        if len(prices) < self.min_prices:
            raise ValueError(
                "São necessários pelo menos 3 preços."
            )
        last = float(prices[-1])
        return [
            last + index + 1
            for index in range(horizon)
        ]


def make_client():
    app = create_app(
        predictor_instance=FakePredictor()
    )
    return TestClient(app)


def test_health_and_readiness():
    with make_client() as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    assert ready.status_code == 200
    assert ready.json()["symbol"] == "TEST"


def test_predict():
    with make_client() as client:
        response = client.post(
            "/predict",
            json={
                "prices": [10, 11, 12],
                "horizon": 2,
            },
        )

    assert response.status_code == 200
    assert response.json()["predictions"] == [
        13.0,
        14.0,
    ]


def test_predict_rejects_short_history():
    with make_client() as client:
        response = client.post(
            "/predict",
            json={
                "prices": [10, 11],
                "horizon": 1,
            },
        )

    assert response.status_code == 422
    assert "pelo menos 3" in response.json()["detail"]


def test_schema_rejects_invalid_values():
    with make_client() as client:
        negative = client.post(
            "/predict",
            json={
                "prices": [-1, 10, 11],
                "horizon": 1,
            },
        )
        horizon = client.post(
            "/predict",
            json={
                "prices": [10, 11, 12],
                "horizon": 31,
            },
        )

    assert negative.status_code == 422
    assert horizon.status_code == 422


def test_feedback_and_metrics():
    with make_client() as client:
        feedback = client.post(
            "/feedback",
            json={
                "predicted_price": 100,
                "actual_price": 102,
            },
        )
        metrics = client.get("/metrics/")

    assert feedback.status_code == 200
    assert feedback.json()["absolute_error"] == 2.0
    assert metrics.status_code == 200
    assert "lstm_api_http_requests_total" in metrics.text
    assert "lstm_model_feedback_total" in metrics.text
