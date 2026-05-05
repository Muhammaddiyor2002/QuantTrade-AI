from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_strategies_endpoint() -> None:
    r = client.get("/strategies")
    assert r.status_code == 200
    body = r.json()
    assert "ma_crossover" in body["strategies"]


def test_backtest_endpoint() -> None:
    r = client.post(
        "/backtest",
        json={
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "limit": 300,
            "strategy": "ma_crossover",
            "params": {"fast": 10, "slow": 30},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "metrics" in body
    assert "final_equity" in body
