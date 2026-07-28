"""
Tests for Comparison Mode (/api/compare) and Feedback API (/api/feedback).
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from main import app, FEEDBACK_FILE


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "fasttext" in data["models_loaded"]
    assert "word2vec" in data["models_loaded"]


def test_compare_endpoint(client):
    payload = {"query": "acount balanc", "top_k": 3}
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "fasttext" in data
    assert "word2vec" in data
    assert data["fasttext"]["predicted_category"] == "Account Management"
    assert "acount" in data["word2vec"]["oov_tokens"]
    assert "balanc" in data["word2vec"]["oov_tokens"]


def test_feedback_endpoint(client):
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)

    payload = {
        "query": "acount balanc",
        "result_id": 58,
        "result_query": "What happens to a dormant account?",
        "category": "Account Management",
        "feedback": "helpful",
        "comments": "Great match despite typo!"
    }
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    assert os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["query"] == "acount balanc"
        assert record["result_id"] == 58
        assert record["feedback"] == "helpful"
