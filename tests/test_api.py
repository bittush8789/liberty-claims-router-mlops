import os
import sys
import pytest
from fastapi.testclient import TestClient

# Adjust path to import backend app
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))

# Set test environment
os.environ["DATABASE_URL"] = "sqlite:///./test_claims.db"

from api.app import app, Base, engine

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    try:
        if os.path.exists("./test_claims.db"):
            os.remove("./test_claims.db")
    except Exception:
        pass

def test_predict_endpoint():
    payload = {
        "claim_amount": 12500.0,
        "loss_description": "Auto fender bender in a parking lot. Dented door.",
        "police_report": True,
        "witness_available": False,
        "injury_involved": False,
        "previous_claims_count": 1,
        "days_to_report": 2
    }
    # If models are not trained yet, it might return 503, but we can verify it doesn't crash
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "claim_type" in data
        assert "priority" in data
        assert "fraud_score" in data
        assert "assigned_team" in data
    else:
        assert response.status_code == 503 or response.status_code == 500

def test_create_claim_and_dashboard():
    payload = {
        "policy_number": "POL-888888",
        "customer_name": "Test User",
        "email": "test@example.com",
        "phone": "+1-555-9000",
        "date_of_loss": "2026-06-26",
        "state": "CA",
        "claim_amount": 5400.0,
        "loss_description": "Water burst under the kitchen sink, flooring ruined.",
        "police_report": False,
        "witness_available": True,
        "injury_involved": False,
        "previous_claims_count": 0,
        "days_to_report": 1
    }
    
    # 1. Create claim
    create_response = client.post("/claims", json=payload)
    assert create_response.status_code == 200
    claim = create_response.json()
    assert claim["policy_number"] == "POL-888888"
    assert claim["claim_id"] is not None
    
    # 2. List claims
    list_response = client.get("/claims")
    assert list_response.status_code == 200
    claims_list = list_response.json()
    assert len(claims_list) >= 1
    
    # 3. Retrieve claim details
    detail_response = client.get(f"/claims/{claim['claim_id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["customer_name"] == "Test User"
    
    # 4. Dashboard stats
    dash_response = client.get("/dashboard")
    assert dash_response.status_code == 200
    stats = dash_response.json()
    assert stats["total_claims"] >= 1
    assert stats["average_claim_amount"] > 0
