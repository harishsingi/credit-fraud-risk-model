from src.model import score
def test_higher_risk_scores_higher():
    assert score(90000,0,0.1)["fraud_probability"] < score(30000,3,0.95)["fraud_probability"]
