import pytest
from src.ranking import HybridRanker

@pytest.fixture
def ranker():
    return HybridRanker()

def test_hybrid_score_calculation(ranker):
    item = {
        "product_id": "P1",
        "product_name": "Test Jeans",
        "category": "jeans",
        "color": "Blue",
        "fit": "Slim Fit",
        "material": "Denim",
        "season": "All-Season",
        "rating": 4.8
    }
    parsed_query = {
        "category": "jeans",
        "color": "Blue",
        "fit": "Slim Fit"
    }
    score = ranker.calculate_hybrid_score(item, semantic_sim=0.8, parsed_query=parsed_query)
    assert 0.0 <= score <= 1.0
    assert score > 0.8  # Metadata match should boost score above raw semantic similarity

def test_ranker_threshold_cutoff(ranker):
    candidates = [
        {"product_id": "P1", "rating": 4.5},
        {"product_id": "P2", "rating": 3.5}
    ]
    distances = [0.1, 0.95]  # P2 has very high distance (low similarity)
    parsed_query = {}

    ranked = ranker.rank(
        candidates=candidates,
        distances=distances,
        parsed_query=parsed_query,
        similarity_threshold=0.50,
        top_k=4
    )
    assert len(ranked) == 1
    assert ranked[0]["product_id"] == "P1"
