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

def test_style_intent_boosts_matching_catalog_text(ranker):
    formal_item = {
        "product_id": "FORMAL",
        "product_name": "Formal Shirt",
        "subcategory": "Formal Dress Shirt",
        "description": "Suitable for office wear",
        "rating": 4.0,
    }
    casual_item = {
        "product_id": "CASUAL",
        "product_name": "Casual Shirt",
        "subcategory": "Casual Button-Down",
        "description": "Suitable for everyday wear",
        "rating": 4.0,
    }
    parsed_query = {"style": "formal"}

    ranked = ranker.rank(
        [casual_item, formal_item], [0.4, 0.4], parsed_query,
        similarity_threshold=0.0, top_k=2
    )
    assert ranked[0]["product_id"] == "FORMAL"

def test_broad_style_intent_excludes_unmatched_gender_only_products(ranker):
    ranked = ranker.rank(
        [
            {
                "product_id": "PARTY",
                "product_name": "Party Evening Shirt",
                "subcategory": "Evening Shirt",
                "description": "Festive party wear",
                "rating": 4.0,
            },
            {
                "product_id": "FORMAL",
                "product_name": "Formal Dress Trousers",
                "subcategory": "Formal Dress Trousers",
                "description": "Office tailoring",
                "rating": 4.8,
            },
        ],
        [0.4, 0.1],
        {"style": "party", "intent": "party wear"},
        similarity_threshold=0.0,
        top_k=2,
    )
    assert [item["product_id"] for item in ranked] == ["PARTY"]

def test_unretrieved_candidates_are_not_given_fake_similarity(ranker):
    ranked = ranker.rank(
        [{"product_id": "RETRIEVED", "rating": 4.0}, {"product_id": "MISSING", "rating": 5.0}],
        [0.2, None],
        {},
        similarity_threshold=0.0,
        top_k=4,
        allow_unretrieved=False,
    )
    assert [item["product_id"] for item in ranked] == ["RETRIEVED"]
    assert ranked[0]["has_retrieved_evidence"] is True

def test_metadata_fallback_marks_missing_vector_evidence_honestly(ranker):
    ranked = ranker.rank(
        [{"product_id": "FALLBACK", "rating": 4.0}],
        [None],
        {},
        similarity_threshold=0.0,
        top_k=1,
        allow_unretrieved=True,
    )
    assert ranked[0]["semantic_similarity"] == 0.0
    assert ranked[0]["has_retrieved_evidence"] is False

def test_occasion_falls_back_to_retrieved_evidence_when_catalog_has_no_style_match(ranker):
    ranked = ranker.rank(
        [{"product_id": "RETRIEVED", "product_name": "Festive Nehru Jacket", "rating": 4.0}],
        [0.35],
        {"style": "party", "intent": "party wear"},
        similarity_threshold=0.0,
        top_k=1,
    )
    assert [item["product_id"] for item in ranked] == ["RETRIEVED"]
