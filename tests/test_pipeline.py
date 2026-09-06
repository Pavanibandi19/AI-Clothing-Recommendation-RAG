import pytest
from src.retrieval import ProductRetriever
from src.agent import RAGAgent

@pytest.fixture
def retriever():
    return ProductRetriever()

@pytest.fixture
def agent():
    return RAGAgent()

def test_full_retrieval_jeans_under_2000(retriever):
    res = retriever.retrieve("blue slim fit jeans under 2000", top_k=4)
    assert res["count"] > 0
    items = res["items"]
    assert len(items) <= 4
    for item in items:
        assert item["price_inr"] <= 2000
        assert item["category"].lower() == "jeans"

def test_full_retrieval_no_match(retriever):
    # Query for something impossible under price limit
    res = retriever.retrieve("jackets under 100", top_k=4)
    assert res["count"] == 0
    assert len(res["items"]) == 0
    assert "No matching products" in res["message"]

def test_formal_dress_for_men_retrieves_formal_dress_shirt(retriever):
    res = retriever.retrieve("yellow formal dress for men", top_k=4)

    assert res["parsed_query"]["category"] is None
    assert res["parsed_query"]["gender"] == "Men"
    assert res["parsed_query"]["color"] == "Yellow"
    assert res["parsed_query"]["style"] == "formal"
    assert res["count"] > 0
    assert any(
        item["product_id"] == "PRD1011"
        and item["category"] == "shirts"
        and item["subcategory"] == "Formal Dress Shirt"
        for item in res["items"]
    )

def test_womens_dresses_do_not_leak_mens_products(retriever):
    res = retriever.retrieve("women's dresses", top_k=10)
    assert res["count"] > 0
    assert all(item["category"] == "dresses" for item in res["items"])
    assert all(item["gender"] in ("Women", "Unisex") for item in res["items"])

@pytest.mark.parametrize("query", [
    "dress for men",
    "formal dress for men",
    "casual dress for men",
    "party dress for men",
])
def test_broad_mens_dress_intent_uses_semantic_retrieval(retriever, query):
    result = retriever.retrieve(query, top_k=4)
    assert result["parsed_query"]["gender"] == "Men"
    assert result["parsed_query"]["category"] is None
    assert result["count"] > 0
    assert all(item["gender"] in ("Men", "Unisex") for item in result["items"])
    assert result["diagnostics"]["retrieved_chunks_count"] > 0

def test_explicit_shirt_and_jeans_categories_remain_hard_filters(retriever):
    shirt_result = retriever.retrieve("formal shirt for men", top_k=4)
    assert shirt_result["parsed_query"]["category"] == "shirts"
    assert all(item["category"] == "shirts" for item in shirt_result["items"])

    jeans_result = retriever.retrieve("men's jeans", top_k=4)
    assert jeans_result["parsed_query"]["category"] == "jeans"
    assert all(item["category"] == "jeans" for item in jeans_result["items"])

@pytest.mark.parametrize("query", [
    "show me cotton dresses for men",
    "show me men's dresses under 1000",
    "show me men's dresses under 4000",
    "show me a formal dress for men under 3000",
])
def test_broad_mens_dress_keeps_material_and_price_hard_constraints(retriever, query):
    result = retriever.retrieve(query, top_k=4)
    parsed = result["parsed_query"]
    assert parsed["gender"] == "Men"
    assert parsed["category"] is None
    if parsed.get("material"):
        assert all(item["material"] == parsed["material"] for item in result["items"])
    if parsed.get("max_price") is not None:
        assert all(item["price_inr"] <= parsed["max_price"] for item in result["items"])

@pytest.mark.parametrize("query", [
    "recommend something stylish for a party",
    "i need a stylish outfit for a party",
    "show me something for a party",
])
def test_party_queries_use_occasion_relevance(retriever, query):
    result = retriever.retrieve(query, top_k=4)
    assert result["parsed_query"]["intent"] == "party wear"
    assert result["count"] > 0
    assert result["diagnostics"]["retrieved_chunks_count"] > 0

def test_vector_results_have_retrieved_evidence(retriever):
    result = retriever.retrieve("blue slim fit jeans under 2000", top_k=4)
    assert result["items"]
    assert all(item["has_retrieved_evidence"] for item in result["items"])

def test_retrieval_top_k_edge_values(retriever):
    assert retriever.retrieve("jeans", top_k=0)["items"] == []
    assert retriever.retrieve("jeans", top_k=-1)["items"] == []
    assert len(retriever.retrieve("jeans", top_k=1)["items"]) <= 1

def test_agent_explanation_generation(agent, retriever):
    res = retriever.retrieve("women's cotton dresses for summer", top_k=4)
    explanation = agent.generate_recommendation_explanation(
        query="women's cotton dresses for summer",
        recommended_products=res["items"],
        parsed_query=res["parsed_query"]
    )
    assert "source" in explanation
    assert "text" in explanation
    assert len(explanation["text"]) > 20
