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
