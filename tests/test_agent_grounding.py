import pytest

from src.agent import RAGAgent


def grounded_product(product_id="PRD1001"):
    return {
        "product_id": product_id,
        "product_name": "Blue Slim Fit Jeans",
        "category": "jeans",
        "subcategory": "Slim Fit Denim Jeans",
        "gender": "Men",
        "color": "Blue",
        "size": "M",
        "fit": "Slim Fit",
        "material": "Denim",
        "brand": "Example Brand",
        "price_inr": 1999,
        "rating": 4.5,
        "season": "All-Season",
        "description": "A blue denim jean with a slim fit.",
        "match_percentage": 90,
        "matched_chunks": [{
            "chunk_id": f"{product_id}_chunk_0",
            "chunk_type": "identity_description",
            "text": "Product: Blue Slim Fit Jeans | Category: jeans | Gender: Men | Color: Blue | Material: Denim",
        }],
    }


def test_grounded_context_contains_full_product_and_chunk_evidence():
    context, allowed_ids = RAGAgent._grounded_context([grounded_product()])
    assert "product_id=PRD1001" in context
    assert "subcategory=Slim Fit Denim Jeans" in context
    assert "price_inr=1999" in context
    assert "retrieved_chunk_evidence" in context
    assert allowed_ids == {"PRD1001"}


def test_agent_refuses_products_without_retrieved_evidence():
    agent = RAGAgent()
    result = agent.generate_recommendation_explanation(
        "blue jeans", [{"product_id": "PRD1001", "product_name": "Unknown"}], {}
    )
    assert result["source"] == "System"
    assert "sufficiently supported" in result["text"]


def test_agent_sends_grounded_context_and_configured_model(monkeypatch):
    agent = RAGAgent(model_name="llama3.2:3b")
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"response": "PRD1001 matches the blue denim request."}

    def fake_post(url, json, timeout):
        captured.update({"url": url, "payload": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(agent, "is_ollama_available", lambda: True)
    monkeypatch.setattr("src.agent.requests.post", fake_post)
    result = agent.generate_recommendation_explanation(
        "blue jeans", [grounded_product()], {}
    )

    assert result["source"] == "Ollama (llama3.2:3b)"
    assert captured["payload"]["model"] == "llama3.2:3b"
    assert "product_id=PRD1001" in captured["payload"]["prompt"]
    assert "retrieved_chunk_evidence" in captured["payload"]["prompt"]
    assert "Do not invent" in captured["payload"]["prompt"]


def test_agent_rejects_unknown_product_id_citation(monkeypatch):
    agent = RAGAgent(model_name="llama3.2:3b")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"response": "PRD9999 is the best match for this request."}

    monkeypatch.setattr(agent, "is_ollama_available", lambda: True)
    monkeypatch.setattr("src.agent.requests.post", lambda *args, **kwargs: Response())
    result = agent.generate_recommendation_explanation(
        "unknown citation test", [grounded_product()], {}
    )
    assert result["source"] == "Smart Fashion Engine (Local Fallback)"