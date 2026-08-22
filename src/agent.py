import logging
import requests
from typing import List, Dict, Any, Optional
from src.config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.model_name = model_name or OLLAMA_MODEL_NAME
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or OLLAMA_TIMEOUT

    def is_ollama_available(self) -> bool:
        """Checks if local Ollama server is running and reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(0.5, 1.0))
            return response.status_code == 200
        except requests.RequestException as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    def generate_recommendation_explanation(
        self, 
        query: str, 
        recommended_products: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a clean, simple 1-2 line explanation for each of the TOP 2 products
        based on the user's actual query and retrieved product attributes.
        """
        if not recommended_products:
            return {
                "source": "System",
                "text": "No matching products found satisfying your search criteria and price limits. Try adjusting your search query or loosening sidebar filters."
            }

        top_products = recommended_products[:2]

        # Attempt Ollama LLM call if server is available
        ollama_active = self.is_ollama_available()
        if ollama_active:
            try:
                context_lines = []
                for idx, p in enumerate(top_products, 1):
                    p_name = p.get("product_name", "Clothing Item")
                    brand = p.get("brand", "")
                    full_name = f"{brand} {p_name}".strip() if brand and not p_name.startswith(brand) else p_name
                    price = p.get("price_inr", 0)
                    fit = p.get("fit", "")
                    mat = p.get("material", "")
                    col = p.get("color", "")
                    cat = p.get("category", "")
                    gen = p.get("gender", "")
                    match_pct = p.get("match_percentage", 90)
                    context_lines.append(
                        f"Product {idx}: {full_name} | Price: Rs {price} | Category: {cat} | "
                        f"Gender: {gen} | Fit: {fit} | Material: {mat} | Color: {col} | Match Score: {match_pct}%"
                    )
                context_str = "\n".join(context_lines)

                prompt = (
                    f"You are an AI fashion stylist.\n"
                    f"User Query: \"{query}\"\n\n"
                    f"Top Products:\n{context_str}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"For each product above, provide a short 1-2 line explanation of why it was recommended for the user's query.\n"
                    f"Do NOT include technical details, vector explanations, or long introductions.\n"
                    f"Use this exact format:\n\n"
                    f"1. [Product Full Name]\n"
                    f"[1-2 line reason why it matches the query]\n\n"
                    f"2. [Product Full Name]\n"
                    f"[1-2 line reason why it matches the query]"
                )

                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 180
                    }
                }

                request_timeout = min(self.timeout, 10) if self.timeout else 10
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=request_timeout
                )

                if resp.status_code == 200:
                    result_json = resp.json()
                    response_text = result_json.get("response", "").strip()
                    if response_text and len(response_text) > 30:
                        return {
                            "source": f"Ollama ({self.model_name})",
                            "text": response_text
                        }
            except (requests.RequestException, ValueError, TypeError) as exc:
                logger.warning("Ollama generation failed or timed out (%s). Falling back to Smart Fashion Engine.", exc)

        # Simple deterministic fallback for top 2 items
        return self._generate_fallback_summary(query, top_products, parsed_query)

    def _generate_fallback_summary(
        self, 
        query: str, 
        products: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple, clean 1-2 line explanation for each of the top 2 products."""
        top_products = products[:2]
        lines = []

        for idx, p in enumerate(top_products, 1):
            name = p.get("product_name", "Clothing Item")
            brand = p.get("brand", "")
            full_title = f"{brand} {name}".strip() if brand and not name.startswith(brand) else name
            match_score = p.get("match_percentage", 90)

            # Build matched attributes description
            matched = []
            if parsed_query.get("gender") and parsed_query["gender"].lower() in p.get("gender", "").lower():
                matched.append(f"{p.get('gender')}'s")
            if parsed_query.get("color") and parsed_query["color"].lower() in p.get("color", "").lower():
                matched.append(p.get("color").lower())
            if parsed_query.get("fit") and parsed_query["fit"].lower() in p.get("fit", "").lower():
                fit_val = p.get("fit", "").strip().lower()
                if not fit_val.endswith("fit"):
                    matched.append(f"{fit_val} fit")
                else:
                    matched.append(fit_val)
            if parsed_query.get("material") and parsed_query["material"].lower() in p.get("material", "").lower():
                matched.append(p.get("material").lower())
            if parsed_query.get("category"):
                matched.append(p.get("category", "").lower())
            if parsed_query.get("season") and parsed_query["season"].lower() in p.get("season", "").lower():
                matched.append(f"for {p.get('season').lower()}")
            if parsed_query.get("max_price") and p.get("price_inr", 0) <= parsed_query["max_price"]:
                matched.append(f"under ₹{int(parsed_query['max_price'])}")

            attr_desc = " ".join(matched) if matched else p.get("category", "clothing")

            if idx == 1:
                reason = f"Recommended because it closely matches your request for {attr_desc} and has a strong relevance score ({match_score}% Match)."
            else:
                reason = f"Recommended because it matches your requested {attr_desc} with high relevance to your search ({match_score}% Match)."

            lines.append(f"**{idx}. {full_title}**\n\n{reason}")

        return {
            "source": "Smart Fashion Engine (Local Fallback)",
            "text": "\n\n".join(lines)
        }
