import logging
import re
import requests
from typing import List, Dict, Any, Optional
from src.config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

_EXPLANATION_CACHE: Dict[str, Dict[str, Any]] = {}

class RAGAgent:
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.model_name = model_name or OLLAMA_MODEL_NAME
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or OLLAMA_TIMEOUT

    def is_ollama_available(self) -> bool:
        """Checks if local Ollama server is running and reachable with instant health check."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(0.2, 0.4))
            return response.status_code == 200
        except requests.RequestException as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    @staticmethod
    def _grounded_context(products: List[Dict[str, Any]]) -> tuple[str, set[str]]:
        """Build a bounded context from retrieved products and their chunks."""
        context_lines = []
        allowed_ids = set()
        for index, product in enumerate(products, 1):
            product_id = str(product.get("product_id", "")).strip()
            chunks = product.get("matched_chunks") or []
            if not product_id or not chunks:
                continue

            allowed_ids.add(product_id)
            fields = (
                f"product_id={product_id}",
                f"product_name={product.get('product_name', '')}",
                f"category={product.get('category', '')}",
                f"subcategory={product.get('subcategory', '')}",
                f"gender={product.get('gender', '')}",
                f"color={product.get('color', '')}",
                f"size={product.get('size', '')}",
                f"fit={product.get('fit', '')}",
                f"material={product.get('material', '')}",
                f"brand={product.get('brand', '')}",
                f"price_inr={product.get('price_inr', '')}",
                f"rating={product.get('rating', '')}",
                f"season={product.get('season', '')}",
                f"description={product.get('description', '')}",
            )
            evidence = "\n".join(
                str(chunk.get("text", ""))[:1200]
                for chunk in chunks[:3]
                if chunk.get("text")
            )
            context_lines.append(
                f"RETRIEVED PRODUCT {index}\n"
                + " | ".join(fields)
                + f"\nretrieved_chunk_evidence:\n{evidence}"
            )

        return "\n\n".join(context_lines)[:12000], allowed_ids

    @staticmethod
    def _contains_unknown_product_ids(response_text: str, allowed_ids: set[str]) -> bool:
        cited_ids = set(re.findall(r"\bPRD\d+\b", response_text.upper()))
        return bool(cited_ids - {product_id.upper() for product_id in allowed_ids})

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
        context_str, allowed_ids = self._grounded_context(top_products)
        if not context_str:
            return {
                "source": "System",
                "text": "No sufficiently supported recommendation was found in the retrieved product evidence.",
            }

        # Fast cache lookup for repeated queries
        cache_key = f"{query.strip().lower()}|" + "|".join(str(p.get('product_id', '')) for p in top_products)
        if cache_key in _EXPLANATION_CACHE:
            return _EXPLANATION_CACHE[cache_key]

        # Attempt Ollama LLM call if server is available
        ollama_active = self.is_ollama_available()
        if ollama_active:
            try:
                prompt = (
                    "You are an AI fashion stylist producing a grounded recommendation.\n"
                    f"User Query: \"{query}\"\n\n"
                    f"RETRIEVED PRODUCT EVIDENCE:\n{context_str}\n\n"
                    "STRICT RULES:\n"
                    "Use only the supplied retrieved product evidence. Do not invent or infer product IDs, names, prices, brands, categories, subcategories, genders, colors, sizes, fits, materials, ratings, seasons, descriptions, or availability. "
                    "If the evidence does not sufficiently support a recommendation, say exactly that. Cite only product IDs present in the evidence. "
                    "For each supported product, provide a short reason grounded in its evidence."
                )

                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "15m",
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 80
                    }
                }

                request_timeout = min(self.timeout, 2.5) if self.timeout else 2.5
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=request_timeout
                )

                if resp.status_code == 200:
                    result_json = resp.json()
                    response_text = result_json.get("response", "").strip()
                    if (
                        response_text
                        and len(response_text) > 30
                        and not self._contains_unknown_product_ids(response_text, allowed_ids)
                    ):
                        res = {
                            "source": f"Ollama ({self.model_name})",
                            "text": response_text
                        }
                        _EXPLANATION_CACHE[cache_key] = res
                        return res
            except (requests.RequestException, ValueError, TypeError) as exc:
                logger.warning("Ollama generation timed out or unavailable (%s). Falling back to Smart Fashion Engine.", exc)

        # Simple deterministic fallback for top 2 items
        res = self._generate_fallback_summary(query, top_products, parsed_query)
        _EXPLANATION_CACHE[cache_key] = res
        return res

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
