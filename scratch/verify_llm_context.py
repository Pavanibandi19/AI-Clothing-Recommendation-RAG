import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import ProductRetriever
from src.agent import RAGAgent
from src.config import DEFAULT_TOP_K, OLLAMA_MODEL_NAME

retriever = ProductRetriever()
agent = RAGAgent()

# 1. Test standard query
q1 = "women's cotton dresses for summer"
res1 = retriever.retrieve(q1, top_k=4)

top_products = res1["items"][:2]
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
    f"User Query: \"{q1}\"\n\n"
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

print(f"Total retrieved products returned by retriever: {len(res1['items'])}")
print(f"Products sent to LLM prompt: {len(top_products)}")
print(f"Total characters in prompt: {len(prompt)}")
print(f"Approximate token count (~4 chars/token): {len(prompt) // 4} tokens")
print(f"Exact prompt content:\n{'='*50}\n{prompt}\n{'='*50}")

# 2. Test large retrieval top_k (e.g. top_k = 50)
res_large = retriever.retrieve(q1, top_k=50)
print(f"\nWhen top_k is set to 50, retriever returns {len(res_large['items'])} items.")
print(f"RAGAgent.generate_recommendation_explanation hard limits prompt context to: top_products = recommended_products[:2]")

# 3. Test very long query (e.g. 2000 characters)
long_query = "blue jeans " * 200
res_long = retriever.retrieve(long_query, top_k=4)
print(f"\nLong query (Length: {len(long_query)} chars) retrieval result count: {res_long['count']}")
exp_long = agent.generate_recommendation_explanation(long_query, res_long["items"], res_long["parsed_query"])
print(f"Agent explanation source for long query: {exp_long['source']}")
print(f"Explanation generated successfully: {bool(exp_long['text'])}")
