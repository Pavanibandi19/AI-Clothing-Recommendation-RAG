import sys
import io
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set stdout to UTF-8 with line buffering
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from src.query_parser import QueryParser
from src.data_loader import DataLoader
from src.retrieval import ProductRetriever
from src.agent import RAGAgent

qp = QueryParser()
loader = DataLoader()
retriever = ProductRetriever()
agent = RAGAgent()

test_queries = [
    "blue slim fit jeans under 2000",
    "black casual t-shirts below 1500",
    "women's cotton dresses for summer",
    "men's oversized hoodies under 2500",
    "blue jeans under \u20b92000",
    "black shirts under \u20b91500",
    "women's dresses below 3000",
    "cotton shirts",
    "party wear dresses",
    "casual clothes for summer"
]

print("=" * 70, flush=True)
print("TASK 15: END-TO-END VERIFICATION OF 10 TARGET QUERIES", flush=True)
print("=" * 70, flush=True)

all_passed = True
total_latencies = []

for idx, query in enumerate(test_queries, 1):
    print(f"\n[{idx}/10] Testing: \"{query}\"", flush=True)
    
    # 1. Validation
    is_valid = qp.is_clothing_query(query)
    print(f"  ✓ Accepted by clothing validator: {is_valid}", flush=True)
    if not is_valid:
        print("  ❌ FAILED VALIDATION!", flush=True)
        all_passed = False
        continue

    # 2. Timing Breakdown
    t0 = time.perf_counter()
    parsed = qp.parse(query)
    t_parse = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    filtered_df = loader.filter_products(parsed)
    t_filter = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    query_vec = retriever.embedder.encode([query])
    t_embed = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    res = retriever.retrieve(query, top_k=4)
    t_total = (time.perf_counter() - t3) * 1000
    total_latencies.append(t_total)

    items = res.get("items", [])
    print(f"  ✓ Constraints: max_p={parsed.get('max_price')}, gender={parsed.get('gender')}, cat={parsed.get('category')}, color={parsed.get('color')}, fit={parsed.get('fit')}, mat={parsed.get('material')}", flush=True)
    print(f"  ✓ Candidates matching hard filters: {len(filtered_df)}", flush=True)
    print(f"  ✓ Items returned: {len(items)}", flush=True)

    # 3. Hard filter verification
    max_p = parsed.get("max_price")
    target_cat = parsed.get("category")
    target_gender = parsed.get("gender")

    for item in items:
        price = item.get("price_inr", 0)
        cat = item.get("category", "")
        gender = item.get("gender", "")
        if max_p is not None and price > max_p:
            print(f"  ❌ HARD PRICE VIOLATION: item {item.get('product_id')} price ₹{price} > max ₹{max_p}", flush=True)
            all_passed = False
        if target_cat and cat.lower() != target_cat.lower():
            print(f"  ❌ HARD CATEGORY VIOLATION: item {item.get('product_id')} cat {cat} != {target_cat}", flush=True)
            all_passed = False
        if target_gender and gender.lower() not in (target_gender.lower(), "unisex"):
            print(f"  ❌ HARD GENDER VIOLATION: item {item.get('product_id')} gender {gender} != {target_gender}", flush=True)
            all_passed = False

    # 4. LLM / Fallback Explanation Generation
    explanation = agent.generate_recommendation_explanation(query, items, parsed)
    print(f"  ✓ AI Explanation ({explanation.get('source')}): {explanation.get('text')[:90]}...", flush=True)
    print(f"  ⚡ Latency: parse={t_parse:.2f}ms, filter={t_filter:.2f}ms, embed={t_embed:.2f}ms, total_retrieval={t_total:.2f}ms", flush=True)

avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
print("\n" + "=" * 70, flush=True)
print(f"OVERALL RESULT: {'✅ ALL 10 QUERIES PASSED WITH 100% COMPLIANCE' if all_passed else '❌ SOME QUERIES FAILED'}", flush=True)
print(f"AVERAGE RETRIEVAL LATENCY: {avg_latency:.2f}ms", flush=True)
print("=" * 70, flush=True)
