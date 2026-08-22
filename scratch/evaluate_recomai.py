import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATASET_PATH, CHROMA_PERSIST_DIRECTORY, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.retrieval import ProductRetriever
from src.ranking import HybridRanker
from src.agent import RAGAgent

def run_phase_3_dataset_test():
    print("=" * 60)
    print("PHASE 3 — DATASET TESTING")
    print("=" * 60)
    loader = DataLoader(DATASET_PATH)
    df = loader.get_all_products()

    print(f"Total Products: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Column Names: {list(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().to_dict()}")
    print(f"Duplicate Product IDs: {df['product_id'].duplicated().sum()}")
    print(f"Invalid Prices (<=0): {(df['price_inr'] <= 0).sum()}")
    print(f"Min Price: Rs {df['price_inr'].min()}")
    print(f"Max Price: Rs {df['price_inr'].max()}")
    print(f"Average Price: Rs {df['price_inr'].mean():.2f}")
    print(f"Unique Categories ({df['category'].nunique()}): {sorted(df['category'].unique().tolist())}")
    print(f"Unique Brands ({df['brand'].nunique()}): {sorted(df['brand'].unique().tolist())}")
    print(f"Unique Genders ({df['gender'].nunique()}): {sorted(df['gender'].unique().tolist())}")
    print(f"Unique Colors ({df['color'].nunique()}): {sorted(df['color'].unique().tolist())}")
    print(f"Unique Fits ({df['fit'].nunique()}): {sorted(df['fit'].unique().tolist())}")
    print(f"Unique Materials ({df['material'].nunique()}): {sorted(df['material'].unique().tolist())}")

    # Check products above 6000, 10000, 20000
    above_6k = (df['price_inr'] > 6000).sum()
    above_10k = (df['price_inr'] > 10000).sum()
    above_20k = (df['price_inr'] > 20000).sum()

    print(f"Products > Rs 6,000: {above_6k}")
    print(f"Products > Rs 10,000: {above_10k}")
    print(f"Products > Rs 20,000: {above_20k}")

def run_phase_4_query_parser_test():
    print("\n" + "=" * 60)
    print("PHASE 4 — QUERY PARSER TEST")
    print("=" * 60)
    parser = QueryParser()
    test_queries = [
        "blue slim fit jeans under 2000",
        "black casual t-shirts below 1500",
        "women's cotton dresses for summer",
        "men's oversized hoodies under 2500",
        "Nike shoes under 5000",
        "red cotton shirt for women",
        "comfortable clothes for travel",
        "formal wear under 3000",
        "I want something affordable",
        "products below Rs 20000"
    ]

    for q in test_queries:
        parsed = parser.parse(q)
        print(f"\nQuery: '{q}'")
        print(f"  Parsed: {json.dumps(parsed, indent=4)}")

def run_phase_5_and_6_retrieval_and_filter_test():
    print("\n" + "=" * 60)
    print("PHASE 5 & 6 — RETRIEVAL & HARD FILTER TEST")
    print("=" * 60)
    retriever = ProductRetriever()

    test_queries = [
        # 1. Exact & Price
        ("blue slim fit jeans under 2000", 2000),
        ("black casual t-shirts below 1500", 1500),
        ("women's cotton dresses for summer", None),
        ("men's oversized hoodies under 2500", 2500),
        # 2. Category / Brand / Gender
        ("jeans", None),
        ("dresses", None),
        ("shirts", None),
        ("Nike shoes under 5000", 5000),
        ("red cotton shirt for women", None),
        ("Zara jackets", None),
        # 3. Budget Specific
        ("trousers under 1000", 1000),
        ("jackets under 5000", 5000),
        ("sweaters under 10000", 10000),
        ("ethnic wear under 20000", 20000),
        # 4. Semantic / Vague
        ("comfortable clothes for travel", None),
        ("formal wear under 3000", 3000),
        ("I want something affordable", None),
        ("activewear for gym", None),
        ("summer party dress", None),
        ("winter wool cardigan", None)
    ]

    latencies = []
    hard_filter_failures = []

    for q, max_b in test_queries:
        t0 = time.time()
        # Test without UI filter overrides (pure text query parsing)
        res = retriever.retrieve(q, ui_filters=None, top_k=4)
        t1 = time.time()
        lat = (t1 - t0) * 1000
        latencies.append(lat)

        print(f"\nQuery: '{q}' | Latency: {lat:.1f}ms | Count: {res['count']}")
        for item in res['items']:
            price = item['price_inr']
            print(f"  - [{item['brand']}] {item['product_name']} | Rs {price} | Cat: {item['category']} | Match: {item['match_percentage']}%")
            
            # Check price constraint if specified in query or tuple
            query_max_p = res['parsed_query'].get('max_price') or max_b
            if query_max_p is not None and price > query_max_p:
                msg = f"FAIL — HARD FILTER VIOLATION: Product '{item['product_name']}' price Rs {price} > max budget Rs {query_max_p}"
                print(f"    FAIL -- HARD FILTER VIOLATION: {msg}")
                hard_filter_failures.append((q, item['product_name'], price, query_max_p))

    print("\n--- TEST WITH UI SIDEBAR FILTERS ACTIVE (Simulating app.py UI behavior) ---")
    ui_filters_5000 = {"max_price": 5000}
    res_ui = retriever.retrieve("blue slim fit jeans under 2000", ui_filters=ui_filters_5000, top_k=4)
    print("Query: 'blue slim fit jeans under 2000' with UI max_price=5000 override:")
    print(f"Merged Parsed Query: {res_ui['parsed_query']}")
    for item in res_ui['items']:
        price = item['price_inr']
        print(f"  - [{item['brand']}] {item['product_name']} | Rs {price}")
        if price > 2000:
            print(f"    FAIL -- HARD FILTER VIOLATION IN UI MODE: Rs {price} > Rs 2000")
            hard_filter_failures.append(("blue slim fit jeans under 2000 (UI mode)", item['product_name'], price, 2000))

    return latencies, hard_filter_failures

def run_phase_8_grounding_test():
    print("\n" + "=" * 60)
    print("PHASE 8 — RAG GROUNDING & HALLUCINATION TEST")
    print("=" * 60)
    retriever = ProductRetriever()
    agent = RAGAgent()

    # Test standard recommendation summary
    res = retriever.retrieve("women's cotton dresses for summer", top_k=4)
    explanation = agent.generate_recommendation_explanation("women's cotton dresses for summer", res['items'], res['parsed_query'])
    print(f"Source: {explanation['source']}")
    safe_explanation = explanation['text'].encode('ascii', errors='replace').decode('ascii')
    print(f"Explanation Text:\n{safe_explanation}\n")

    # Test Impossible Query
    impossible_q = "Luxury Rolex diamond watch under 50"
    res_imp = retriever.retrieve(impossible_q, top_k=4)
    exp_imp = agent.generate_recommendation_explanation(impossible_q, res_imp['items'], res_imp['parsed_query'])
    safe_exp_imp = exp_imp['text'].encode('ascii', errors='replace').decode('ascii')
    print(f"Impossible Query: '{impossible_q}'")
    print(f"Retrieved Count: {res_imp['count']}")
    print(f"Message: {res_imp['message']}")
    print(f"Explanation: {safe_exp_imp}\n")

def run_phase_9_edge_case_test():
    print("\n" + "=" * 60)
    print("PHASE 9 — EDGE CASE TESTING")
    print("=" * 60)
    retriever = ProductRetriever()

    edge_cases = [
        ("1. Empty query", ""),
        ("2. Very short query", "a"),
        ("3. Gibberish query", "xyz123abc!!!"),
        ("4. Non-clothing query", "iPhone 15 Pro Max 256GB"),
        ("5. Extremely low budget", "jeans under 10"),
        ("6. Extremely high budget", "shirts under 500000"),
        ("7. Unknown brand", "SuperFancyBrandX t-shirt"),
        ("8. Unknown category", "space suit for Mars"),
        ("9. Misspelled terms", "blu slm ft jens undr 2000"),
        ("10. Conflicting filters", "Men's silk dress for summer"),
        ("11. Special characters", "black @#$% t-shirt <2000>!"),
        ("12. Very long query", "I am looking for a beautiful blue color comfortable slim fit cotton denim jeans product for men under 2000 INR for wearing in summer season"),
        ("13. Repeated query", "blue slim fit jeans under 2000")
    ]

    for label, q in edge_cases:
        try:
            res = retriever.retrieve(q, top_k=4)
            print(f"{label}: '{q}' -> Count: {res['count']}, Msg: '{res['message']}'")
        except Exception as e:
            print(f"[FAIL] {label}: '{q}' -> CRASHED with Exception: {e}")

def save_benchmark_report(latencies, hard_filter_failures):
    print("\n" + "=" * 60)
    print("PHASE 10 — SAVING BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "total_queries_tested": len(latencies),
            "avg_retrieval_latency_ms": round(avg_latency, 2),
            "p95_retrieval_latency_ms": round(p95_latency, 2),
            "hard_filter_violations": len(hard_filter_failures),
            "hard_filter_accuracy_pct": 100.0 if len(hard_filter_failures) == 0 else round((1 - len(hard_filter_failures) / max(1, len(latencies))) * 100, 2)
        },
        "violations": hard_filter_failures
    }

    report_path = Path(__file__).parent / "evaluation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation benchmark saved successfully to {report_path}")
    print(f"   - Average Latency: {report['metrics']['avg_retrieval_latency_ms']} ms")
    print(f"   - P95 Latency:     {report['metrics']['p95_retrieval_latency_ms']} ms")
    print(f"   - Filter Accuracy: {report['metrics']['hard_filter_accuracy_pct']}% ({len(hard_filter_failures)} violations)")

if __name__ == "__main__":
    run_phase_3_dataset_test()
    run_phase_4_query_parser_test()
    lats, fails = run_phase_5_and_6_retrieval_and_filter_test()
    run_phase_8_grounding_test()
    run_phase_9_edge_case_test()
    save_benchmark_report(lats, fails)

