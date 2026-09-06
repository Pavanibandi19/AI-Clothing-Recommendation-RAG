import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import ProductRetriever
from src.agent import RAGAgent
from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.rag.chunker import ProductChunker
from src.embedder import VectorEmbedder
import chromadb
from src.config import (
    CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME,
    DATASET_PATH, EMBEDDING_MODEL_NAME
)

print("=== STARTING QA AUDIT SCRIPT ===")

retriever = ProductRetriever()
agent = RAGAgent()
parser = QueryParser()
loader = DataLoader()
chunker = ProductChunker()
embedder = VectorEmbedder()

# 1. Inspect Dataset & Brands
df = loader.get_all_products()
brands = df["brand"].dropna().unique().tolist()
print(f"Dataset Size: {len(df)} rows")
print(f"Sample Brands: {brands[:10]}")

# 2. Inspect Vector DB & Chunking
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
coll = client.get_collection(CHROMA_COLLECTION_NAME)
coll_count = coll.count()
print(f"ChromaDB Collection '{CHROMA_COLLECTION_NAME}' count: {coll_count}")

# Sample chunk inspect
sample_chunks = coll.get(limit=3, include=["metadatas", "documents"])
print(f"Sample stored chunk IDs: {sample_chunks['ids']}")
print(f"Sample chunk document: {sample_chunks['documents'][0]}")
print(f"Sample chunk metadata: {sample_chunks['metadatas'][0]}")

# 3. Test Mandatory Queries
test_queries = [
    "blue slim fit jeans under 2000",
    "black casual t-shirts below 1500",
    "women's cotton dresses for summer",
    "men's oversized hoodies under 2500",
    "formal linen shirts above 2000",
    "running shoes under 3000",
    "winter jackets for men",
    "cotton shirts for women",
    f"shirts from {brands[0]}",
    "between 2000 and 5000",
    "below ₹1500"
]

print("\n--- TESTING MANDATORY RETRIEVAL QUERIES ---")
retrieval_results = {}
for q in test_queries:
    t0 = time.perf_counter()
    res = retriever.retrieve(q, top_k=4)
    t1 = time.perf_counter()
    retrieval_results[q] = {
        "count": res["count"],
        "items": [
            {
                "product_id": item["product_id"],
                "name": item["product_name"],
                "brand": item["brand"],
                "price": item["price_inr"],
                "category": item["category"],
                "gender": item["gender"],
                "color": item.get("color"),
                "fit": item.get("fit"),
                "material": item.get("material"),
                "similarity": item.get("semantic_similarity"),
                "hybrid_score": item.get("hybrid_score")
            } for item in res["items"]
        ],
        "parsed_query": res["parsed_query"],
        "latency_ms": round((t1 - t0) * 1000, 2),
        "message": res.get("message")
    }
    print(f"Query: '{q}' -> {res['count']} items returned in {retrieval_results[q]['latency_ms']}ms")
    for it in res["items"]:
        print(f"   * [{it['product_id']}] {it['brand']} {it['product_name']} | ₹{it['price_inr']} | Cat: {it['category']} | Gen: {it['gender']} | Mat: {it.get('material')} | Fit: {it.get('fit')} | Col: {it.get('color')} | Sim: {it.get('semantic_similarity')}")

# 4. Test Query Parsing Edge Cases
parse_cases = [
    "under 2000",
    "below ₹1500",
    "above 2000",
    "between 2000 and 5000",
    "less than 3500.50",
    "blue cotton shirt for girls",
    "oversized hoodies for boys",
    "unisex winter jackets",
    "party wear dresses",
    "running shoes",
    "cotton shirts under ₹2,000",
    "maxi dresses below rs. 3000",
    "shirts with 0 price",
    "jackets under -500"
]
print("\n--- TESTING QUERY PARSING ---")
parsed_output = {}
for q in parse_cases:
    p = parser.parse(q)
    is_cloth = parser.is_clothing_query(q)
    parsed_output[q] = {"parsed": p, "is_clothing_query": is_cloth}
    print(f"Query: '{q}' => Clothing: {is_cloth} => Parsed: min={p['min_price']}, max={p['max_price']}, cat={p['category']}, gen={p['gender']}, col={p['color']}, fit={p['fit']}, mat={p['material']}, season={p['season']}")

# 5. Error Testing Suite
error_cases = [
    "",
    "   ",
    "xyz random gibberish 123",
    "unsupported spacecraft rocket",
    "shoes with price 0",
    "jackets under -100",
    "jackets under 999999999",
    "dresses with €$%^&*()!~`",
    "a" * 1000, # Long query
    "jackets under 10" # Impossible price (no match)
]
print("\n--- TESTING ERROR CASES ---")
error_results = {}
for q in error_cases:
    try:
        is_cloth = parser.is_clothing_query(q)
        res = retriever.retrieve(q, top_k=4)
        error_results[q] = {
            "status": "PASS",
            "is_clothing_query": is_cloth,
            "count": res["count"],
            "message": res.get("message")
        }
        print(f"Error test '{q[:30]}...' -> Handled gracefully. Count={res['count']}")
    except Exception as exc:
        error_results[q] = {
            "status": "FAIL",
            "exception": str(exc)
        }
        print(f"Error test '{q[:30]}...' -> CRASHED: {exc}")

# 6. Trace Complete End-to-End Query
trace_query = "women's cotton dresses for summer"
print(f"\n--- TRACING E2E QUERY: '{trace_query}' ---")
t_parsed = parser.parse(trace_query)
print(f"1. Parsed: {t_parsed}")

q_emb = embedder.encode([trace_query])
print(f"2. Query Embedding: shape={q_emb.shape}, norm={np.linalg.norm(q_emb):.4f}" if "np" in dir() else f"2. Query Embedding: shape={q_emb.shape}")

# Vector search raw chunks
raw_search = coll.query(query_embeddings=q_emb.tolist(), n_results=10)
print(f"3. ChromaDB raw retrieved chunk count: {len(raw_search['ids'][0])}")
for cid, dist, meta, doc in zip(raw_search['ids'][0][:5], raw_search['distances'][0][:5], raw_search['metadatas'][0][:5], raw_search['documents'][0][:5]):
    print(f"   Chunk: {cid} | Dist: {dist:.4f} | Product: {meta.get('product_id')} | Type: {meta.get('chunk_type')}")
    print(f"      Text: {doc[:100]}...")

res_trace = retriever.retrieve(trace_query, top_k=2)
print(f"4. Retrieval + Hard Filter + Reranking -> {len(res_trace['items'])} items")
for item in res_trace['items']:
    print(f"   Item: [{item['product_id']}] {item['brand']} {item['product_name']} | ₹{item['price_inr']} | Score: {item['hybrid_score']} | Matched chunks: {len(item.get('matched_chunks', []))}")

explanation = agent.generate_recommendation_explanation(trace_query, res_trace['items'], t_parsed)
print(f"5. Agent Explanation:\nSource: {explanation['source']}\nText:\n{explanation['text']}")

# Save results for reporting
with open(PROJECT_ROOT / "scratch" / "qa_audit_results.json", "w") as f:
    json.dump({
        "retrieval": retrieval_results,
        "query_parsing": parsed_output,
        "error_tests": error_results
    }, f, indent=2)

print("\n=== QA AUDIT COMPLETE ===")
