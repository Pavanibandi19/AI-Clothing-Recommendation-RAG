import time
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# 1. Measure Streamlit App Boot / Import Time
t0 = time.time()
import app.app
app_boot_time = time.time() - t0

# 2. Measure Retrieval Engine Init
t0 = time.time()
from src.retrieval import ProductRetriever
retriever = ProductRetriever()
retriever_init_time = time.time() - t0

# 3. Measure Search Queries across different categories and constraints
queries = [
    "blue slim fit jeans under 2000",
    "black casual t-shirts below 1500",
    "women cotton dress for summer",
    "mens oversized hoodies under 2500"
]

retrieval_timings = []
for q in queries:
    t0 = time.time()
    res = retriever.retrieve(q)
    dt = time.time() - t0
    retrieval_timings.append((q, dt, len(res["items"]), res))

# 4. Measure AI Stylist Rationale Generation
from src.agent import RAGAgent
agent = RAGAgent()
sample_query, _, _, sample_res = retrieval_timings[0]
t0 = time.time()
expl = agent.generate_recommendation_explanation(sample_query, sample_res["items"], sample_res["parsed_query"])
agent_time = time.time() - t0

# 5. Measure Cached AI Stylist Rationale Generation
t0 = time.time()
expl_cached = agent.generate_recommendation_explanation(sample_query, sample_res["items"], sample_res["parsed_query"])
agent_cached_time = time.time() - t0

print("\n================ SYSTEM RESPONSE TIME REPORT ================\n")
print(f"1. Streamlit App Boot Time:          {app_boot_time:.3f} s  ({app_boot_time*1000:.1f} ms)")
print(f"2. Retrieval Engine Ready State:     {retriever_init_time:.3f} s  ({retriever_init_time*1000:.1f} ms)")
print("\n3. Query-to-Product Search Times:")
for q, dt, count, _ in retrieval_timings:
    print(f"   • \"{q}\" -> {count} products in {dt*1000:.2f} ms ({dt:.4f} s)")

avg_retrieval = sum(t[1] for t in retrieval_timings) / len(retrieval_timings)
print(f"\n   ==> Average Product Retrieval Latency: {avg_retrieval*1000:.2f} ms ({avg_retrieval:.4f} s)")

print(f"\n4. AI Stylist Rationale Generation:   {agent_time*1000:.2f} ms ({agent_time:.3f} s) [{expl.get('source')}]")
print(f"5. AI Stylist Rationale (Cached):     {agent_cached_time*1000:.3f} ms ({agent_cached_time:.5f} s)")
print("\n=============================================================\n")
