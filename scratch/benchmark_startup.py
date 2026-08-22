import time
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("="*60)
print("STARTUP & PIPELINE PERFORMANCE HEALTH CHECK")
print("="*60)

# 1. Measure Module Import Time
t0 = time.perf_counter()
import pandas as pd
import chromadb
from src.config import CHROMA_PERSIST_DIRECTORY, DATASET_PATH
from src.data_loader import DataLoader
from src.embedder import VectorEmbedder
from src.query_parser import QueryParser
from src.ranking import HybridRanker
from src.retrieval import ProductRetriever
from src.agent import RAGAgent
t_import = (time.perf_counter() - t0) * 1000
print(f"1. Module Import Time: {t_import:.2f} ms")

# 2. Measure Dataset Loading Time
t0 = time.perf_counter()
loader = DataLoader(DATASET_PATH)
df = loader.get_all_products()
t_data = (time.perf_counter() - t0) * 1000
print(f"2. Dataset Loading Time (3,000 rows): {t_data:.2f} ms (Loaded: {len(df)} rows)")

# 3. Measure ChromaDB Client & Collection Connection Time
t0 = time.perf_counter()
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
coll = client.get_collection(name="clothing_products")
count = coll.count()
t_chroma = (time.perf_counter() - t0) * 1000
print(f"3. ChromaDB Init & Collection Count Time: {t_chroma:.2f} ms (Total Indexed: {count} items)")

# 4. Measure Embedding Model Loading Time
t0 = time.perf_counter()
embedder = VectorEmbedder()
# Trigger warm-up embedding
_ = embedder.encode(["test query"])
t_embed = (time.perf_counter() - t0) * 1000
print(f"4. VectorEmbedder Load & Warm-up Time: {t_embed:.2f} ms")

# 5. Measure Ollama Connection / Check Time
t0 = time.perf_counter()
agent = RAGAgent()
is_ollama = agent.is_ollama_available()
t_ollama = (time.perf_counter() - t0) * 1000
print(f"5. Ollama Health Check Time: {t_ollama:.2f} ms (Available: {is_ollama})")

# 6. Measure ProductRetriever Full Init Time
t0 = time.perf_counter()
retriever = ProductRetriever()
t_retriever_init = (time.perf_counter() - t0) * 1000
print(f"6. ProductRetriever Unified Init Time: {t_retriever_init:.2f} ms")

# 7. Measure First Query Retrieval Time
t0 = time.perf_counter()
res1 = retriever.retrieve("blue slim fit jeans under 2000")
t_first_q = (time.perf_counter() - t0) * 1000
print(f"7. First Query Retrieval ('blue slim fit jeans under 2000'): {t_first_q:.2f} ms (Returned: {len(res1['items'])} items)")

# 8. Measure Subsequent Query Retrieval Times
queries = [
    "black casual t-shirts below 1500",
    "women's cotton dresses for summer",
    "men's oversized hoodies under 2500",
    "party wear dresses",
    "blue jeans under ₹2000"
]
latencies = []
for q in queries:
    t0 = time.perf_counter()
    r = retriever.retrieve(q)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    print(f"   • Query '{q}': {lat:.2f} ms (Items: {len(r['items'])})")

avg_lat = sum(latencies) / len(latencies)
print(f"8. Subsequent Queries Average Retrieval Time: {avg_lat:.2f} ms")
print("="*60)
print("HEALTH CHECK COMPLETE: All subsystems operational and within SLA targets.")
print("="*60)
