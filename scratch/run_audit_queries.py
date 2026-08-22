import sys
import io
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.query_parser import QueryParser
from src.retrieval import ProductRetriever
from src.agent import RAGAgent

qp = QueryParser()
retriever = ProductRetriever()
agent = RAGAgent()

queries = [
    "blue slim fit jeans under 2000",
    "black casual t-shirts below 1500",
    "women's cotton dresses for summer",
    "men's oversized hoodies under 2500",
    "blue jeans under \u20b92000",
    "black shirts under \u20b91500",
    "women's dresses below 3000",
    "cotton shirts",
    "party wear dresses",
    "casual clothes for summer",
    "jeans under \u20b92,000",
    "jeans below rs. 2000",
    "jeans below INR 2000",
    "blue jeans under 2000 rupees",
    "black t-shirts below 1500",
    "black tshirts below 1500",
    "women's dresses",
    "womens dresses",
    "men's jeans",
    "mens jeans",
    "party wear",
    "casual wear for summer",
    "cotton shirts under \u20b92000",
    "dresses below rs. 3000",
    "dresses under INR 3000",
    "shirts up to 2500",
    "jeans less than 2000",
    "JEANS UNDER 2000",
    "Jeans under 2000"
]

print("=== QUERY PARSER TESTS ===")
for q in queries:
    parsed = qp.parse(q)
    extracted = {k: v for k, v in parsed.items() if v is not None and k not in ("cleaned_query", "raw_query")}
    print(f"'{q}' -> {extracted}")

