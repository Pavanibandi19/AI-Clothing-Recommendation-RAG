import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import ProductRetriever
from src.query_parser import QueryParser
from src.ranking import HybridRanker
from src.data_loader import DataLoader
from src.agent import RAGAgent

def audit_queries():
    retriever = ProductRetriever()
    queries = [
        "blue slim fit jeans under 2000",
        "black casual t-shirts below 1500",
        "women's cotton dresses for summer",
        "men's oversized hoodies under 2500",
        "jeans",
        "dresses",
        "shirts"
    ]
    
    # Default UI filter dict as generated in app/app.py:
    # Notice app.py line 205: min_value=0, max_value=20000, value=(0, 20000)
    # line 286: "max_price": price_range[1] if price_range[1] < 6000 else None
    
    print("=== TESTING QUERIES WITH NO SIDEBAR OVERRIDES ===")
    for q in queries:
        res = retriever.retrieve(q, ui_filters=None, top_k=4)
        print(f"\nQuery: '{q}' -> Count: {res['count']}")
        print(f"Parsed: {res['parsed_query']}")
        for item in res['items']:
            print(f"  - [{item['brand']}] {item['product_name']} | Price: Rs {item['price_inr']} | Cat: {item['category']} | Gender: {item['gender']} | Score: {item['hybrid_score']}")
            
    print("\n=== TESTING QUERIES WITH APP.PY DEFAULT UI FILTERS (price_range=(0, 20000), < 6000 check) ===")
    # When slider is (0, 5000):
    ui_filters_5000 = {
        "category": None, "gender": None, "color": None, "fit": None, "material": None,
        "min_price": None, "max_price": 5000
    }
    res = retriever.retrieve("blue slim fit jeans under 2000", ui_filters=ui_filter_dict_5000, top_k=4) if 'ui_filter_dict_5000' in locals() else retriever.retrieve("blue slim fit jeans under 2000", ui_filters=ui_filters_5000, top_k=4)
    print(f"\nQuery: 'blue slim fit jeans under 2000' with UI max_price=5000:")
    print(f"Parsed & Merged: {res['parsed_query']}")
    for item in res['items']:
        print(f"  - [{item['brand']}] {item['product_name']} | Price: Rs {item['price_inr']}")

if __name__ == "__main__":
    audit_queries()
