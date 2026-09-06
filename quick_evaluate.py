"""
Quick Evaluation Report for AI Clothing Recommendation System (RECOMAI)
Evaluates structured chunk-based RAG retrieval, constraint adherence, deduplication and grounding.
"""

import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.retrieval import ProductRetriever
from src.data_loader import DataLoader
from src.config import DATASET_PATH


class QuickEvaluation:
    def __init__(self):
        self.retriever = ProductRetriever()
        self.loader = DataLoader(DATASET_PATH)
        self.results = {
            "timestamp": time.time(),
            "test_cases": [],
            "summary": {}
        }
    
    def test_budget_constraint(self):
        """Verify no product exceeds budget constraint"""
        logger.info("Testing Budget Constraint Enforcement...")
        query = "jeans under 2000"
        results = self.retriever.retrieve(query, top_k=10)
        
        violations = [p for p in results["items"] if p["price_inr"] > 2000]
        passed = len(violations) == 0 and len(results["items"]) > 0
        
        self.results["test_cases"].append({
            "test": "Budget Constraint",
            "passed": passed,
            "details": f"{len(results['items'])} items returned, max price: {max([p['price_inr'] for p in results['items']]) if results['items'] else 0}"
        })
        return passed
    
    def test_no_match_handling(self):
        """Verify graceful empty result handling for unmatchable budget"""
        logger.info("Testing No-Match Handling...")
        query = "jackets under 100"
        results = self.retriever.retrieve(query, top_k=4)
        
        passed = len(results["items"]) == 0
        self.results["test_cases"].append({
            "test": "No-Match Handling",
            "passed": passed,
            "details": f"Returns empty: {passed}, Message: {results.get('message', '')}"
        })
        return passed

    def test_hallucination_guard_out_of_catalog(self):
        """Verify system does NOT fabricate products for out-of-catalog items (e.g. luxury watch)"""
        logger.info("Testing Out-of-Catalog / Hallucination Guard...")
        query = "luxury Rolex watch under 50"
        results = self.retriever.retrieve(query, top_k=4)
        
        passed = len(results["items"]) == 0
        self.results["test_cases"].append({
            "test": "Hallucination Guard (Out-of-Catalog)",
            "passed": passed,
            "details": f"Correctly returned 0 items for impossible out-of-catalog query: {passed}"
        })
        return passed
    
    def test_category_filtering(self):
        """Verify category constraints"""
        logger.info("Testing Category Filtering...")
        query = "blue t-shirts under 1000"
        results = self.retriever.retrieve(query, top_k=5)
        
        violations = [p for p in results["items"] if p["category"].lower() != "t-shirts"]
        passed = len(violations) == 0 and len(results["items"]) > 0
        
        self.results["test_cases"].append({
            "test": "Category Filtering",
            "passed": passed,
            "details": f"{len(results['items'])} items, all correct category: {passed}"
        })
        return passed
    
    def test_gender_filtering(self):
        """Verify gender constraints with Unisex fallback"""
        logger.info("Testing Gender Filtering...")
        query = "women's hoodies under 2000"
        results = self.retriever.retrieve(query, top_k=5)
        
        violations = [p for p in results["items"] if p["gender"].lower() not in ["women", "unisex"]]
        passed = len(violations) == 0 and len(results["items"]) > 0
        
        self.results["test_cases"].append({
            "test": "Gender Filtering",
            "passed": passed,
            "details": f"{len(results['items'])} items, all correct gender: {passed}"
        })
        return passed
    
    def test_vector_search_availability(self):
        """Check ChromaDB chunk collection status"""
        logger.info("Testing Vector Search Status...")
        query = "casual cotton shirts"
        results = self.retriever.retrieve(query, top_k=4)
        
        chroma_active = results.get("vector_search_active", False)
        chroma_size = results.get("chroma_count", 0)
        
        self.results["test_cases"].append({
            "test": "Vector Search Availability",
            "passed": chroma_size > 0 and chroma_active,
            "details": f"ChromaDB Active: {chroma_active}, Indexed Chunks: {chroma_size}"
        })
        return chroma_size > 0 and chroma_active
    
    def test_metadata_and_price_range(self):
        """Verify price range constraints: formal linen shirts above 2000"""
        logger.info("Testing Price Range & Material Constraints...")
        query = "formal linen shirts above 2000"
        results = self.retriever.retrieve(query, top_k=4)
        
        all_valid = all(
            p["price_inr"] >= 2000 and
            p["category"].lower() == "shirts"
            for p in results["items"]
        )
        passed = all_valid and len(results["items"]) > 0
        
        self.results["test_cases"].append({
            "test": "Price Range & Category (Above 2000)",
            "passed": passed,
            "details": f"{len(results['items'])} items returned, all prices >= 2000: {passed}"
        })
        return passed

    def test_broad_budget_query(self):
        """Verify 'something under 1000' returns affordable items within budget"""
        logger.info("Testing Broad Budget Query (under 1000)...")
        query = "something under 1000"
        results = self.retriever.retrieve(query, top_k=4)
        
        all_within_budget = all(p["price_inr"] <= 1000 for p in results["items"])
        passed = all_within_budget and len(results["items"]) > 0
        
        self.results["test_cases"].append({
            "test": "Broad Budget Query (under 1000)",
            "passed": passed,
            "details": f"{len(results['items'])} items returned, all <= 1000: {passed}"
        })
        return passed

    def test_chunk_retrieval_and_product_deduplication(self):
        """Verify that chunk-level retrieval aggregates into unique products without duplicate cards"""
        logger.info("Testing Chunk Retrieval & Product Deduplication...")
        query = "blue slim fit jeans under 2000"
        results = self.retriever.retrieve(query, top_k=4)
        
        items = results["items"]
        pids = [p["product_id"] for p in items]
        unique_pids = set(pids)
        passed = len(pids) == len(unique_pids) and len(items) > 0
        
        diag = results.get("diagnostics", {})
        chunks_retrieved = diag.get("retrieved_chunks_count", 0)
        
        self.results["test_cases"].append({
            "test": "Chunk Retrieval & Deduplication",
            "passed": passed,
            "details": f"{chunks_retrieved} chunks retrieved -> {len(items)} unique products returned (0 duplicates)"
        })
        return passed
    
    def test_latency(self):
        """Benchmark retrieval performance"""
        logger.info("Testing Retrieval Latency...")
        query = "black slim fit jeans under 2500"
        
        start = time.time()
        results = self.retriever.retrieve(query, top_k=4)
        latency = time.time() - start
        
        passed = latency < 2.0  # Should complete in < 2 seconds
        
        self.results["test_cases"].append({
            "test": "Retrieval Latency",
            "passed": passed,
            "details": f"Completed in {latency:.4f}s ({latency*1000:.1f}ms) (target: < 2s), {len(results['items'])} items"
        })
        return passed
    
    def run_all(self):
        """Execute all tests"""
        logger.info("=" * 70)
        logger.info("QUICK EVALUATION - RECOMAI CHUNK-BASED RAG SYSTEM")
        logger.info("=" * 70)
        
        tests = [
            self.test_budget_constraint,
            self.test_no_match_handling,
            self.test_hallucination_guard_out_of_catalog,
            self.test_category_filtering,
            self.test_gender_filtering,
            self.test_vector_search_availability,
            self.test_metadata_and_price_range,
            self.test_broad_budget_query,
            self.test_chunk_retrieval_and_product_deduplication,
            self.test_latency,
        ]
        
        passed = 0
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"ERROR in {test_func.__name__}: {e}")
                self.results["test_cases"].append({
                    "test": test_func.__name__,
                    "passed": False,
                    "error": str(e)
                })
        
        total = len(tests)
        self.results["summary"] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{(passed/total*100):.0f}%",
            "status": "PASS" if passed == total else "PARTIAL"
        }
        
        logger.info("=" * 70)
        logger.info(f"Results: {passed}/{total} tests passed ({self.results['summary']['pass_rate']})")
        logger.info(f"Status: {self.results['summary']['status']}")
        logger.info("=" * 70)
        
        return self.results
    
    def save(self):
        """Save results"""
        with open("evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Results saved to evaluation_results.json")
    
    def print_report(self):
        """Print summary"""
        print("\n" + "=" * 70)
        print("RECOMAI RAG EVALUATION SUMMARY")
        print("=" * 70)
        for test in self.results["test_cases"]:
            status = "[PASS]" if test.get("passed") else "[FAIL]"
            print(f"{status} {test.get('test', 'Unknown')}")
            print(f"   --> {test.get('details', '')}")
        
        print("\n" + "=" * 70)
        summary = self.results["summary"]
        print(f"RESULTS: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']})")
        print(f"STATUS:  {summary['status']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluator = QuickEvaluation()
    evaluator.run_all()
    evaluator.save()
    evaluator.print_report()
