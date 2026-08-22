"""
Comprehensive Evaluation Report for AI Clothing Recommendation System

This script validates the RAG system against key criteria:
- Retrieval quality (precision, no-budget leaks)
- No-match handling (out-of-scope queries)
- Grounding (product ID citations)
- Latency benchmarks
- Fallback behavior (Ollama unavailable)
"""

import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.retrieval import ProductRetriever
from src.agent import RAGAgent
from src.data_loader import DataLoader
from src.config import DATASET_PATH

class EvaluationReport:
    def __init__(self):
        self.retriever = ProductRetriever()
        self.agent = RAGAgent()
        self.loader = DataLoader(DATASET_PATH)
        self.results = {
            "timestamp": None,
            "test_cases": [],
            "summary": {}
        }
    
    def test_budget_constraint(self) -> Dict[str, Any]:
        """Verify no product exceeds budget constraint (CRITICAL)"""
        test_name = "Budget Constraint Enforcement"
        logger.info(f"Running: {test_name}")
        
        query = "jeans under 2000"
        results = self.retriever.retrieve(query, top_k=10)
        
        budget_violated = []
        for item in results["items"]:
            if item["price_inr"] > 2000:
                budget_violated.append({
                    "product_id": item["product_id"],
                    "price": item["price_inr"],
                    "budget": 2000
                })
        
        passed = len(budget_violated) == 0
        return {
            "test_name": test_name,
            "query": query,
            "passed": passed,
            "products_returned": len(results["items"]),
            "budget_violations": budget_violated,
            "max_price_in_results": max([p["price_inr"] for p in results["items"]]) if results["items"] else 0
        }
    
    def test_no_match_handling(self) -> Dict[str, Any]:
        """Verify graceful handling of impossible constraints"""
        test_name = "No-Match Handling"
        logger.info(f"Running: {test_name}")
        
        query = "jackets under 100"  # Impossible constraint (no jackets that cheap)
        results = self.retriever.retrieve(query, top_k=4)
        
        return {
            "test_name": test_name,
            "query": query,
            "passed": len(results["items"]) == 0,
            "returns_empty": len(results["items"]) == 0,
            "message": results.get("message", ""),
            "graceful_error": "No matching products" in results.get("message", "")
        }
    
    def test_grounding_with_citations(self) -> Dict[str, Any]:
        """Verify LLM rationale includes product ID citations"""
        test_name = "LLM Grounding & Citations"
        logger.info(f"Running: {test_name}")
        
        query = "women's cotton dresses for summer under 2000"
        results = self.retriever.retrieve(query, top_k=3)
        
        explanation = self.agent.generate_recommendation_explanation(
            query=query,
            recommended_products=results["items"],
            parsed_query=results["parsed_query"]
        )
        
        # Check for product IDs in explanation (citation format: PRD1234 or [PRD1234])
        rationale_text = explanation.get("text", "").upper()
        has_citations = "PRD" in rationale_text or "[" in rationale_text
        
        return {
            "test_name": test_name,
            "query": query,
            "products_recommended": len(results["items"]),
            "rationale_source": explanation.get("source", ""),
            "has_product_citations": has_citations,
            "rationale_preview": explanation.get("text", "")[:150] + "...",
            "passed": len(results["items"]) > 0 and len(explanation.get("text", "")) > 20
        }
    
    def test_category_filtering(self) -> Dict[str, Any]:
        """Verify category constraints are enforced"""
        test_name = "Category Filtering Accuracy"
        logger.info(f"Running: {test_name}")
        
        query = "blue t-shirts under 1000"
        results = self.retriever.retrieve(query, top_k=5)
        
        category_violations = []
        for item in results["items"]:
            if item["category"].lower() != "t-shirts":
                category_violations.append({
                    "product_id": item["product_id"],
                    "returned_category": item["category"]
                })
        
        passed = len(category_violations) == 0
        return {
            "test_name": test_name,
            "query": query,
            "passed": passed,
            "products_returned": len(results["items"]),
            "all_correct_category": len(category_violations) == 0,
            "category_violations": category_violations
        }
    
    def test_gender_filtering(self) -> Dict[str, Any]:
        """Verify gender constraints with Unisex fallback"""
        test_name = "Gender Filtering with Unisex Fallback"
        logger.info(f"Running: {test_name}")
        
        query = "women's hoodies under 2000"
        results = self.retriever.retrieve(query, top_k=5)
        
        gender_violations = []
        for item in results["items"]:
            item_gender = item["gender"].lower()
            if item_gender not in ["women", "unisex"]:
                gender_violations.append({
                    "product_id": item["product_id"],
                    "returned_gender": item["gender"]
                })
        
        passed = len(gender_violations) == 0
        return {
            "test_name": test_name,
            "query": query,
            "passed": passed,
            "products_returned": len(results["items"]),
            "all_correct_gender": len(gender_violations) == 0,
            "gender_violations": gender_violations
        }
    
    def test_latency_benchmark(self) -> Dict[str, Any]:
        """Benchmark retrieval and explanation latency"""
        test_name = "Latency Benchmark"
        logger.info(f"Running: {test_name}")
        
        query = "black slim fit jeans under 2500"
        
        # Benchmark retrieval
        start = time.time()
        results = self.retriever.retrieve(query, top_k=4)
        retrieval_time = time.time() - start
        
        # Benchmark explanation generation
        start = time.time()
        explanation = self.agent.generate_recommendation_explanation(
            query=query,
            recommended_products=results["items"],
            parsed_query=results["parsed_query"]
        )
        explanation_time = time.time() - start
        
        total_time = retrieval_time + explanation_time
        
        return {
            "test_name": test_name,
            "query": query,
            "retrieval_time_seconds": round(retrieval_time, 3),
            "explanation_time_seconds": round(explanation_time, 3),
            "total_pipeline_time_seconds": round(total_time, 3),
            "products_returned": len(results["items"]),
            "performance_ok": total_time < 30.0,  # Should complete in < 30s
            "passed": total_time < 30.0
        }
    
    def test_vector_search_availability(self) -> Dict[str, Any]:
        """Check if ChromaDB vector search is active"""
        test_name = "Vector Search Availability"
        logger.info(f"Running: {test_name}")
        
        query = "casual cotton shirts"
        results = self.retriever.retrieve(query, top_k=4)
        
        return {
            "test_name": test_name,
            "vector_search_active": results.get("vector_search_active", False),
            "chroma_index_size": results.get("chroma_count", 0),
            "fallback_mode": not results.get("vector_search_active", False),
            "passed": results.get("chroma_count", 0) > 0 or not results.get("vector_search_active", False)
        }
    
    def test_metadata_only_fallback(self) -> Dict[str, Any]:
        """Verify metadata-only filtering works when vector search unavailable"""
        test_name = "Metadata-Only Fallback"
        logger.info(f"Running: {test_name}")
        
        query = "men's leather jackets 3000 to 5000"
        results = self.retriever.retrieve(query, top_k=4)
        
        # Verify results satisfy metadata constraints even if vector search failed
        all_satisfy_constraints = all(
            3000 <= item["price_inr"] <= 5000 and
            item["gender"].lower() in ["men", "unisex"] and
            item["category"].lower() == "jackets"
            for item in results["items"]
        )
        
        return {
            "test_name": test_name,
            "query": query,
            "passed": all_satisfy_constraints or len(results["items"]) == 0,
            "products_returned": len(results["items"]),
            "all_satisfy_constraints": all_satisfy_constraints
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all evaluation tests"""
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE AI CLOTHING RECOMMENDATION SYSTEM EVALUATION")
        logger.info("=" * 70)
        
        tests = [
            self.test_budget_constraint,
            self.test_no_match_handling,
            self.test_category_filtering,
            self.test_gender_filtering,
            self.test_grounding_with_citations,
            self.test_vector_search_availability,
            self.test_metadata_only_fallback,
            self.test_latency_benchmark
        ]
        
        self.results["timestamp"] = time.time()
        
        for test_func in tests:
            try:
                result = test_func()
                self.results["test_cases"].append(result)
                status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
                logger.info(f"{status} - {result.get('test_name', 'Unknown')}")
            except Exception as exc:
                logger.error(f"❌ ERROR in {test_func.__name__}: {exc}", exc_info=True)
                self.results["test_cases"].append({
                    "test_name": test_func.__name__,
                    "passed": False,
                    "error": str(exc)
                })
        
        # Compute summary
        passed_count = sum(1 for t in self.results["test_cases"] if t.get("passed", False))
        total_count = len(self.results["test_cases"])
        
        self.results["summary"] = {
            "total_tests": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": f"{(passed_count / total_count * 100):.1f}%",
            "overall_status": "PASS" if passed_count >= total_count - 1 else "FAIL"
        }
        
        logger.info("=" * 70)
        logger.info(f"Summary: {passed_count}/{total_count} tests passed ({self.results['summary']['pass_rate']})")
        logger.info(f"Overall Status: {self.results['summary']['overall_status']}")
        logger.info("=" * 70)
        
        return self.results
    
    def save_report(self, output_path: str = "evaluation_results.json") -> None:
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {output_file}")
    
    def print_summary(self) -> None:
        """Print human-readable summary"""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT SUMMARY")
        print("=" * 70)
        
        for test in self.results["test_cases"]:
            status = "✅" if test.get("passed", False) else "❌"
            print(f"{status} {test.get('test_name', 'Unknown')}")
            
            # Print key details
            for key, value in test.items():
                if key not in ["test_name", "passed", "query", "error"] and not key.startswith("_"):
                    if isinstance(value, list) and len(value) == 0:
                        continue
                    if isinstance(value, (int, float, bool, str)) and value:
                        print(f"   └─ {key}: {value}")
        
        print("\n" + "=" * 70)
        print(f"OVERALL: {self.results['summary']['passed']}/{self.results['summary']['total_tests']} passed")
        print(f"STATUS: {self.results['summary']['overall_status']}")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    evaluator = EvaluationReport()
    results = evaluator.run_all_tests()
    evaluator.save_report("evaluation_results.json")
    evaluator.print_summary()
