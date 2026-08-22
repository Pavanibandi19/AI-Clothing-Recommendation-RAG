import logging
import chromadb
from typing import List, Dict, Any, Optional
from src.config import CHROMA_PERSIST_DIRECTORY, DATASET_PATH
from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.ranking import HybridRanker
from src.embedder import VectorEmbedder

logger = logging.getLogger(__name__)

class ProductRetriever:
    def __init__(self, dataset_path: Optional[str] = None, chroma_dir: Optional[str] = None):
        self.loader = DataLoader(dataset_path or DATASET_PATH)
        self.query_parser = QueryParser()
        self.ranker = HybridRanker()
        self.embedder = VectorEmbedder()
        
        chroma_path = chroma_dir or CHROMA_PERSIST_DIRECTORY
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.chroma_client.get_collection(name="clothing_products")
        except (ValueError, RuntimeError, TypeError) as exc:
            logger.warning(
                "ChromaDB collection unavailable (%s). Running in metadata-filtering fallback mode.",
                exc,
            )
            self.chroma_client = None
            self.collection = None

    def retrieve(
        self,
        query: str,
        ui_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Executes Hard Metadata Filtering FIRST, followed by Vector Similarity Search on candidates,
        and finally Hybrid Ranking with progressive filter relaxation fallback on zero results.
        """
        try:
            top_k = max(0, int(top_k))
        except (TypeError, ValueError):
            top_k = 4

        # 1. Parse natural language query
        parsed_query = self.query_parser.parse(query)

        # 2. Merge UI sidebar filters with parsed query constraints
        merged_filters = parsed_query.copy()

        if ui_filters:
            if ui_filters.get("category"):
                merged_filters["category"] = ui_filters["category"]
            if ui_filters.get("gender"):
                merged_filters["gender"] = ui_filters["gender"]
            if ui_filters.get("color"):
                merged_filters["color"] = ui_filters["color"]
            if ui_filters.get("fit"):
                merged_filters["fit"] = ui_filters["fit"]
            if ui_filters.get("material"):
                merged_filters["material"] = ui_filters["material"]
            if ui_filters.get("max_price") is not None:
                if parsed_query.get("max_price") is not None:
                    merged_filters["max_price"] = min(float(parsed_query["max_price"]), float(ui_filters["max_price"]))
                else:
                    merged_filters["max_price"] = ui_filters["max_price"]
            if ui_filters.get("min_price") is not None:
                if parsed_query.get("min_price") is not None:
                    merged_filters["min_price"] = max(float(parsed_query["min_price"]), float(ui_filters["min_price"]))
                else:
                    merged_filters["min_price"] = ui_filters["min_price"]

        # 3. Hard Metadata Filtering FIRST via pandas dataset
        filtered_df = self.loader.filter_products(merged_filters)

        # Zero-result fallback: Relax optional filters if initial strict filter produces 0 items
        if filtered_df.empty:
            relaxed_filters = {
                "category": merged_filters.get("category"),
                "gender": merged_filters.get("gender"),
                "max_price": merged_filters.get("max_price"),
                "min_price": merged_filters.get("min_price")
            }
            # Only retry if optional filters were actually present
            if any(merged_filters.get(k) for k in ["color", "material", "fit", "season", "brand", "subcategory"]):
                logger.info("Strict hard filter returned 0 products; retrying with relaxed core filters: %s", relaxed_filters)
                filtered_df = self.loader.filter_products(relaxed_filters)

        vector_search_active = bool(self.collection and self.collection.count() > 0)
        chroma_count = self.collection.count() if vector_search_active else 0

        # 4. If products are still empty (e.g. incompatible category+gender or impossible budget limit), return 0 gracefully
        if filtered_df.empty:
            return {
                "items": [],
                "count": 0,
                "parsed_query": merged_filters,
                "vector_search_active": vector_search_active,
                "chroma_count": chroma_count,
                "diagnostics": {
                    "vector_search_active": vector_search_active,
                    "chroma_count": chroma_count,
                    "candidates_count": 0,
                    "distances_map": {}
                },
                "message": "No matching products were found satisfying your specific budget or category requirements."
            }

        candidate_ids = set(filtered_df["product_id"].tolist())
        candidate_dict = {row["product_id"]: row.to_dict() for _, row in filtered_df.iterrows()}

        # 5. Vector search using ChromaDB if collection exists
        distances_map = {}
        if vector_search_active:
            query_embedding = self.embedder.encode([query]).tolist()
            
            # Query ChromaDB with candidate ID filtering if candidate set is reasonably sized
            where_clause = None
            if len(candidate_ids) <= 300:
                where_clause = {"product_id": {"$in": list(candidate_ids)}}

            try:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(self.collection.count(), max(top_k * 5, 50)),
                    where=where_clause
                )

                if results and "ids" in results and results["ids"]:
                    retrieved_ids = results["ids"][0]
                    retrieved_dists = results["distances"][0]
                    for p_id, dist in zip(retrieved_ids, retrieved_dists):
                        if p_id in candidate_ids:
                            distances_map[p_id] = dist
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Warning during vector query execution: %s", exc)

        # Neutral fallback distance (1.0 => 0.0 semantic similarity) for candidates not returned by vector top n
        candidates_to_rank = []
        distances_to_rank = []
        for p_id, p_dict in candidate_dict.items():
            candidates_to_rank.append(p_dict)
            distances_to_rank.append(distances_map.get(p_id, 1.0))

        # 6. Perform Hybrid Ranking
        final_items = self.ranker.rank(
            candidates=candidates_to_rank,
            distances=distances_to_rank,
            parsed_query=merged_filters,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )

        # 7. Final Hard Constraint Safety Filter
        safe_items = []
        for item in final_items:
            price = float(item.get("price_inr", 0))
            if merged_filters.get("max_price") is not None and price > float(merged_filters["max_price"]):
                continue
            if merged_filters.get("min_price") is not None and price < float(merged_filters["min_price"]):
                continue
            safe_items.append(item)
        final_items = safe_items

        return {
            "items": final_items,
            "count": len(final_items),
            "parsed_query": merged_filters,
            "vector_search_active": vector_search_active,
            "chroma_count": chroma_count,
            "diagnostics": {
                "vector_search_active": vector_search_active,
                "chroma_count": chroma_count,
                "candidates_count": len(candidate_ids),
                "distances_map": distances_map
            },
            "message": "Success" if final_items else "No matching products met the similarity threshold."
        }
