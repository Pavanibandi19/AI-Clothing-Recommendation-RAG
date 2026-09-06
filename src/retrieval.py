import logging
import chromadb
from typing import List, Dict, Any, Optional
from src.config import (
    CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, 
    DATASET_PATH, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
)
from src.data_loader import DataLoader
from src.query_parser import QueryParser
from src.ranking import HybridRanker
from src.embedder import VectorEmbedder

logger = logging.getLogger(__name__)

_CHROMA_CLIENT_CACHE: Dict[str, Any] = {}
_COLLECTION_CACHE: Dict[str, Any] = {}

class ProductRetriever:
    """
    RAG Product Retriever with Structured Chunk Search & Product Aggregation.
    
    Pipeline:
    1. Parse natural language query & merge with UI filters.
    2. Hard metadata filtering on dataset.
    3. Dense semantic vector retrieval on ChromaDB chunks collection.
    4. Group retrieved chunks by product_id and aggregate best distance & chunk context.
    5. Deduplicate to unique products.
    6. Hybrid reranking and hard budget safety verification.
    """
    def __init__(
        self, 
        dataset_path: Optional[str] = None, 
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        loader: Optional[DataLoader] = None,
        embedder: Optional[VectorEmbedder] = None
    ):
        self.loader = loader or DataLoader(dataset_path or DATASET_PATH)
        self.query_parser = QueryParser()
        self.ranker = HybridRanker()
        self.embedder = embedder or VectorEmbedder()
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        
        chroma_path = chroma_dir or CHROMA_PERSIST_DIRECTORY
        cache_key = f"{chroma_path}_{self.collection_name}"
        
        global _CHROMA_CLIENT_CACHE, _COLLECTION_CACHE
        if cache_key in _COLLECTION_CACHE:
            self.chroma_client = _CHROMA_CLIENT_CACHE.get(cache_key)
            self.collection = _COLLECTION_CACHE.get(cache_key)
        else:
            try:
                self.chroma_client = chromadb.PersistentClient(path=chroma_path)
                try:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
                except Exception:
                    # Graceful fallback to legacy collection name if primary chunk collection is not yet built
                    try:
                        self.collection = self.chroma_client.get_collection(name="clothing_products")
                        self.collection_name = "clothing_products"
                    except Exception:
                        self.collection = None
                
                _CHROMA_CLIENT_CACHE[cache_key] = self.chroma_client
                _COLLECTION_CACHE[cache_key] = self.collection
            except (ValueError, RuntimeError, TypeError, Exception) as exc:
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
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes Chunk-Level Vector Retrieval -> Product Aggregation -> Hard Filter Verification -> Hybrid Reranking.
        """
        try:
            top_k = max(0, int(top_k))
        except (TypeError, ValueError):
            top_k = DEFAULT_TOP_K

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

        # 3. Hard Metadata Filtering FIRST via dataset
        filtered_df = self.loader.filter_products(merged_filters)

        # Zero-result fallback: Relax optional filters if initial strict filter produces 0 items
        if filtered_df.empty:
            relaxed_filters = {
                "category": merged_filters.get("category"),
                "gender": merged_filters.get("gender"),
                "max_price": merged_filters.get("max_price"),
                "min_price": merged_filters.get("min_price")
            }
            if any(merged_filters.get(k) for k in ["color", "material", "fit", "season", "brand", "subcategory"]):
                logger.info("Strict hard filter returned 0 products; retrying with relaxed core filters: %s", relaxed_filters)
                filtered_df = self.loader.filter_products(relaxed_filters)

        vector_search_active = bool(self.collection and self.collection.count() > 0)
        chroma_count = self.collection.count() if vector_search_active else 0

        # If no products satisfy hard filters, return graceful empty result
        if filtered_df.empty:
            return {
                "items": [],
                "count": 0,
                "parsed_query": merged_filters,
                "vector_search_active": vector_search_active,
                "chroma_count": chroma_count,
                "diagnostics": {
                    "vector_search_active": vector_search_active,
                    "collection_name": self.collection_name,
                    "chroma_count": chroma_count,
                    "retrieved_chunks_count": 0,
                    "retrieved_chunk_ids": [],
                    "represented_product_ids": [],
                    "candidates_count": 0,
                    "distances_map": {},
                    "chunk_details": {}
                },
                "message": "No matching products were found satisfying your specific budget or category requirements."
            }

        candidate_ids = set(filtered_df["product_id"].tolist())
        candidate_dict = {row["product_id"]: row.to_dict() for _, row in filtered_df.iterrows()}

        # 4. Dense Vector Retrieval on Chunks Collection
        distances_map = {}
        grouped_chunks = {}
        retrieved_chunk_ids = []
        represented_product_ids = set()

        if vector_search_active:
            query_embedding = self.embedder.encode([query]).tolist()
            
            try:
                n_fetch = min(self.collection.count(), max(top_k * 25, 100))
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_fetch
                )

                if results and "ids" in results and results["ids"]:
                    chunk_ids = results["ids"][0]
                    chunk_dists = results["distances"][0]
                    chunk_metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(chunk_ids)
                    chunk_docs = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(chunk_ids)

                    retrieved_chunk_ids = chunk_ids

                    # Group retrieved chunks by product_id and compute best semantic distance per product
                    for c_id, dist, meta, doc_text in zip(chunk_ids, chunk_dists, chunk_metas, chunk_docs):
                        p_id = meta.get("product_id")
                        if not p_id:
                            # Parse from chunk_id pattern (e.g. PRD1001_chunk_0 -> PRD1001)
                            p_id = c_id.split("_chunk_")[0] if "_chunk_" in c_id else c_id

                        represented_product_ids.add(p_id)

                        if p_id in candidate_ids:
                            # Record best (lowest) cosine distance for this product
                            if p_id not in distances_map or dist < distances_map[p_id]:
                                distances_map[p_id] = dist

                            # Aggregate matched chunk info for LLM context grounding
                            if p_id not in grouped_chunks:
                                grouped_chunks[p_id] = []
                            grouped_chunks[p_id].append({
                                "chunk_id": c_id,
                                "distance": dist,
                                "text": doc_text,
                                "chunk_type": meta.get("chunk_type", "general")
                            })

            except (TypeError, ValueError, RuntimeError, Exception) as exc:
                logger.warning("Warning during vector query execution: %s", exc)

        # 5. Prepare Candidate Products with Aggregated Chunk Context
        candidates_to_rank = []
        distances_to_rank = []
        for p_id, p_dict in candidate_dict.items():
            p_copy = p_dict.copy()
            # Attach matched chunks context for this product
            p_copy["matched_chunks"] = grouped_chunks.get(p_id, [])
            candidates_to_rank.append(p_copy)
            # None means ChromaDB returned no evidence for this product. Do not
            # turn that absence into a fake distance or similarity score.
            distances_to_rank.append(distances_map.get(p_id))

        # 6. Perform Hybrid Reranking (Combines best semantic score + metadata score + rating boost)
        final_items = self.ranker.rank(
            candidates=candidates_to_rank,
            distances=distances_to_rank,
            parsed_query=merged_filters,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            allow_unretrieved=not vector_search_active,
        )

        # 7. Final Hard Constraint Safety Filter (Strict budget cap enforcement)
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
                "collection_name": self.collection_name,
                "chroma_count": chroma_count,
                "retrieved_chunks_count": len(retrieved_chunk_ids),
                "retrieved_chunk_ids": retrieved_chunk_ids[:15],
                "represented_product_ids": list(represented_product_ids)[:15],
                "unique_products_before_filtering": len(represented_product_ids),
                "candidates_count": len(candidate_ids),
                "distances_map": distances_map,
                "grouped_chunks": grouped_chunks
            },
            "message": "Success" if final_items else "No matching products met the similarity threshold."
        }
