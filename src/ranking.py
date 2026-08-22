from typing import List, Dict, Any

class HybridRanker:
    def __init__(self, semantic_weight: float = 0.65, metadata_weight: float = 0.35):
        self.semantic_weight = semantic_weight
        self.metadata_weight = metadata_weight

    def calculate_hybrid_score(
        self, 
        item: Dict[str, Any], 
        semantic_sim: float, 
        parsed_query: Dict[str, Any]
    ) -> float:
        """
        Calculates hybrid relevance score combining semantic similarity and metadata attribute matches.
        """
        # Ensure semantic similarity is bounded between 0.0 and 1.0
        base_sim = max(0.0, min(1.0, float(semantic_sim)))

        metadata_score = 0.0
        max_possible_meta = 0.0

        # Category match boost
        if parsed_query.get("category"):
            max_possible_meta += 0.30
            if str(item.get("category", "")).lower() == str(parsed_query["category"]).lower():
                metadata_score += 0.30

        # Color match boost
        if parsed_query.get("color"):
            max_possible_meta += 0.25
            if str(item.get("color", "")).lower() == str(parsed_query["color"]).lower():
                metadata_score += 0.25

        # Fit match boost
        if parsed_query.get("fit"):
            max_possible_meta += 0.20
            if str(item.get("fit", "")).lower() == str(parsed_query["fit"]).lower():
                metadata_score += 0.20

        # Material match boost
        if parsed_query.get("material"):
            max_possible_meta += 0.15
            if str(item.get("material", "")).lower() == str(parsed_query["material"]).lower():
                metadata_score += 0.15

        # Season match boost
        if parsed_query.get("season"):
            max_possible_meta += 0.10
            item_season = str(item.get("season", "")).lower()
            query_season = str(parsed_query["season"]).lower()
            if item_season == query_season or item_season == "all-season":
                metadata_score += 0.10

        # Normalize metadata boost score
        if max_possible_meta > 0:
            norm_metadata_score = metadata_score / max_possible_meta
        else:
            norm_metadata_score = 0.5  # Neutral metadata score if no metadata in query

        # Small rating boost (0.0 to 0.05)
        rating = float(item.get("rating", 4.0))
        rating_boost = max(0.0, (rating - 3.5) / 1.5) * 0.05

        # Final hybrid score calculation
        total_score = (self.semantic_weight * base_sim) + (self.metadata_weight * norm_metadata_score) + rating_boost
        return round(min(1.0, total_score), 4)

    def rank(
        self, 
        candidates: List[Dict[str, Any]], 
        distances: List[float], 
        parsed_query: Dict[str, Any], 
        similarity_threshold: float = 0.20,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate products based on hybrid score and filters out items below similarity threshold.
        """
        ranked_results = []

        for item, dist in zip(candidates, distances):
            # ChromaDB cosine distance range: 0.0 (identical) to 2.0 (opposite)
            # Convert cosine distance to cosine similarity: sim = 1 - (dist / 2) or 1 - dist
            # For normalized embeddings: cosine_similarity = 1.0 - (cosine_distance)
            semantic_sim = max(0.0, 1.0 - float(dist))

            hybrid_score = self.calculate_hybrid_score(item, semantic_sim, parsed_query)

            # Apply similarity threshold cutoff
            if hybrid_score >= float(similarity_threshold):
                item_copy = item.copy()
                item_copy["semantic_similarity"] = round(semantic_sim, 4)
                item_copy["hybrid_score"] = hybrid_score
                item_copy["match_percentage"] = int(hybrid_score * 100)
                ranked_results.append(item_copy)

        # Sort descending by hybrid_score
        ranked_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return ranked_results[:top_k]
