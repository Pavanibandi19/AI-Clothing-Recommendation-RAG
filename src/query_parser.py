import re
from typing import Dict, Any, Optional
from src.config import (
    VALID_CATEGORIES, VALID_GENDERS, VALID_FITS, 
    VALID_MATERIALS, VALID_COLORS, VALID_SEASONS
)

class QueryParser:
    def __init__(self):
        # Category synonyms mapping to standard categories
        self.category_synonyms = {
            "jeans": "jeans",
            "jean": "jeans",
            "denim": "jeans",
            "denims": "jeans",
            "t-shirt": "t-shirts",
            "t-shirts": "t-shirts",
            "tshirt": "t-shirts",
            "tshirts": "t-shirts",
            "tee": "t-shirts",
            "tees": "t-shirts",
            "polo": "t-shirts",
            "polos": "t-shirts",
            "shirt": "shirts",
            "shirts": "shirts",
            "trouser": "trousers",
            "trousers": "trousers",
            "pant": "trousers",
            "pants": "trousers",
            "chinos": "trousers",
            "slacks": "trousers",
            "cargo": "trousers",
            "cargos": "trousers",
            "legging": "trousers",
            "leggings": "trousers",
            "jogger": "trousers",
            "joggers": "trousers",
            "sweatpants": "trousers",
            "bottoms": "trousers",
            "dress": "dresses",
            "dresses": "dresses",
            "maxi dress": "dresses",
            "maxi dresses": "dresses",
            "frock": "dresses",
            "frocks": "dresses",
            "gown": "dresses",
            "gowns": "dresses",
            "skirt": "skirts",
            "skirts": "skirts",
            "jacket": "jackets",
            "jackets": "jackets",
            "blazer": "jackets",
            "blazers": "jackets",
            "coat": "jackets",
            "coats": "jackets",
            "overcoat": "jackets",
            "overcoats": "jackets",
            "hoodie": "hoodies",
            "hoodies": "hoodies",
            "sweatshirt": "hoodies",
            "sweatshirts": "hoodies",
            "sweater": "sweaters",
            "sweaters": "sweaters",
            "cardigan": "sweaters",
            "cardigans": "sweaters",
            "pullover": "sweaters",
            "pullovers": "sweaters",
            "jumper": "sweaters",
            "jumpers": "sweaters",
            "top": "tops",
            "tops": "tops",
            "blouse": "tops",
            "blouses": "tops",
            "tunic": "tops",
            "tunics": "tops",
            "crop top": "tops",
            "crop tops": "tops",
            "tank top": "tops",
            "tank tops": "tops",
            "short": "shorts",
            "shorts": "shorts",
            "kurta": "ethnic wear",
            "kurtas": "ethnic wear",
            "kurti": "ethnic wear",
            "kurtis": "ethnic wear",
            "ethnic": "ethnic wear",
            "ethnic wear": "ethnic wear",
            "saree": "ethnic wear",
            "sarees": "ethnic wear",
            "sari": "ethnic wear",
            "saris": "ethnic wear",
            "anarkali": "ethnic wear",
            "lehenga": "ethnic wear",
            "activewear": "activewear",
            "sportswear": "activewear",
            "gymwear": "activewear",
            "workout": "activewear",
            "gym": "activewear",
            "track": "activewear"
        }

        # Fit synonyms (Strict fit cuts only - "maxi", "floral" are style descriptors, NOT fits)
        self.fit_synonyms = {
            "slim fit": "Slim Fit",
            "slim": "Slim Fit",
            "regular fit": "Regular Fit",
            "regular": "Regular Fit",
            "relaxed fit": "Relaxed Fit",
            "relaxed": "Relaxed Fit",
            "oversized": "Oversized",
            "oversize": "Oversized",
            "skinny fit": "Skinny Fit",
            "skinny": "Skinny Fit"
        }

        # Recognized style descriptors (for contextual understanding, not hard metadata filters)
        self.style_descriptors = [
            "maxi", "midi", "mini", "floral", "printed", "casual", 
            "formal", "party", "solid", "striped", "sleeveless", 
            "embroidered", "graphic"
        ]

        # Clothing vocabulary regex patterns for query validation
        self.clothing_validation_patterns = [
            r"\bjeans?\b",
            r"\bdenims?\b",
            r"\bt[- ]?shirts?\b",
            r"\btshirts?\b",
            r"\btees?\b",
            r"\bpolos?\b",
            r"\bshirts?\b",
            r"\bdress(?:es)?\b",
            r"\bmaxi\b",
            r"\bfrocks?\b",
            r"\bgowns?\b",
            r"\bhoodies?\b",
            r"\bsweatshirts?\b",
            r"\bpullovers?\b",
            r"\bjackets?\b",
            r"\bblazers?\b",
            r"\bcoats?\b",
            r"\bovercoats?\b",
            r"\bwindbreakers?\b",
            r"\bsweaters?\b",
            r"\bcardigans?\b",
            r"\bjumpers?\b",
            r"\btops?\b",
            r"\bcrop[- ]?tops?\b",
            r"\btank[- ]?tops?\b",
            r"\bblouses?\b",
            r"\btunics?\b",
            r"\bskirts?\b",
            r"\btrousers?\b",
            r"\bpants?\b",
            r"\bchinos?\b",
            r"\bslacks?\b",
            r"\bcargos?\b",
            r"\bleggings?\b",
            r"\bjoggers?\b",
            r"\bsweatpants?\b",
            r"\bbottoms?\b",
            r"\bshorts?\b",
            r"\bkurtas?\b",
            r"\bkurtis?\b",
            r"\bsarees?\b",
            r"\bsaris?\b",
            r"\banarkalis?\b",
            r"\blehengas?\b",
            r"\bsuits?\b",
            r"\btuxedos?\b",
            r"\bjumpsuits?\b",
            r"\brompers?\b",
            r"\bactivewear\b",
            r"\bsportswear\b",
            r"\bgymwear\b",
            r"\bstreetwear\b",
            r"\btracksuits?\b",
            r"\bclothings?\b",
            r"\bclothes\b",
            r"\bapparels?\b",
            r"\boutfits?\b",
            r"\battires?\b",
            r"\bgarments?\b",
            r"\bfashions?\b",
            r"\bwardrobes?\b",
            r"\bparty[- ]?wear\b",
            r"\bcasual[- ]?wear\b",
            r"\bformal[- ]?wear\b",
            r"\bethnic[- ]?wear\b",
            r"\bsummer[- ]?wear\b",
            r"\bwinter[- ]?wear\b",
            r"\bbeach[- ]?wear\b",
            r"\bdaily[- ]?wear\b",
            r"\bnight[- ]?wear\b",
            r"\bsleep[- ]?wear\b",
            r"\bwork[- ]?wear\b",
            r"\bwear\b"
        ]

    def is_clothing_query(self, query: str) -> bool:
        """
        Validates whether a search query contains legitimate clothing-related terms,
        categories, styles, occasions, or apparel attributes.
        """
        if not query or not str(query).strip():
            return False

        query_clean = str(query).strip().lower()

        # Remove numbers and punctuation to verify the query has textual substance
        text_only = re.sub(r"[\d\u20b9,.$%#@!*?&/\\()<>{}\[\]=+\-_:;\"']", " ", query_clean).strip()
        if not text_only:
            return False

        # 1. Match against comprehensive clothing patterns
        for pattern in self.clothing_validation_patterns:
            if re.search(pattern, query_clean):
                return True

        # 2. Check if query parser extracts any known category, material, fit, or gender
        parsed = self.parse(query)
        if parsed.get("category") or parsed.get("fit") or parsed.get("material") or parsed.get("style"):
            return True

        return False

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parses a natural language query string into structured filter constraints
        and cleaned semantic query string.
        """
        raw_query = query.strip()
        query_lower = raw_query.lower()

        parsed = {
            "raw_query": raw_query,
            "max_price": None,
            "min_price": None,
            "gender": None,
            "category": None,
            "subcategory": None,
            "color": None,
            "fit": None,
            "material": None,
            "season": None,
            "style": None,
            "cleaned_query": query_lower
        }

        # 1. Extract Price Constraints (max_price & min_price)
        price_number = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
        between_match = re.search(
            rf"\bbetween\s+(?:\u20b9|rs\.?|inr|rupees)?\s*({price_number})\s+and\s+(?:\u20b9|rs\.?|inr|rupees)?\s*({price_number})",
            query_lower,
        )
        if between_match:
            parsed["min_price"] = float(between_match.group(1).replace(",", ""))
            parsed["max_price"] = float(between_match.group(2).replace(",", ""))

        max_price_patterns = [
            rf"(?:under|below|less than|within|upto|up to|budget|max|<=|<|\u20b9|rs\.?|inr|rupees)\s*(?:\u20b9|rs\.?|inr|rupees)?\s*({price_number})",
            rf"({price_number})\s*(?:rs|rupees|inr)?\s*(?:or less|or below|max)"
        ]

        if parsed["max_price"] is None:
            for pattern in max_price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    parsed["max_price"] = float(match.group(1).replace(",", ""))
                    break

        min_price_patterns = [
            rf"(?:above|over|from|more than|min|minimum|>=|>)\s*(?:\u20b9|rs\.?|inr|rupees)?\s*({price_number})"
        ]

        if parsed["min_price"] is None:
            for pattern in min_price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    parsed["min_price"] = float(match.group(1).replace(",", ""))
                    break

        # 2. Extract Gender
        if re.search(r"\b(women|woman|women's|womens|ladies|female|girls)\b", query_lower):
            parsed["gender"] = "Women"
        elif re.search(r"\b(men|man|men's|mens|guys|male|boys)\b", query_lower):
            parsed["gender"] = "Men"
        elif re.search(r"\b(unisex)\b", query_lower):
            parsed["gender"] = "Unisex"

        # 3. Extract Category (sorted by length to match multi-word phrases first)
        for key in sorted(self.category_synonyms.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, query_lower):
                parsed["category"] = self.category_synonyms[key]
                break

        # 4. Extract Fit (Only genuine fit cuts - do not treat style descriptors like maxi as fit)
        for key in sorted(self.fit_synonyms.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, query_lower):
                parsed["fit"] = self.fit_synonyms[key]
                break

        # 5. Extract Style Descriptors
        for style in self.style_descriptors:
            pattern = r"\b" + re.escape(style) + r"\b"
            if re.search(pattern, query_lower):
                parsed["style"] = style
                break

        # 6. Extract Color
        for color in VALID_COLORS:
            c_pattern = r"\b" + re.escape(color.lower()) + r"\b"
            if color.lower() == "grey":
                c_pattern = r"\b(grey|gray)\b"
            if re.search(c_pattern, query_lower):
                parsed["color"] = color
                break

        # 7. Extract Material
        for material in VALID_MATERIALS:
            pattern = r"\b" + re.escape(material.lower()) + r"\b"
            if re.search(pattern, query_lower):
                parsed["material"] = material
                break

        # 8. Extract Season
        for season in VALID_SEASONS:
            if season.lower() == "all-season":
                continue
            pattern = r"\b" + re.escape(season.lower()) + r"\b"
            if re.search(pattern, query_lower):
                parsed["season"] = season
                break

        # Create cleaned semantic query by removing stop tokens
        stop_words = [
            "under", "below", "above", "over", "rs", "inr", "rupees", 
            "less than", "more than", "budget", "upto", "up to", 
            "for", "with", "a", "an", "the", "in"
        ]
        words = query_lower.split()
        cleaned_words = [w for w in words if w not in stop_words and not w.isdigit()]
        parsed["cleaned_query"] = " ".join(cleaned_words) if cleaned_words else query_lower

        return parsed
