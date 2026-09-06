import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class ProductDocument:
    """Represents a structured textual document constructed from a raw product record."""
    product_id: str
    product_name: str
    brand: str
    category: str
    subcategory: str
    gender: str
    color: str
    size: str
    fit: str
    material: str
    price_inr: float
    rating: float
    season: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Constructs a comprehensive, unified product textual representation."""
        return (
            f"Product: {self.product_name} | Brand: {self.brand} | Category: {self.category} | "
            f"Subcategory: {self.subcategory} | Gender: {self.gender} | Color: {self.color} | "
            f"Size: {self.size} | Fit: {self.fit} | Material: {self.material} | "
            f"Season: {self.season} | Price: Rs {int(self.price_inr)} | Rating: {self.rating} | "
            f"Description: {self.description}"
        )


@dataclass
class ProductChunk:
    """Represents an atomic, semantic chunk of a product document ready for embedding."""
    chunk_id: str
    product_id: str
    chunk_index: int
    total_chunks: int
    chunk_type: str
    text: str
    metadata: Dict[str, Any]


class ProductChunker:
    """
    Structured Semantic Chunker for Clothing Product Catalogs.
    
    Transforms product documents into domain-specific semantic chunks:
    - Chunk 1: Product Identity & Description
    - Chunk 2: Attributes, Fit, Material & Styling
    - Chunk 3: Commercial, Price & Rating Details
    
    Handles longer descriptions with recursive boundary splitting while preserving
    product identity, chunk indexing, and rich metadata on every chunk.
    """
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_product_document(self, row: Union[pd.Series, Dict[str, Any]]) -> ProductDocument:
        """Constructs a clean ProductDocument from a raw dataframe row or dictionary."""
        d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        
        pid = str(d.get("product_id", "")).strip()
        pname = str(d.get("product_name", "")).strip()
        brand = str(d.get("brand", "")).strip()
        category = str(d.get("category", "")).strip()
        subcategory = str(d.get("subcategory", "")).strip()
        gender = str(d.get("gender", "")).strip()
        color = str(d.get("color", "")).strip()
        size = str(d.get("size", "")).strip()
        fit = str(d.get("fit", "")).strip()
        material = str(d.get("material", "")).strip()
        
        try:
            price_inr = float(d.get("price_inr", 0))
        except (ValueError, TypeError):
            price_inr = 0.0

        try:
            rating = float(d.get("rating", 4.0))
        except (ValueError, TypeError):
            rating = 4.0

        season = str(d.get("season", "")).strip()
        description = str(d.get("description", "")).strip()

        meta = {
            "product_id": pid,
            "product_name": pname,
            "brand": brand,
            "category": category.lower(),
            "subcategory": subcategory.lower(),
            "gender": gender.lower(),
            "color": color.lower(),
            "size": size,
            "fit": fit.lower(),
            "material": material.lower(),
            "season": season.lower(),
            "price_inr": price_inr,
            "rating": rating
        }

        return ProductDocument(
            product_id=pid,
            product_name=pname,
            brand=brand,
            category=category,
            subcategory=subcategory,
            gender=gender,
            color=color,
            size=size,
            fit=fit,
            material=material,
            price_inr=price_inr,
            rating=rating,
            season=season,
            description=description,
            metadata=meta
        )

    def _split_long_text(self, text: str) -> List[str]:
        """Splits long text recursively by logical delimiters into overlapping chunks."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find best split point (sentence break or word break)
            split_at = -1
            for delimiter in [". ", "? ", "! ", "\n\n", "\n", ", ", " "]:
                idx = text.rfind(delimiter, start, end)
                if idx > start + int(self.chunk_size * 0.3):
                    split_at = idx + len(delimiter)
                    break

            if split_at == -1:
                split_at = end

            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            # Advance by step size minus overlap
            start = max(start + 1, split_at - self.chunk_overlap)

        return chunks

    def chunk_product(self, doc: ProductDocument) -> List[ProductChunk]:
        """
        Deconstructs a ProductDocument into semantically meaningful chunks with metadata.
        """
        raw_chunks = []

        # 1. Chunk 1: Product Identity & Description
        identity_text = (
            f"Product: {doc.product_name} | Brand: {doc.brand} | Category: {doc.category} | "
            f"Description: {doc.description}"
        ).strip()
        
        # Check if description alone exceeds chunk size
        if len(identity_text) > self.chunk_size:
            desc_splits = self._split_long_text(doc.description)
            for i, desc_part in enumerate(desc_splits):
                raw_chunks.append({
                    "type": "identity_description",
                    "text": f"Product: {doc.product_name} | Brand: {doc.brand} | Category: {doc.category} | Description: {desc_part}"
                })
        else:
            raw_chunks.append({
                "type": "identity_description",
                "text": identity_text
            })

        # 2. Chunk 2: Attributes, Fit, Material & Styling
        attributes_text = (
            f"Product: {doc.product_name} | Category: {doc.category} ({doc.subcategory}) | "
            f"Gender: {doc.gender} | Fit: {doc.fit} | Material: {doc.material} | "
            f"Color: {doc.color} | Season: {doc.season}"
        ).strip()
        raw_chunks.append({
            "type": "attributes_style",
            "text": attributes_text
        })

        # 3. Chunk 3: Commercial & Pricing Details
        pricing_text = (
            f"Product: {doc.product_name} | Brand: {doc.brand} | Price: Rs {int(doc.price_inr)} | "
            f"Customer Rating: {doc.rating}/5.0 | Product ID: {doc.product_id}"
        ).strip()
        raw_chunks.append({
            "type": "commercial_pricing",
            "text": pricing_text
        })

        # Filter any empty chunks
        valid_raw_chunks = [c for c in raw_chunks if c["text"].strip()]
        total_chunks = len(valid_raw_chunks)

        # Build final ProductChunk objects with complete metadata
        product_chunks = []
        for idx, item in enumerate(valid_raw_chunks):
            chunk_id = f"{doc.product_id}_chunk_{idx}"
            
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_type": item["type"]
            })

            chunk_obj = ProductChunk(
                chunk_id=chunk_id,
                product_id=doc.product_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                chunk_type=item["type"],
                text=item["text"],
                metadata=chunk_metadata
            )
            product_chunks.append(chunk_obj)

        return product_chunks

    def chunk_dataset(self, df: pd.DataFrame) -> List[ProductChunk]:
        """Processes an entire DataFrame of products into a unified list of ProductChunk objects."""
        all_chunks = []
        for _, row in df.iterrows():
            doc = self.create_product_document(row)
            chunks = self.chunk_product(doc)
            all_chunks.extend(chunks)
        return all_chunks
