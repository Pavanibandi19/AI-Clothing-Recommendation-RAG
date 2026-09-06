import pytest
import pandas as pd
from src.rag.chunker import ProductDocument, ProductChunk, ProductChunker
from src.retrieval import ProductRetriever
from src.embedder import VectorEmbedder
from src.data_loader import DataLoader


def test_short_product_creates_chunks():
    chunker = ProductChunker(chunk_size=400, chunk_overlap=50)
    short_row = {
        "product_id": "PRD_TEST_01",
        "product_name": "Slim Fit Blue Denim Jeans",
        "brand": "Levis",
        "category": "jeans",
        "subcategory": "slim fit jeans",
        "gender": "Men",
        "color": "Blue",
        "size": "32",
        "fit": "Slim Fit",
        "material": "Denim",
        "price_inr": 1899,
        "rating": 4.5,
        "season": "All-Season",
        "description": "Classic blue stretch denim."
    }
    doc = chunker.create_product_document(short_row)
    chunks = chunker.chunk_product(doc)

    assert len(chunks) >= 1
    assert all(isinstance(c, ProductChunk) for c in chunks)
    for c in chunks:
        assert c.product_id == "PRD_TEST_01"
        assert c.text.strip() != ""
        assert "product_name" in c.metadata
        assert c.metadata["brand"] == "Levis"
        assert c.metadata["price_inr"] == 1899.0


def test_long_product_creates_multiple_chunks():
    chunker = ProductChunker(chunk_size=150, chunk_overlap=30)
    long_description = (
        "This is an ultra-premium handcrafted organic cotton tailored shirt designed "
        "specifically for formal events, boardroom meetings, and luxury evening wear. "
        "Features reinforced mother-of-pearl buttons, double-needle stitching throughout, "
        "and breathable Italian luxury weave engineered for all-day comfort in tropical weather."
    )
    long_row = {
        "product_id": "PRD_LONG_01",
        "product_name": "Handcrafted Luxury Shirt",
        "brand": "Raymond",
        "category": "shirts",
        "subcategory": "formal shirts",
        "gender": "Men",
        "color": "White",
        "size": "40",
        "fit": "Regular Fit",
        "material": "Cotton",
        "price_inr": 3499,
        "rating": 4.8,
        "season": "Summer",
        "description": long_description
    }
    doc = chunker.create_product_document(long_row)
    chunks = chunker.chunk_product(doc)

    assert len(chunks) >= 3
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Every chunk_id must be unique"


def test_chunk_metadata_and_indexing_integrity():
    chunker = ProductChunker()
    row = {
        "product_id": "PRD_INDEX_99",
        "product_name": "Summer Cotton Sundress",
        "brand": "Zara",
        "category": "dresses",
        "subcategory": "sundress",
        "gender": "Women",
        "color": "Yellow",
        "size": "M",
        "fit": "Relaxed Fit",
        "material": "Cotton",
        "price_inr": 1299,
        "rating": 4.2,
        "season": "Summer",
        "description": "Lightweight breathable breezy summer sundress."
    }
    doc = chunker.create_product_document(row)
    chunks = chunker.chunk_product(doc)

    assert len(chunks) > 0
    total = len(chunks)
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.total_chunks == total
        assert c.chunk_id == f"PRD_INDEX_99_chunk_{idx}"
        assert c.metadata["chunk_index"] == idx
        assert c.metadata["total_chunks"] == total
        assert c.metadata["product_id"] == "PRD_INDEX_99"


def test_dataset_chunking():
    chunker = ProductChunker()
    df = pd.DataFrame([
        {
            "product_id": f"P{i}",
            "product_name": f"Item {i}",
            "brand": "BrandX",
            "category": "t-shirts",
            "subcategory": "crew neck",
            "gender": "Men",
            "color": "Black",
            "size": "L",
            "fit": "Regular Fit",
            "material": "Cotton",
            "price_inr": 500 + i * 100,
            "rating": 4.0,
            "season": "Summer",
            "description": f"Quality cotton item {i}."
        }
        for i in range(5)
    ])
    chunks = chunker.chunk_dataset(df)
    assert len(chunks) >= 5
    product_ids_represented = {c.product_id for c in chunks}
    assert product_ids_represented == {f"P{i}" for i in range(5)}


def test_retriever_deduplicates_chunks_to_unique_products():
    retriever = ProductRetriever()
    result = retriever.retrieve("black cotton t-shirts under 2000", top_k=4)

    assert "items" in result
    items = result["items"]
    assert len(items) <= 4

    # Check deduplication: each product in items has a unique product_id
    pids = [item["product_id"] for item in items]
    assert len(pids) == len(set(pids)), "Recommended items must be unique products"

    # Check hard price constraint
    for item in items:
        assert float(item["price_inr"]) <= 2000
