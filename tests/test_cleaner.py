import pytest
from pathlib import Path
from src.data_loader import DataLoader

def test_data_loader_initialization():
    loader = DataLoader()
    df = loader.get_all_products()
    assert not df.empty
    assert len(df) >= 2000
    
    required_cols = [
        "product_id", "product_name", "category", "subcategory", 
        "gender", "color", "size", "fit", "material", "brand", 
        "price_inr", "rating", "season", "description"
    ]
    for col in required_cols:
        assert col in df.columns

def test_data_types_cleaning():
    loader = DataLoader()
    df = loader.get_all_products()
    assert df["price_inr"].dtype in ["float64", "int64"]
    assert df["rating"].dtype in ["float64", "int64"]
    assert (df["price_inr"] > 0).all()
    assert (df["rating"] >= 1.0).all() and (df["rating"] <= 5.0).all()

def test_get_product_by_id():
    loader = DataLoader()
    product = loader.get_product_by_id("PRD1001")
    assert product is not None
    assert product["product_id"] == "PRD1001"
    assert "product_name" in product
