import pytest
from src.data_loader import DataLoader

@pytest.fixture
def loader():
    return DataLoader()

def test_hard_price_filtering(loader):
    filtered = loader.filter_products({"max_price": 1500})
    assert not filtered.empty
    # CRITICAL: No product should exceed max_price
    assert (filtered["price_inr"] <= 1500).all()

def test_category_filtering(loader):
    filtered = loader.filter_products({"category": "jeans"})
    assert not filtered.empty
    assert (filtered["category"].str.lower() == "jeans").all()

def test_gender_filtering(loader):
    filtered = loader.filter_products({"gender": "Women"})
    assert not filtered.empty
    # Must be either Women or Unisex
    allowed = ["women", "unisex"]
    assert filtered["gender"].str.lower().isin(allowed).all()

def test_combined_filters(loader):
    filters = {
        "gender": "Men",
        "category": "hoodies",
        "max_price": 2500,
        "color": "Black"
    }
    filtered = loader.filter_products(filters)
    if not filtered.empty:
        assert (filtered["price_inr"] <= 2500).all()
        assert (filtered["category"].str.lower() == "hoodies").all()
        assert (filtered["color"].str.lower() == "black").all()

def test_impossible_filter_returns_empty(loader):
    filters = {
        "category": "dresses",
        "gender": "Men",  # Dresses are Women only in dataset
        "max_price": 50   # No dress under 50 INR
    }
    filtered = loader.filter_products(filters)
    assert filtered.empty
