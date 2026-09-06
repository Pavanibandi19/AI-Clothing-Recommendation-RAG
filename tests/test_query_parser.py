import pytest
from src.query_parser import QueryParser

@pytest.fixture
def parser():
    return QueryParser()

def test_parse_blue_jeans_under_2000(parser):
    res = parser.parse("blue slim fit jeans under 2000")
    assert res["category"] == "jeans"
    assert res["color"] == "Blue"
    assert res["fit"] == "Slim Fit"
    assert res["max_price"] == 2000.0

def test_parse_black_casual_tshirts(parser):
    res = parser.parse("black casual t-shirts below 1500")
    assert res["category"] == "t-shirts"
    assert res["color"] == "Black"
    assert res["max_price"] == 1500.0

def test_parse_womens_cotton_dresses(parser):
    res = parser.parse("women's cotton dresses for summer")
    assert res["gender"] == "Women"
    assert res["material"] == "Cotton"
    assert res["category"] == "dresses"
    assert res["season"] == "Summer"

def test_dress_shirt_phrases_use_shirts_category(parser):
    for query in ("men's dress shirts", "formal dress shirt for men"):
        res = parser.parse(query)
        assert res["category"] == "shirts"

def test_mens_formal_dress_is_semantic_style_not_dresses_category(parser):
    res = parser.parse("yellow formal dress for men")
    assert res["gender"] == "Men"
    assert res["color"] == "Yellow"
    assert res["style"] == "formal"
    assert res["category"] is None
    assert res["intent"] == "formal wear"

@pytest.mark.parametrize("query", [
    "dress for men",
    "show me a dress for men",
    "men's dress",
    "formal dress for men",
    "casual dress for men",
    "party dress for men",
])
def test_mens_dress_outfit_language_is_soft_intent(parser, query):
    result = parser.parse(query)
    assert result["gender"] == "Men"
    assert result["category"] is None
    assert result["intent"] in ("broad clothing", "formal wear", "casual wear", "party wear")

@pytest.mark.parametrize("color", [
    "red", "blue", "yellow", "green", "black", "white", "pink",
    "orange", "purple", "brown", "grey", "gray", "navy", "maroon",
])
def test_formal_dress_context_never_uses_color_as_dress_category(parser, color):
    result = parser.parse(f"suggest me a {color} formal dress for men under 3000")
    assert result["gender"] == "Men"
    assert result["category"] is None
    assert result["style"] == "formal"
    assert result["max_price"] == 3000.0
    assert result["color"] == ("Grey" if color == "gray" else color.title())

def test_formal_shirt_and_explicit_womens_dresses_remain_categories(parser):
    assert parser.parse("yellow formal shirt for men")["category"] == "shirts"
    assert parser.parse("women's dresses")["category"] == "dresses"
    assert parser.parse("women's dress")["category"] == "dresses"
    assert parser.parse("women's cotton dress")["category"] == "dresses"

@pytest.mark.parametrize("query", [
    "can you recommend something blue for a man?",
    "something nice for a woman",
    "recommend something for a party",
    "something affordable for women",
    "something comfortable for men",
    "what should I wear to a party",
    "something stylish for a man",
])
def test_natural_clothing_recommendation_queries_are_valid(parser, query):
    assert parser.is_clothing_query(query)

def test_natural_recommendation_parser_extracts_supported_signals(parser):
    blue_man = parser.parse("can you recommend something blue for a man?")
    assert blue_man["gender"] == "Men"
    assert blue_man["color"] == "Blue"
    assert blue_man["intent"] == "broad clothing"

    party = parser.parse("recommend something stylish for a party")
    assert party["style"] == "party"
    assert party["intent"] == "party wear"

    affordable = parser.parse("something affordable for women")
    assert affordable["gender"] == "Women"
    assert affordable["intent"] == "budget clothing"

@pytest.mark.parametrize(
    ("query", "expected_category", "expected_style"),
    [
        ("dress", "dresses", None),
        ("dress shirt", "shirts", None),
        ("formal dress", None, "formal"),
        ("formal dress shirt", "shirts", "formal"),
        ("party dress", "dresses", "party"),
        ("casual dress", "dresses", "casual"),
        ("formal wear", None, "formal"),
        ("summer dress", "dresses", "summer"),
    ],
)
def test_ambiguous_dress_phrases_are_interpreted_by_context(parser, query, expected_category, expected_style):
    result = parser.parse(query)
    assert result["category"] == expected_category
    assert result["style"] == expected_style

def test_parse_mens_oversized_hoodies(parser):
    res = parser.parse("men's oversized hoodies under 2500")
    assert res["gender"] == "Men"
    assert res["fit"] == "Oversized"
    assert res["category"] == "hoodies"
    assert res["max_price"] == 2500.0

def test_parse_min_price(parser):
    res = parser.parse("jackets above 3000")
    assert res["category"] == "jackets"
    assert res["min_price"] == 3000.0

def test_parse_comma_decimal_and_currency_prices(parser):
    cases = {
        "jeans under ₹2000": (None, 2000.0),
        "jeans below rs. 2000": (None, 2000.0),
        "jeans below 2,000": (None, 2000.0),
        "jeans under 1999.99": (None, 1999.99),
        "jeans above 1000": (1000.0, None),
        "jeans from 1000": (1000.0, None),
        "jeans between 1000 and 2,000": (1000.0, 2000.0),
        "dresses under INR 3000": (None, 3000.0),
        "shirts up to 2500": (None, 2500.0),
        "jeans less than 2000": (None, 2000.0),
        "blue jeans under 2000 rupees": (None, 2000.0),
    }

    for query, expected in cases.items():
        parsed = parser.parse(query)
        assert (parsed["min_price"], parsed["max_price"]) == expected
        for key in ("min_price", "max_price"):
            if parsed[key] is not None:
                assert isinstance(parsed[key], float)

def test_tshirt_query_is_valid(parser):
    assert parser.is_clothing_query("black casual t-shirts below 1500")
    assert parser.is_clothing_query("black tshirts")
    assert parser.is_clothing_query("black t-shirts")
    assert parser.is_clothing_query("black men's t-shirts")

def test_dresses_query_is_valid(parser):
    assert parser.is_clothing_query("women's cotton dresses for summer")
    assert parser.is_clothing_query("women dresses")
    assert parser.is_clothing_query("dresses below rs. 3000")
    assert parser.is_clothing_query("party wear dresses under 4000")

def test_hoodies_query_is_valid(parser):
    assert parser.is_clothing_query("men's oversized hoodies under 2500")
    assert parser.is_clothing_query("hoodies")
    assert parser.is_clothing_query("sweatshirts")

def test_all_popular_queries_are_valid(parser):
    popular_searches = [
        "blue slim fit jeans under 2000",
        "black casual t-shirts below 1500",
        "women's cotton dresses for summer",
        "men's oversized hoodies under 2500"
    ]
    for query in popular_searches:
        assert parser.is_clothing_query(query)

def test_edge_case_clothing_queries_are_valid(parser):
    valid_edge_cases = [
        "JEANS UNDER 2000",
        "Jeans under 2000",
        "jeans UNDER ₹2000",
        "jeans under ₹2,000",
        "jeans below rs. 2000",
        "jeans below INR 2000",
        "blue jeans under 2000 rupees",
        "black t-shirts below 1500",
        "black tshirts below 1500",
        "women's dresses",
        "womens dresses",
        "men's jeans",
        "mens jeans",
        "cotton shirts",
        "party wear dresses",
        "casual clothes",
        "casual clothes for summer",
        "summer dresses",
        "party wear",
        "casual wear for summer",
        "cotton shirts under ₹2000",
        "dresses below rs. 3000",
        "blue denim jeans",
        "men shirts",
        "activewear"
    ]
    for q in valid_edge_cases:
        assert parser.is_clothing_query(q), f"Failed for valid query: {q}"

def test_invalid_queries_are_rejected(parser):
    invalid_queries = [
        "hello",
        "asdfgh",
        "123",
        "123456",
        "",
        "   ",
        "???",
        "xyz random words"
    ]
    for q in invalid_queries:
        assert not parser.is_clothing_query(q), f"Expected false for invalid query: {q}"
