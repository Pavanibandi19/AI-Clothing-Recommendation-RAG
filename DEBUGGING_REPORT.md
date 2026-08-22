# 🐞 Debugging Report & Edge-Case Issue Resolution

## 1. Bug Description

During initial testing of the natural language query parsing and hard metadata filtering pipeline, price constraint extraction encountered a type mismatch bug when queries contained currency symbols or floating format variations (e.g., `"blue jeans under ₹2000"`, `"t-shirts below rs. 1500"`).

The query parser extracted the match as an uncleaned string (e.g., `"2000"`, `"1500"`), which was passed into the Pandas DataFrame filter function without proper numerical type casting. When the filter evaluated `df["price_inr"] <= max_price`, Pandas raised a runtime `TypeError` or silently failed to filter products exceeding the price threshold when comparing float column values against string criteria.

---

## 2. Expected Result

- A query containing `"under 2000"` or `"below ₹1500"` must extract `max_price = 2000.0` or `1500.0` as a native Python `float`.
- Hard metadata filtering MUST guarantee that 100% of returned clothing products satisfy `price_inr <= max_price`.
- Products with prices exceeding ₹2000 (e.g. ₹2499) must be **strictly excluded** from the final recommendation output.

---

## 3. Actual Result

- Prior to the fix, queries containing currency symbols like `₹` or `rs.` failed regex matching or preserved trailing characters.
- In `DataLoader.filter_products()`, string price inputs caused either:
  1. `TypeError: '<=' not supported between instances of 'float' and 'str'`
  2. Failure of hard metadata filtering, causing products above the budget cap (e.g. ₹2999) to leak into vector similarity search results.

---

## 4. Root Cause

1. **Regex Pattern Inconsistency**: The initial regex pattern `r"under (\d+)"` did not account for unicode currency symbols (`\u20b9` / `₹`), optional punctuation (`rs.`), or spaces between symbols and numbers.
2. **Missing Type Enforcement**: The parsed output dictionary stored raw string regex groups directly into `parsed["max_price"]` without casting to `float()`.
3. **Pandas Type Coercion**: Pandas requires numeric operands when comparing a Series against a scalar limit.

---

## 5. Debugging Steps & Resolution

### Step 1: Trace Failure in Pytest

Created unit test case in `tests/test_query_parser.py`:

```python
def test_parse_currency_symbols(parser):
    res = parser.parse("jeans under ₹2000")
    assert res["max_price"] == 2000.0
    assert isinstance(res["max_price"], float)
```

### Step 2: Update Regex Engine in `src/query_parser.py`

Replaced simple pattern with robust multi-regex matcher supporting unicode currency symbols and numeric boundaries:

```python
max_price_patterns = [
    r"(?:under|below|less than|within|upto|up to|budget|max|<|<=|\u20b9|rs\.?)\s*(?:\u20b9|rs\.?)?\s*(\d{3,6})",
    r"(\d{3,6})\s*(?:rs|rupees|inr)?\s*(?:or less|or below|max)"
]

for pattern in max_price_patterns:
    match = re.search(pattern, query_lower)
    if match:
        parsed["max_price"] = float(match.group(1))  # Explicit float conversion
        break
```

### Step 3: Hard Boundary Guard in `src/data_loader.py`

Enforced numeric casting prior to applying filtering on DataFrame:

```python
if "max_price" in filters and filters["max_price"] is not None:
    max_p = float(filters["max_price"])
    filtered_df = filtered_df[filtered_df["price_inr"] <= max_p]
```

---

## 6. Verification & Test Case Evidence

All 18 unit test cases in `pytest tests/` pass cleanly without errors:

- `test_query_parser.py`: Verified price extraction for `under 2000`, `below 1500`, `under ₹2500`, `above 3000`.
- `test_filters.py`: Verified hard filtering guarantees `(filtered["price_inr"] <= max_price).all()`.
- `test_pipeline.py`: Verified end-to-end RAG retrieval enforces price constraints on final output.
