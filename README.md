# 👔 RECOMAI - AI Clothing Product Recommendation System (RAG)

A production-grade, modular **AI Clothing Product Recommendation System** built with an explicit **Document → Chunking → Embedding → Vector DB (ChromaDB) → Semantic Retrieval → Product Aggregation → Hard Filtering → Hybrid Reranking → Grounded LLM (Ollama Llama 3.2 3B) → Streamlit UI** architecture.

---

## 🌟 Key Features

- **Explicit Document & Semantic Chunking**: Converts product catalog records into structured documents and splits them into distinct, focused semantic chunks (Identity, Attributes, and Commercial details) with preserved metadata.
- **Natural Language Query Parsing**: Extracts hard constraints (max/min price, category, subcategory, gender, color, fit, material, season) automatically from natural queries.
- **Chunk-Level Dense Vector Search**: High-performance semantic vector retrieval powered by **ChromaDB** and **Sentence Transformers** (`all-MiniLM-L6-v2`).
- **Product-Level Deduplication & Aggregation**: Groups retrieved chunks by `product_id`, aggregating best semantic similarity scores and multi-chunk context for each unique product.
- **Hard Metadata Filtering FIRST**: Enforces strict budget and categorical boundaries (e.g. price <= ₹2000, gender = Men), eliminating price leaks and hallucinations. During active vector retrieval, products without directly retrieved chunks are not assigned synthetic similarity.
- **Hybrid Relevance Reranking**: Blends semantic similarity with attribute match bonuses, fit preferences, and customer rating boosts.
- **Grounded AI Fashion Stylist (Ollama)**: When Ollama and `llama3.2:3b` are available, only top retrieved products and chunk evidence are supplied; unknown product-ID citations are rejected. Otherwise the deterministic fallback is used.
- **Smart Fallback Engine**: Seamlessly falls back to an offline rule-based fashion recommendation engine if Ollama is unavailable.
- **Developer / RAG Diagnostics Panel**: Built-in diagnostics expander displaying query embeddings, retrieved chunk IDs, product aggregation stats, and distance scores.
- **Zero Paid APIs**: Runs 100% locally with open-source models and libraries.

---

## 🏗️ End-to-End RAG Architecture

```text
Raw Product Dataset (data/clothing_products.csv)
        ↓
Data Cleaning & Validation (src/data_loader.py)
        ↓
Product Document Creation (ProductDocument in src/rag/chunker.py)
        ↓
Structured Semantic Chunking (ProductChunker: Identity, Attributes, Commercial)
        ↓
Chunk Metadata Tracking (chunk_id, product_id, chunk_index, total_chunks)
        ↓
Embedding Model (SentenceTransformers all-MiniLM-L6-v2 / SVD Fallback)
        ↓
ChromaDB Vector Store (recomai_product_chunks collection)
        ↓
Chunk-Level Similarity Search (Query Embedding → ChromaDB Cosine Search)
        ↓
Chunk Grouping & Product Aggregation (Deduplicate chunks → Unique products)
        ↓
Hard Metadata Filtering (Strict budget, category & gender enforcement)
        ↓
Hybrid Reranking (Semantic score + attribute match + rating boost)
        ↓
Top Product Selection (Top-k unique products)
        ↓
Llama 3.2 3B through Ollama (src/agent.py)
        ↓
Grounded Recommendation Explanation & Stylist Rationale
        ↓
Streamlit Dark-Themed UI (app/app.py)
```

---

## 🧩 Why Chunking?

> Product records are converted into textual documents and split into meaningful chunks before embedding. This allows semantic retrieval to operate on smaller, focused pieces of product information rather than treating an entire product record as one undifferentiated vector.

### Semantic Chunking Strategy

Each clothing product record is converted into a structured document and split into 3 domain-specific chunks:

1. **Chunk 1: Product Identity & Description**
   - Focus: Product name, brand, primary category, and full stylistic description.
   - Example: `"Product: Blue Slim Fit Jeans | Brand: Levi's | Category: jeans | Description: Stretch denim crafted for comfort..."`
2. **Chunk 2: Attributes, Fit, Material & Styling**
   - Focus: Gender, fit type, fabric material, color, season, and subcategory.
   - Example: `"Product: Blue Slim Fit Jeans | Category: jeans (slim fit jeans) | Gender: Men | Fit: Slim Fit | Material: Denim | Color: Blue"`
3. **Chunk 3: Commercial & Pricing Information**
   - Focus: Price in INR, customer ratings, value, and product ID anchor.
   - Example: `"Product: Blue Slim Fit Jeans | Brand: Levi's | Price: Rs 1899 | Customer Rating: 4.5/5.0 | Product ID: PRD1001"`

For products with lengthy narrative descriptions, recursive boundary splitting is applied with `chunk_size = 400` and `chunk_overlap = 50`.

Every chunk preserves the original `product_id`, a unique `chunk_id` (e.g. `PRD1001_chunk_0`), `chunk_index`, `total_chunks`, and the complete attribute metadata dictionary.

---

## 🛠️ Tech Stack & Tested Versions

- **Language**: Python 3.10+ (Tested on Python 3.11, 3.12, 3.14.7)
- **Frontend / UI**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Vector Database**: ChromaDB (Persistent local storage)
- **Embedding Models**: Sentence Transformers (`all-MiniLM-L6-v2`, 384-dimensional dense vectors) / Scikit-Learn TruncatedSVD fallback
- **Local LLM**: Ollama (`llama3.2:3b`)
- **Testing**: Pytest (95 tests currently collected; verify with `pytest --collect-only -q`)

---

## 📂 Project Structure

```text
AI Clothing/
├── app/
│   └── app.py                      # Streamlit UI & RAG Diagnostics Panel
├── src/
│   ├── rag/
│   │   ├── __init__.py             # RAG module exports
│   │   └── chunker.py              # ProductDocument, ProductChunk & ProductChunker
│   ├── chunker.py                  # Convenience alias for rag.chunker
│   ├── config.py                   # Global configuration & environment settings
│   ├── data_loader.py              # Dataset loading & metadata hard filter engine
│   ├── query_parser.py             # Natural language constraint parser
│   ├── embedder.py                 # Lazy-loaded vector embedder (ST & fallback)
│   ├── retrieval.py                # Chunk retrieval & product aggregation engine
│   ├── ranking.py                  # Hybrid ranking scoring engine
│   └── agent.py                    # Ollama LLM RAG agent & Smart Fashion fallback
├── scripts/
│   └── build_vector_store.py       # Chunk-based vector store builder
├── build_index.py                  # Backward-compatible build script alias
├── data/
│   └── clothing_products.csv       # 3,000 product dataset
├── tests/
│   ├── test_chunking.py            # Structured chunking & aggregation tests
│   ├── test_cleaner.py             # Dataset cleaning tests
│   ├── test_embedder_offline_fallback.py # Offline fallback embedder tests
│   ├── test_filters.py             # Hard filtering tests
│   ├── test_pipeline.py            # End-to-end pipeline tests
│   ├── test_query_parser.py        # Query parser constraint tests
│   └── test_ranking.py             # Hybrid ranking scoring tests
├── generate_dataset.py             # 3,000 product synthetic dataset generator
├── quick_evaluate.py               # Fast automated evaluation suite (10 test cases)
├── evaluate_system.py              # Comprehensive evaluation suite
├── requirements.txt                # Python package dependencies
├── .env.example                    # Environment configuration template
└── README.md                       # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
# Navigate to project directory
cd "AI Clothing"

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Set Up Ollama LLM

```bash
# Install Ollama from https://ollama.ai and pull the model:
ollama pull llama3.2:3b
ollama serve
```
Verify the exact model before testing generation:

```bash
ollama list
curl http://localhost:11434/api/tags
```

If Ollama is unavailable or the model is missing, the application uses its Smart Fashion Engine fallback and reports that source.

### 3. Generate Dataset & Build Chunk Vector Store

```bash
# Generate 3,000 clothing products
python generate_dataset.py

# Ingest, chunk, embed, and index into ChromaDB
python scripts/build_vector_store.py
```

Development option with product limit:
```bash
python scripts/build_vector_store.py --limit 2000
```

### 4. Launch the Streamlit Application

```bash
python -m streamlit run app/app.py
```

---

## 🧪 Testing & Evaluation

### Run the Test Suite

```bash
pytest tests/
```

The suite covers chunking, cleaning, embedding fallback, hard filters, broad-intent parsing, evidence-aware ranking, grounded prompt construction, and end-to-end retrieval. The Ollama-dependent explanation test requires a reachable Ollama service.

### Run Automated System Evaluation

```bash
python quick_evaluate.py
```

Evaluates 10 real test scenarios:
- **Budget Constraint Enforcement** (e.g. `jeans under 2000`)
- **No-Match / Extreme Budget Handling** (e.g. `jackets under 100`)
- **Out-of-Catalog / Hallucination Guard** (e.g. `luxury Rolex watch under 50`)
- **Category & Subcategory Matching** (e.g. `blue t-shirts under 1000`)
- **Gender & Unisex Fallback** (e.g. `women's hoodies under 2000`)
- **Price Range & Material Filter** (e.g. `formal linen shirts above 2000`)
- **Broad Budget Queries** (e.g. `something under 1000`)
- **Chunk Retrieval & Product Deduplication** (e.g. `blue slim fit jeans under 2000`)
- **Sub-100ms Latency Benchmarks**

---

## 📋 Configuration Reference

All settings can be customized in `.env`:

```env
# Ollama LLM Settings
OLLAMA_MODEL_NAME=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=5

# Vector Database & Chunking Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=recomai_product_chunks
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
DATASET_PATH=./data/clothing_products.csv
CHUNK_SIZE=400
CHUNK_OVERLAP=50

# Retrieval Defaults
DEFAULT_TOP_K=4
DEFAULT_SIMILARITY_THRESHOLD=0.20
```

---

## 🏆 Project Status

- **Status**: Local RAG prototype; Llama generation requires verified Ollama availability
- **Architecture**: Explicit Structured Chunk RAG Pipeline
- **Vector Store**: 9,000 Chunks / 3,000 Products indexed in `recomai_product_chunks`
- **Retrieval Latency**: ~25 – 65 ms per query
- **Evaluation**: Run `python quick_evaluate.py` for the current 10-scenario report; historical report files may contain older counts.
