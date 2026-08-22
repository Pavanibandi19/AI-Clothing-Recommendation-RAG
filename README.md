# 👔 AI Clothing Product Recommendation System (RAG)

A production-grade, beginner-friendly **AI Clothing Product Recommendation System** built with **Retrieval-Augmented Generation (RAG)**, **Streamlit**, **Pandas**, **ChromaDB**, **Sentence Transformers**, and local **Ollama LLM**.

---

## 🌟 Key Features

- **Natural Language Query Parsing**: Extracts hard constraints (max/min price, category, subcategory, gender, color, fit, material, season) automatically from freeform text queries.
- **Hard Metadata Filtering FIRST**: Enforces strict metadata boundaries (e.g. price <= ₹2000, gender = Men) *before* vector search, preventing price leaks or category mismatches.
- **Vector Similarity Search**: Powered by persistent **ChromaDB** vector database with **Sentence Transformers** (`all-MiniLM-L6-v2`) embeddings.
- **Hybrid Relevance Ranking**: Combines vector cosine similarity with attribute matching boosts (color, fit, material, rating).
- **Ollama Local LLM RAG Rationale**: Connects to local Ollama (`llama3.2:3b`) to generate personalized fashion advice explaining why recommended items match user preferences.
- **Smart Fallback Engine**: Seamlessly functions with a local rule-based AI engine if Ollama is offline or unavailable.
- **Modern Dark UI**: Designed with glassmorphic cards, purple/blue glowing accents, interactive sidebar filters, search history, and clickable example pills.
- **Zero Paid APIs**: Runs 100% locally with free, open-source libraries and local models.

---

## 🏗️ RAG Architecture

```text
+-----------------------------------------------------------------------+
|                         User Natural Language Query                   |
|             e.g., "blue slim fit jeans under 2000"                    |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    1. Query Parser (src/query_parser.py)              |
|   Extracts: category=jeans, color=Blue, fit=Slim Fit, max_price=2000 INR |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                 2. Hard Metadata Filter (src/data_loader.py)           |
|   Filters 3,000 dataset rows FIRST -> strictly excludes items > ₹2000  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                 3. Vector Similarity Search (ChromaDB)                |
|   Calculates cosine similarity on filtered candidate product subset   |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                4. Hybrid Ranking Engine (src/ranking.py)              |
|   Combines embedding similarity + attribute match score + rating      |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                  5. RAG LLM Agent (src/agent.py - Ollama)             |
|   Generates concise personalized fashion recommendations & rationale  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|              6. Dark-Themed Streamlit UI (app/app.py)                 |
|   Product cards, score badges, sidebar filters & search history       |
+-----------------------------------------------------------------------+
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Frontend / UI**: Streamlit (Dark Theme CSS)
- **Data Processing**: Pandas, NumPy
- **Vector Database**: ChromaDB (Persistent storage)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) / Scikit-Learn SVD fallback
- **LLM / RAG Agent**: Ollama (`llama3.2:3b`)
- **Testing**: Pytest

### Tested Versions

| Component | Version(s) Tested | Minimum Required |
| --- | --- | --- |
| Python | 3.14.7 | 3.10+ |
| Streamlit | 1.30.0+ | 1.30.0 |
| ChromaDB | 0.4.0+ | 0.4.0 |
| Sentence Transformers | 2.2.0+ | 2.2.0 |
| Pandas | 2.0.0+ | 2.0.0 |
| Ollama | llama3.2:3b | (Optional) |
| PyTorch | 2.0.0+ | 2.0.0 (for SentenceTransformer) |

---

## 📊 Dataset Overview

The system includes a balanced synthetic dataset of **3,000 clothing products** stored in `data/clothing_products.csv`.

- **Categories**: shirts, t-shirts, jeans, trousers, dresses, skirts, jackets, hoodies, sweaters, tops, shorts, ethnic wear, activewear.
- **Attributes per Product**: `product_id`, `product_name`, `category`, `subcategory`, `gender`, `color`, `size`, `fit`, `material`, `brand`, `price_inr`, `rating`, `season`, `description`.
- **Top Brands**: Levi's, Zara, H&M, Nike, Roadster, FabIndia, Allen Solly, Puma, Adidas, US Polo Assn, Tommy Hilfiger, W, Biba, HRX, Jack & Jones.

---

## 📂 Project Structure

```text
AI Clothing/
├── app/
│   └── app.py                  # Streamlit Dark Theme Application
├── src/
│   ├── config.py               # Environment configuration & constants
│   ├── data_loader.py          # Pandas dataset loader & hard filter engine
│   ├── query_parser.py         # Natural language query attribute parser
│   ├── embedder.py             # Vector embedding engine
│   ├── retrieval.py            # Hard filtering + ChromaDB retriever
│   ├── ranking.py              # Hybrid ranking scoring engine
│   └── agent.py                # Ollama RAG LLM integration & fallback
├── data/
│   └── clothing_products.csv   # 3,000 product dataset
├── tests/
│   ├── test_cleaner.py         # Dataset loading unit tests
│   ├── test_filters.py         # Hard filtering unit tests
│   ├── test_query_parser.py    # Query parser unit tests
│   ├── test_ranking.py        # Hybrid ranker unit tests
│   └── test_pipeline.py      # End-to-end retrieval & RAG pipeline tests
├── generate_dataset.py         # Dataset generator script
├── build_index.py              # ChromaDB vector index builder
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .env                        # Active environment variables
├── .gitignore                  # Git ignore rules
├── DEBUGGING_REPORT.md         # Documented bug resolution report
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone or Open Project Directory

```bash
cd "AI Clothing"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Build Vector Index

```bash
python generate_dataset.py
python build_index.py
```

---

## 🧪 Testing & Evaluation

### Run Unit Tests (Fast - 7.3 seconds)

```bash
pytest tests/ -v -k "not agent_explanation"
```

#### Expected Output

```text
18 passed, 1 deselected in 7.33s
```

### Run Full Test Suite (With LLM Tests - 60+ seconds)

```bash
pytest tests/ -v
```

### Run Comprehensive System Evaluation

```bash
python evaluate_system.py
```

This comprehensive evaluation covers:

- ✅ **Budget Constraint Enforcement** - Ensures no product exceeds specified price limits
- ✅ **No-Match Handling** - Gracefully handles impossible filter combinations
- ✅ **Product Citations** - Verifies LLM rationale includes product IDs
- ✅ **Category Filtering** - Ensures category constraints are enforced
- ✅ **Gender Filtering** - Validates gender constraints with Unisex fallback
- ✅ **Vector Search Availability** - Checks ChromaDB index status
- ✅ **Metadata-Only Fallback** - Verifies fallback behavior when vector search unavailable
- ✅ **Latency Benchmarks** - Measures end-to-end performance

**Output**: `evaluation_results.json`

---

## 📋 Configuration

Edit `.env` to customize:

```bash
cp .env.example .env
```

### Key Settings

```env
# Ollama LLM Configuration
OLLAMA_MODEL_NAME=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60  # Seconds to wait for Ollama response

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
DATASET_PATH=./data/clothing_products.csv

# Retrieval
DEFAULT_TOP_K=4
DEFAULT_SIMILARITY_THRESHOLD=0.20
```

---

## 🦙 Ollama Local Setup (Optional but Recommended)

1. Download and install Ollama from [ollama.com](https://ollama.com).

2. Pull the recommended local model:

   ```bash
   ollama pull llama3.2:3b
   ```

3. Start the Ollama server:

   ```bash
   ollama serve
   ```

4. Verify configuration in `.env`:

   ```env
   OLLAMA_MODEL_NAME=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_TIMEOUT=60
   ```

> **Note**: If Ollama is not running, the application automatically uses the internal smart rule engine without throwing errors!

---

## 💻 Running the Streamlit Web Application

Start the Streamlit web server:

```bash
python -m streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 Example Queries to Try

- `blue slim fit jeans under 2000`
- `black casual t-shirts below 1500`
- `women's cotton dresses for summer`
- `men's oversized hoodies under 2500`
- `formal linen shirts above 2000`

---

## 🧪 Running Unit Tests

Run all unit test suites using Pytest:

```bash
python -m pytest tests/
```

---

## 🐞 Debugging Report

Refer to [DEBUGGING_REPORT.md](DEBUGGING_REPORT.md) for details on an edge-case bug identified and resolved during query parsing and numerical price boundary validation.

---

## 🖥️ Tested Platform & Dependency Compatibility

| Component | Tested Version / Specification |
| --- | --- |
| **Python** | 3.10, 3.11, 3.12, 3.13 |
| **Streamlit** | >= 1.30.0 |
| **ChromaDB** | >= 0.4.0 |
| **Sentence Transformers** | >= 2.2.0 (`all-MiniLM-L6-v2`) |
| **Pandas** | >= 2.0.0 |
| **PyTorch** | >= 2.0.0 |
| **Ollama Model** | `llama3.2:3b` |
