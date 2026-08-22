import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Ollama LLM Settings
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "5"))

# Vector Database & Embedding Settings
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "clothing_products.csv"))

# Retrieval Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.20"))

# Category Taxonomy Definition
VALID_CATEGORIES = [
    "shirts", "t-shirts", "jeans", "trousers", "dresses", 
    "skirts", "jackets", "hoodies", "sweaters", "tops", 
    "shorts", "ethnic wear", "activewear"
]

VALID_GENDERS = ["Men", "Women", "Unisex"]

VALID_FITS = ["Slim Fit", "Regular Fit", "Relaxed Fit", "Oversized", "Skinny Fit"]

VALID_MATERIALS = ["Cotton", "Denim", "Polyester", "Linen", "Wool", "Silk", "Rayon", "Leather"]

VALID_COLORS = [
    "Black", "Blue", "White", "Red", "Green", 
    "Navy", "Grey", "Beige", "Yellow", "Pink", 
    "Olive", "Maroon", "Brown", "Purple"
]

VALID_SEASONS = ["Summer", "Winter", "Spring", "Autumn", "All-Season"]
