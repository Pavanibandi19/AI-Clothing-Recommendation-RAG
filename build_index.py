import sys
import logging
from pathlib import Path
import chromadb

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.config import CHROMA_PERSIST_DIRECTORY, DATASET_PATH
from src.data_loader import DataLoader
from src.embedder import VectorEmbedder

logger = logging.getLogger(__name__)

def build_vector_index():
    print(f"Loading dataset from {DATASET_PATH}...")
    loader = DataLoader(DATASET_PATH)
    df = loader.get_all_products()
    print(f"Loaded {len(df)} products for vector indexing.")

    embedder = VectorEmbedder()

    # Build rich text document strings
    documents = []
    ids = []
    metadatas = []

    for _, row in df.iterrows():
        doc_text = (
            f"Product: {row['product_name']} | Category: {row['category']} | "
            f"Subcategory: {row['subcategory']} | Gender: {row['gender']} | "
            f"Color: {row['color']} | Fit: {row['fit']} | Material: {row['material']} | "
            f"Brand: {row['brand']} | Season: {row['season']} | Price: Rs {row['price_inr']} | "
            f"Description: {row['description']}"
        )
        documents.append(doc_text)
        ids.append(str(row['product_id']))

        meta = {
            "product_id": str(row['product_id']),
            "category": str(row['category']).lower(),
            "subcategory": str(row['subcategory']).lower(),
            "gender": str(row['gender']).lower(),
            "color": str(row['color']).lower(),
            "fit": str(row['fit']).lower(),
            "material": str(row['material']).lower(),
            "brand": str(row['brand']).lower(),
            "season": str(row['season']).lower(),
            "price_inr": float(row['price_inr']),
            "rating": float(row['rating'])
        }
        metadatas.append(meta)

    print("Generating dense vector embeddings...")
    embeddings = embedder.fit_transform(documents)

    print(f"Initializing persistent ChromaDB at {CHROMA_PERSIST_DIRECTORY}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

    # Delete existing collection if present
    try:
        client.delete_collection(name="clothing_products")
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.info("Creating new collection; previous delete notice: %s", exc)

    collection = client.create_collection(
        name="clothing_products",
        metadata={"hnsw:space": "cosine"}
    )

    # Insert in batches of 500
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=documents[i:end]
        )

    print(f"[SUCCESS] ChromaDB index built successfully with {collection.count()} products!")

if __name__ == "__main__":
    build_vector_index()
