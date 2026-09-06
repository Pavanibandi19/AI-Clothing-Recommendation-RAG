import sys
import argparse
import logging
from pathlib import Path
import chromadb

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, 
    DATASET_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)
from src.data_loader import DataLoader
from src.rag.chunker import ProductChunker
from src.embedder import VectorEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_vector_store")


def build_vector_store(
    dataset_path: str = DATASET_PATH,
    chroma_dir: str = CHROMA_PERSIST_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME,
    limit: int = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
):
    print("=" * 60)
    print("[RECOMAI] CHROMA VECTOR STORE INGESTION PIPELINE")
    print("=" * 60)

    # 1. Load Cleaned Products Dataset
    print(f"\n[Step 1] Loading products from {dataset_path}...")
    loader = DataLoader(dataset_path)
    df = loader.get_all_products()
    if limit and limit > 0:
        df = df.head(limit).copy()
    product_count = len(df)
    print(f"   --> Loaded {product_count} cleaned products.")

    # 2. Structured Semantic Chunking
    print(f"\n[Step 2] Running Structured Semantic Chunking (size={chunk_size}, overlap={chunk_overlap})...")
    chunker = ProductChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_dataset(df)
    chunk_count = len(chunks)
    avg_chunks = chunk_count / product_count if product_count > 0 else 0.0

    print(f"   --> Created {chunk_count} semantic chunks from {product_count} products.")
    print(f"   --> Average chunks per product: {avg_chunks:.2f}")

    # Extract chunk lists for ChromaDB
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]
    chunk_metadatas = [c.metadata for c in chunks]

    # 3. Dense Vector Embeddings Generation
    print(f"\n[Step 3] Generating dense vector embeddings via '{EMBEDDING_MODEL_NAME}'...")
    embedder = VectorEmbedder()
    embeddings = embedder.fit_transform(chunk_texts)
    embedding_dim = embeddings.shape[1]
    print(f"   --> Generated {len(embeddings)} normalized embeddings (Dimension: {embedding_dim}).")

    # 4. ChromaDB Vector Store Persistence
    print(f"\n[Step 4] Connecting to ChromaDB at '{chroma_dir}'...")
    client = chromadb.PersistentClient(path=chroma_dir)

    # Clean existing collection if it exists to ensure clean index
    try:
        client.delete_collection(name=collection_name)
        logger.info("Previous collection '%s' cleared for fresh rebuild.", collection_name)
    except (ValueError, RuntimeError, TypeError, Exception) as exc:
        logger.debug("Creating new collection (previous notice: %s)", exc)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"   --> Inserting {chunk_count} chunk vectors in batches into collection '{collection_name}'...")
    batch_size = 400
    for i in range(0, chunk_count, batch_size):
        end = i + batch_size
        collection.add(
            ids=chunk_ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=chunk_metadatas[i:end],
            documents=chunk_texts[i:end]
        )
        print(f"       Indexed chunk {i+1} to {min(end, chunk_count)} / {chunk_count}...")

    final_count = collection.count()

    print("\n" + "=" * 60)
    print("[SUCCESS] INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Products:                {product_count}")
    print(f"Chunks created:          {chunk_count}")
    print(f"Average chunks/product:  {avg_chunks:.2f}")
    print(f"Embedding model:         {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension:     {embedding_dim}")
    print(f"ChromaDB collection:     {collection_name}")
    print(f"Indexed records count:   {final_count}")
    print("=" * 60 + "\n")

    return {
        "product_count": product_count,
        "chunk_count": chunk_count,
        "avg_chunks_per_product": avg_chunks,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": embedding_dim,
        "collection_name": collection_name,
        "indexed_count": final_count
    }


def main():
    parser = argparse.ArgumentParser(description="RECOMAI - Build ChromaDB Chunk Vector Store")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to products CSV dataset")
    parser.add_argument("--chroma-dir", type=str, default=CHROMA_PERSIST_DIRECTORY, help="Path to ChromaDB directory")
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of products to index")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for product document splitting")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap")

    args = parser.parse_args()
    build_vector_store(
        dataset_path=args.dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        limit=args.limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )


if __name__ == "__main__":
    main()
