import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.config import EMBEDDING_MODEL_NAME, BASE_DIR

EMBEDDER_CACHE_PATH = BASE_DIR / "chroma_db" / "tfidf_svd_embedder.pkl"

class VectorEmbedder:
    """
    Production-grade Vector Embedder.
    Attempts SentenceTransformer (all-MiniLM-L6-v2) first.
    If PyTorch/DLL environment errors occur on Windows, seamlessly falls back to 
    a 384-dimensional TF-IDF + TruncatedSVD dense semantic vector space.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, n_components: int = 384):
        self.model_name = model_name
        self.n_components = n_components
        self.st_model = None
        self.tfidf = None
        self.svd = None
        self.is_fitted = False

        # Attempt to load SentenceTransformer safely without forcing a remote download.
        # If the model is not already cached locally, we intentionally fall back to
        # the TF-IDF + SVD path instead of hanging while trying to fetch it.
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name, local_files_only=True)
            print(f"Loaded SentenceTransformer ('{model_name}') successfully!")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            self.st_model = None
            print(f"SentenceTransformer not available ({exc}). Using TF-IDF + TruncatedSVD dense semantic vector space.")

        # Load cached TF-IDF + SVD model if available and ST model is not used
        if self.st_model is None and EMBEDDER_CACHE_PATH.exists():
            self._load_cache()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fits embedding model on dataset and returns dense normalized float vectors."""
        if self.st_model is not None:
            vectors = self.st_model.encode(texts, show_progress_bar=False)
            return np.array(vectors, dtype=np.float32)
        else:
            self.tfidf = TfidfVectorizer(
                max_features=10000, 
                stop_words='english', 
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            sparse_matrix = self.tfidf.fit_transform(texts)

            # TruncatedSVD cannot fit a one-row or one-feature dataset.
            if sparse_matrix.shape[0] < 2 or sparse_matrix.shape[1] < 2:
                self.svd = None
                dense_vectors = sparse_matrix.toarray()
                if dense_vectors.shape[1] > self.n_components:
                    dense_vectors = dense_vectors[:, :self.n_components]
                elif dense_vectors.shape[1] < self.n_components:
                    dense_vectors = np.pad(
                        dense_vectors,
                        ((0, 0), (0, self.n_components - dense_vectors.shape[1])),
                    )
            else:
                dim = min(self.n_components, sparse_matrix.shape[1] - 1, sparse_matrix.shape[0] - 1)
                self.svd = TruncatedSVD(n_components=dim, random_state=42)
                dense_vectors = self.svd.fit_transform(sparse_matrix)

            # L2 Normalize vectors to unit length for cosine similarity
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = (dense_vectors / norms).astype(np.float32)

            self.is_fitted = True
            self._save_cache()
            return normalized

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes query or documents into dense normalized float vectors."""
        if self.st_model is not None:
            vectors = self.st_model.encode(texts, show_progress_bar=False)
            return np.array(vectors, dtype=np.float32)
        else:
            if not self.is_fitted or self.tfidf is None:
                if EMBEDDER_CACHE_PATH.exists():
                    self._load_cache()
                else:
                    # Fallback fit if cache doesn't exist yet
                    return self.fit_transform(texts)

            sparse_matrix = self.tfidf.transform(texts)
            if self.svd is None:
                dense_vectors = sparse_matrix.toarray()
                if dense_vectors.shape[1] > self.n_components:
                    dense_vectors = dense_vectors[:, :self.n_components]
                elif dense_vectors.shape[1] < self.n_components:
                    dense_vectors = np.pad(
                        dense_vectors,
                        ((0, 0), (0, self.n_components - dense_vectors.shape[1])),
                    )
            else:
                dense_vectors = self.svd.transform(sparse_matrix)
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (dense_vectors / norms).astype(np.float32)

    def _save_cache(self):
        try:
            EMBEDDER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDER_CACHE_PATH, "wb") as f:
                pickle.dump({"tfidf": self.tfidf, "svd": self.svd, "n_components": self.n_components}, f)
        except (OSError, TypeError, ValueError) as exc:
            print(f"Warning: Could not save embedder cache: {exc}")

    def _load_cache(self):
        try:
            with open(EMBEDDER_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
                self.tfidf = data["tfidf"]
                self.svd = data["svd"]
                self.n_components = data.get("n_components", 384)
                self.is_fitted = True
        except (FileNotFoundError, OSError, pickle.PickleError, ValueError, TypeError) as exc:
            print(f"Warning: Could not load embedder cache: {exc}")
            self.is_fitted = False
