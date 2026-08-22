import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class VectorEmbedder:
    """
    Robust embedding engine. Attempts SentenceTransformer first.
    If PyTorch/DLL issues occur on Python 3.14, seamlessly uses TF-IDF + SVD
    dense 384-dimensional semantic embedding model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_components: int = 384):
        self.model_name = model_name
        self.st_model = None
        self.tfidf = None
        self.svd = None
        self.n_components = n_components
        
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name)
            print("Successfully initialized SentenceTransformer model!")
        except Exception as e:
            print(f"SentenceTransformer unavailable ({e}), utilizing TF-IDF + SVD dense semantic vectorizer.")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.st_model is not None:
            return self.st_model.encode(texts, show_progress_bar=False)
        else:
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            sparse_matrix = self.tfidf.fit_transform(texts)
            dim = min(self.n_components, sparse_matrix.shape[1] - 1, sparse_matrix.shape[0] - 1)
            dim = max(16, dim)
            self.svd = TruncatedSVD(n_components=dim, random_state=42)
            dense_vectors = self.svd.fit_transform(sparse_matrix)
            # Normalize to unit length for cosine similarity
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense_vectors / norms

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.st_model is not None:
            return self.st_model.encode(texts, show_progress_bar=False)
        else:
            if self.tfidf is None or self.svd is None:
                raise ValueError("Embedder has not been fitted yet.")
            sparse_matrix = self.tfidf.transform(texts)
            dense_vectors = self.svd.transform(sparse_matrix)
            norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense_vectors / norms

if __name__ == "__main__":
    embedder = VectorEmbedder()
    sample_texts = ["blue slim fit jeans", "black casual t-shirt", "cotton summer dress"]
    vecs = embedder.fit_transform(sample_texts)
    print("Embedded shape:", vecs.shape)
