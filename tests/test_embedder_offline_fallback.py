import sys
import types


def test_vector_embedder_uses_local_only_fallback(monkeypatch):
    calls = []

    def fake_sentence_transformer(model_name, **kwargs):
        calls.append({"model_name": model_name, **kwargs})
        raise OSError("offline model not cached")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import src.embedder as embedder_module

    cache_path = embedder_module.EMBEDDER_CACHE_PATH
    if cache_path.exists():
        cache_path.unlink()

    embedder = embedder_module.VectorEmbedder()

    assert calls
    assert calls[0].get("local_files_only") is True
    assert embedder.st_model is None

def test_vector_embedder_handles_one_product_fallback(monkeypatch):
    def fake_sentence_transformer(model_name, **kwargs):
        raise OSError("offline model not cached")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from src.embedder import VectorEmbedder

    embedder = VectorEmbedder()
    vectors = embedder.fit_transform(["blue cotton shirt"])
    query_vector = embedder.encode(["blue shirt"])

    assert vectors.shape == (1, 384)
    assert query_vector.shape == (1, 384)
