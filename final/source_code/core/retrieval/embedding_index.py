import os
import json
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from core.cache import CacheManager

@dataclass
class SemanticMatch:
    window_index: int
    score: float

class EmbeddingIndex:
    """Dense vector index backed by sentence-transformers and FAISS."""

    DEFAULT_MODEL = "intfloat/multilingual-e5-large"
    CACHE_DIR = "outputs/cache/embeddings"

    def __init__(
        self,
        documents: list[str],
        model_name: str | None = None,
        cache_key: str = "",
    ):
        self.documents = [doc or "" for doc in documents]
        self.enabled = bool(self.documents)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_key = cache_key
        self.cache_manager = CacheManager(run_key="embedding_index")
        self.model = None
        self.index = None
        self.document_embeddings: np.ndarray | None = None
        self._dimension = 0
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if self.enabled:
            self._build_index()

    def search(
        self,
        query_text: str,
        allowed_indexes: set[int] | None = None,
        top_k: int = 20,
        min_score: float = 0.25,
    ) -> list[SemanticMatch]:
        if not self.enabled or not query_text.strip() or self.index is None:
            return []

        query_embedding = self._encode_query(query_text)
        if query_embedding is None:
            return []

        search_k = min(top_k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding, search_k)

        matches = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if allowed_indexes is not None and int(idx) not in allowed_indexes:
                continue
            if float(score) < min_score:
                continue
            matches.append(SemanticMatch(window_index=int(idx), score=round(float(score), 4)))
            if len(matches) >= top_k:
                break

        return matches

    def _build_index(self) -> None:
        cached = self._load_cache()
        if cached is not None:
            self.document_embeddings = cached
            self._dimension = cached.shape[1]
            self._build_faiss_index(cached)
            print(f"  [EMBEDDING] Loaded cached embeddings: {cached.shape}")
            return

        self._load_model()
        print(f"  [EMBEDDING] Encoding {len(self.documents)} documents with {self.model_name}...")

        # Prefix for e5 models
        prefixed_docs = [f"passage: {doc}" for doc in self.documents]

        self.document_embeddings = self.model.encode(
            prefixed_docs,
            batch_size=int(os.environ.get("ESG_EMBEDDING_BATCH_SIZE", "64")),
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._dimension = self.document_embeddings.shape[1]
        self._build_faiss_index(self.document_embeddings)
        self._save_cache(self.document_embeddings)
        print(f"  [EMBEDDING] Index built: {self.document_embeddings.shape}, dim={self._dimension}")

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        import faiss
        self._dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self._dimension)
        self.index.add(embeddings.astype(np.float32))

    def _encode_query(self, query_text: str) -> np.ndarray | None:
        self._load_model()
        prefixed = f"query: {query_text}"
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import torch

        # Force CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [EMBEDDING] Loading model {self.model_name} on {device.upper()}...")
        
        local_only = os.environ.get("ESG_EMBEDDING_LOCAL_ONLY", "1") != "0"
        
        self.model = SentenceTransformer(
            self.model_name,
            local_files_only=local_only,
            device=device,
        )

    def _cache_path(self) -> str:
        content_hash = self._cache_fingerprint()[:16]
        return os.path.join(self.CACHE_DIR, f"emb_{content_hash}.npy")

    def _cache_fingerprint(self) -> str:
        return CacheManager.hash_json({
            "schema_version": "embedding_index_v2",
            "cache_key": self.cache_key,
            "document_count": len(self.documents),
            "model_name": self.model_name,
        })

    def _load_cache(self) -> np.ndarray | None:
        if not self.cache_key: return None
        path = self._cache_path()
        if os.path.exists(path):
            try:
                return np.load(path)
            except:
                return None
        return None

    def _save_cache(self, embeddings: np.ndarray) -> None:
        if not self.cache_key: return
        np.save(self._cache_path(), embeddings)
