"""Local LangChain-Chroma semantic retrieval for the NEAT assistant."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

try:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError as exc:  # pragma: no cover - depends on optional installation
    raise ImportError(
        "Semantic retrieval requires the optional assistant dependencies. "
        "Install them with: python -m pip install -e \".[assistant]\""
    ) from exc

from tools.assistant_retrieval import (
    KnowledgeSection,
    SearchResult,
)


MODEL_NAME = "all-MiniLM-L6-v2"


def prepare_local_embedding_runtime() -> None:
    """Load ONNX Runtime on the process main thread before Qt workers use it.

    On Windows, first importing ONNX Runtime from a QThread can fail during DLL
    initialization even though the package is installed correctly.
    """

    try:
        import onnxruntime  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The local semantic-search runtime could not start. Reinstall the "
            "assistant dependencies and confirm that ONNX Runtime can load."
        ) from exc


class LocalMiniLMEmbeddings(Embeddings):
    """Adapt Chroma's local ONNX MiniLM model to LangChain's interface."""

    def __init__(self) -> None:
        self._model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

    @staticmethod
    def _to_lists(vectors) -> list[list[float]]:
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._to_lists(self._model(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def knowledge_fingerprint(sections: Sequence[KnowledgeSection]) -> str:
    """Return a stable fingerprint so changed knowledge gets a new collection."""

    digest = hashlib.sha256()
    for section in sorted(sections, key=lambda item: item.source_id):
        digest.update(section.source_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(section.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class ChromaSemanticRetriever:
    """Persistent semantic retriever backed by LangChain and local Chroma."""

    def __init__(
        self,
        sections: Sequence[KnowledgeSection],
        *,
        persist_directory: Path,
        rebuild: bool = False,
    ) -> None:
        if not sections:
            raise ValueError("At least one knowledge section is required")

        self.sections = list(sections)
        self.persist_directory = persist_directory.resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = knowledge_fingerprint(self.sections)
        self.collection_name = f"neat_assistant_{self.fingerprint}"
        self.embedding_model = MODEL_NAME
        self._section_by_source = {
            section.source_id: section for section in self.sections
        }
        self._embeddings = LocalMiniLMEmbeddings()
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.persist_directory),
        )
        self.reused_existing_index = self._prepare_collection(rebuild=rebuild)

    def _prepare_collection(self, *, rebuild: bool) -> bool:
        expected_ids = set(self._section_by_source)
        existing = self._vector_store.get(include=[])
        existing_ids = set(existing.get("ids") or [])

        if not rebuild and existing_ids == expected_ids:
            return True

        if existing_ids:
            self._vector_store.reset_collection()

        documents = [
            Document(
                page_content=section.text,
                metadata={
                    "source_id": section.source_id,
                    "filename": section.filename,
                    "anchor": section.anchor,
                    "heading": section.heading,
                    "heading_path": section.heading_path,
                    "knowledge_fingerprint": self.fingerprint,
                },
            )
            for section in self.sections
        ]
        self._vector_store.add_documents(
            documents=documents,
            ids=[section.source_id for section in self.sections],
        )
        return False

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        """Return semantic matches, with higher normalized scores ranked first."""

        if limit <= 0:
            return []

        matches = self._vector_store.similarity_search_with_score(query, k=limit)
        results = []
        for document, distance in matches:
            source_id = str(document.metadata.get("source_id", ""))
            section = self._section_by_source.get(source_id)
            if section is None:
                continue
            normalized_score = 1.0 / (1.0 + max(float(distance), 0.0))
            results.append(SearchResult(section=section, score=normalized_score))
        return results
