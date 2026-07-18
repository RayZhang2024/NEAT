"""Release-safe paths and retrieval fallback for the desktop assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from tools.assistant_retrieval import (
    BM25Retriever,
    KnowledgeSection,
    SectionRetriever,
)


_AUTO_SEMANTIC_RETRIEVER = object()


def default_assistant_index_directory() -> Path:
    """Return a writable per-user location for the semantic-search index."""

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        base_directory = Path(local_app_data)
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        base_directory = (
            Path(cache_home).expanduser()
            if cache_home
            else Path.home() / ".cache"
        )
    return base_directory / "NEAT" / "assistant_cache" / "chroma"


def prepare_optional_semantic_runtime() -> bool:
    """Prepare semantic retrieval, returning False when lexical fallback is needed."""

    try:
        from tools.assistant_semantic_retrieval import (
            prepare_local_embedding_runtime,
        )

        prepare_local_embedding_runtime()
    except Exception:
        return False
    return True


def create_release_safe_retriever(
    sections: Sequence[KnowledgeSection],
    *,
    persist_directory: Path | None = None,
    semantic_retriever_class=_AUTO_SEMANTIC_RETRIEVER,
) -> SectionRetriever:
    """Prefer semantic search and fall back to bundled BM25 when unavailable."""

    if semantic_retriever_class is _AUTO_SEMANTIC_RETRIEVER:
        try:
            from tools.assistant_semantic_retrieval import ChromaSemanticRetriever
        except Exception:
            return BM25Retriever(sections)
        semantic_retriever_class = ChromaSemanticRetriever

    try:
        return semantic_retriever_class(
            sections,
            persist_directory=(
                Path(persist_directory)
                if persist_directory is not None
                else default_assistant_index_directory()
            ),
        )
    except Exception:
        # First-use model downloads and native ONNX loading can fail on offline
        # or restricted machines. BM25 uses the same approved sources and keeps
        # the assistant functional without weakening grounding.
        return BM25Retriever(sections)


__all__ = [
    "create_release_safe_retriever",
    "default_assistant_index_directory",
    "prepare_optional_semantic_runtime",
]
