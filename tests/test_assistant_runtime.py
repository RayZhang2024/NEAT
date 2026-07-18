"""Tests for release-safe assistant paths and offline retrieval fallback."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.assistant_retrieval import BM25Retriever, KnowledgeSection
from tools.assistant_runtime import (
    create_release_safe_retriever,
    default_assistant_index_directory,
)


class _UnavailableSemanticRetriever:
    def __init__(self, sections, *, persist_directory) -> None:
        raise RuntimeError("embedding model is unavailable offline")


class AssistantRuntimeTests(unittest.TestCase):
    def test_windows_index_uses_local_app_data_not_application_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LOCALAPPDATA": directory}, clear=False):
                path = default_assistant_index_directory()
        self.assertEqual(
            path,
            Path(directory) / "NEAT" / "assistant_cache" / "chroma",
        )

    def test_semantic_failure_falls_back_to_bundled_bm25(self) -> None:
        sections = [
            KnowledgeSection(
                source_id="faq.md#test",
                filename="faq.md",
                anchor="test",
                heading="Test",
                heading_path="FAQ > Test",
                text="Macro-pixel fitting combines neighbouring pixels.",
            )
        ]
        retriever = create_release_safe_retriever(
            sections,
            semantic_retriever_class=_UnavailableSemanticRetriever,
        )
        self.assertIsInstance(retriever, BM25Retriever)
        self.assertEqual(len(retriever.search("macro pixel", limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
