"""Dependency-light tests for semantic retrieval configuration."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from tools.assistant_retrieval import load_knowledge_base
from tools.assistant_semantic_retrieval import (
    knowledge_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
class AssistantSemanticConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)

    def test_knowledge_fingerprint_is_stable_and_content_sensitive(self) -> None:
        original = knowledge_fingerprint(self.sections)
        reordered = knowledge_fingerprint(list(reversed(self.sections)))
        shortened = knowledge_fingerprint(self.sections[:-1])

        self.assertEqual(original, reordered)
        self.assertNotEqual(original, shortened)
        self.assertEqual(len(original), 16)

    def test_semantic_search_runs_inside_qt_worker_thread(self) -> None:
        script = textwrap.dedent(
            """
            from pathlib import Path
            import NEAT
            from PyQt5.QtCore import QThread
            from tools.assistant_retrieval import load_knowledge_base
            from tools.assistant_semantic_retrieval import ChromaSemanticRetriever

            root = Path.cwd()

            class Worker(QThread):
                def __init__(self):
                    super().__init__()
                    self.error = None
                    self.result_count = 0

                def run(self):
                    try:
                        sections = load_knowledge_base(root / "docs" / "assistant")
                        retriever = ChromaSemanticRetriever(
                            sections,
                            persist_directory=root / ".assistant_cache" / "chroma",
                        )
                        self.result_count = len(retriever.search(
                            "What is the difference between individual edge fitting "
                            "and pattern fitting?",
                            limit=3,
                        ))
                    except Exception as exc:
                        self.error = exc

            worker = Worker()
            worker.start()
            if not worker.wait(30000):
                raise RuntimeError("Qt semantic-search worker timed out")
            if worker.error is not None:
                raise worker.error
            if worker.result_count != 3:
                raise AssertionError(worker.result_count)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
