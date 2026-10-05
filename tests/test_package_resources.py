"""Contracts for package-owned assistant and splash resources."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from NEAT.package_resources import assistant_knowledge_root, launch_splash_resource
from tools.assistant_retrieval import KNOWLEDGE_FILENAMES, load_knowledge_base


class _MemoryFile:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def is_file(self) -> bool:
        return True

    def read_text(self, encoding: str = "utf-8") -> str:
        if encoding != "utf-8":
            raise AssertionError(encoding)
        return self._text


class _MissingFile:
    def is_file(self) -> bool:
        return False


class _MemoryDirectory:
    def __init__(self, files: dict[str, _MemoryFile]) -> None:
        self._files = files

    def joinpath(self, child: str) -> _MemoryFile | _MissingFile:
        return self._files.get(child, _MissingFile())


class PackageResourceTests(unittest.TestCase):
    def test_knowledge_root_resolves_from_top_level_package_resource(self) -> None:
        package_root = Mock()
        knowledge_root = object()
        package_root.joinpath.return_value = knowledge_root

        with patch("NEAT.package_resources.files", return_value=package_root) as files:
            result = assistant_knowledge_root()

        files.assert_called_once_with("NEAT")
        package_root.joinpath.assert_called_once_with("knowledge")
        self.assertIs(result, knowledge_root)

    def test_source_package_resource_contains_exact_approved_corpus(self) -> None:
        root = assistant_knowledge_root()
        markdown_files = {item.name for item in root.iterdir() if item.name.endswith(".md")}
        self.assertEqual(markdown_files, set(KNOWLEDGE_FILENAMES))
        sections = load_knowledge_base(root)
        self.assertTrue(sections)
        self.assertEqual({section.filename for section in sections}, set(KNOWLEDGE_FILENAMES))
        self.assertTrue(
            all(section.source_id.startswith(section.filename.lower() + "#") for section in sections)
        )

    def test_loader_accepts_traversable_style_resources(self) -> None:
        directory = _MemoryDirectory(
            {
                filename: _MemoryFile(filename, f"# Guide\n\n## {filename}\n\nAdvice")
                for filename in KNOWLEDGE_FILENAMES
            }
        )
        sections = load_knowledge_base(directory)  # type: ignore[arg-type]
        self.assertEqual(len(sections), len(KNOWLEDGE_FILENAMES))
        self.assertTrue(all(section.text.endswith("Advice") for section in sections))

    def test_path_override_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in KNOWLEDGE_FILENAMES:
                (root / filename).write_text(
                    f"# Guide\n\n## {filename}\n\nAdvice", encoding="utf-8"
                )
            sections = load_knowledge_base(root)
        self.assertEqual({section.filename for section in sections}, set(KNOWLEDGE_FILENAMES))

    def test_missing_allow_list_file_fails_closed(self) -> None:
        directory = _MemoryDirectory(
            {
                filename: _MemoryFile(filename, f"## {filename}\n\nAdvice")
                for filename in KNOWLEDGE_FILENAMES[:-1]
            }
        )
        with self.assertRaisesRegex(FileNotFoundError, KNOWLEDGE_FILENAMES[-1]):
            load_knowledge_base(directory)  # type: ignore[arg-type]

    def test_launch_splash_resource_is_readable(self) -> None:
        splash = launch_splash_resource()
        self.assertTrue(splash.is_file())
        self.assertTrue(splash.read_bytes())


if __name__ == "__main__":
    unittest.main()
