"""Tests for the immutable structure and zero-copy image payload contract."""

import ast
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np

from NEAT.domain import LoadedImageRun


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SOURCE = REPOSITORY_ROOT / "NEAT" / "domain" / "preprocessing_inputs.py"


class ArraySubclass(np.ndarray):
    """An ndarray subclass used to protect the public isinstance contract."""


class LoadedImageRunConstructionTests(unittest.TestCase):
    def test_single_source_defaults_provenance_and_accepts_empty_frames(self):
        run = LoadedImageRun("synthetic/nonexistent", {})

        self.assertEqual(run.primary_source, "synthetic/nonexistent")
        self.assertEqual(dict(run.frames), {})
        self.assertEqual(run.source_folders, ("synthetic/nonexistent",))
        self.assertEqual(run.load_errors, ())

    def test_multi_source_run_keeps_independent_primary_identity_and_errors(self):
        frames = {"edge-A": np.ones((2, 3))}
        run = LoadedImageRun(
            "logical-sample",
            frames,
            source_folders=["physical/run-2", "physical/run-1"],
            load_errors=["one file was unreadable"],
        )

        self.assertEqual(run.primary_source, "logical-sample")
        self.assertEqual(tuple(run.frames), ("edge-A",))
        self.assertEqual(run.source_folders, ("physical/run-2", "physical/run-1"))
        self.assertEqual(run.load_errors, ("one file was unreadable",))

    def test_primary_source_must_be_nonempty_string_but_is_not_resolved(self):
        with self.assertRaises(ValueError):
            LoadedImageRun("", {})
        with self.assertRaises(TypeError):
            LoadedImageRun(123, {})  # type: ignore[arg-type]

        synthetic = "not-on-disk/../still-not-normalized"
        self.assertEqual(LoadedImageRun(synthetic, {}).primary_source, synthetic)

    def test_frames_requires_mapping_with_nonempty_string_keys_and_ndarrays(self):
        with self.assertRaises(TypeError):
            LoadedImageRun("source", [("001", np.ones((1,)))])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LoadedImageRun("source", {"": np.ones((1,))})
        with self.assertRaises(TypeError):
            LoadedImageRun("source", {1: np.ones((1,))})  # type: ignore[dict-item]
        with self.assertRaises(TypeError):
            LoadedImageRun("source", {"001": [1, 2]})  # type: ignore[dict-item]

    def test_nonnumeric_keys_ndarray_subclasses_and_unvalidated_shapes_are_valid(self):
        frame = np.array(17).view(ArraySubclass)
        run = LoadedImageRun("source", {"not-numeric": frame})

        self.assertIs(run.frames["not-numeric"], frame)
        self.assertIsInstance(run.frames["not-numeric"], ArraySubclass)
        self.assertEqual(run.frames["not-numeric"].shape, ())

    def test_mapping_snapshot_preserves_order_and_array_references(self):
        array_b = np.array([11])
        array_a = np.array([22])
        replacement = np.array([33])
        original = {"b": array_b, "a": array_a}
        run = LoadedImageRun("source", original)

        original["b"] = replacement
        del original["a"]
        original["c"] = np.array([44])

        self.assertEqual(tuple(run.frames), ("b", "a"))
        self.assertIs(run.frames["b"], array_b)
        self.assertIs(run.frames["a"], array_a)
        self.assertIsNot(run.frames["b"], replacement)

    def test_frame_arrays_are_zero_copy_and_remain_mutable(self):
        array = np.array([[3, 4]])
        run = LoadedImageRun("source", {"001": array})

        self.assertIs(run.frames["001"], array)
        array[0, 1] = 99
        self.assertEqual(run.frames["001"][0, 1], 99)


class LoadedImageRunProvenanceTests(unittest.TestCase):
    def test_source_folder_order_and_caller_snapshot_are_preserved(self):
        folders = ["run-b", "run-a", "run-b"]
        run = LoadedImageRun("logical", {}, source_folders=folders)
        folders.reverse()
        folders.append("run-c")

        self.assertEqual(run.source_folders, ("run-b", "run-a", "run-b"))

    def test_source_folders_reject_invalid_collection_shapes_and_values(self):
        for invalid in ("folder", b"folder", {"folder"}, frozenset({"folder"})):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                LoadedImageRun("source", {}, source_folders=invalid)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            LoadedImageRun("source", {}, source_folders=[])
        with self.assertRaises(ValueError):
            LoadedImageRun("source", {}, source_folders=["run", ""])
        with self.assertRaises(TypeError):
            LoadedImageRun("source", {}, source_folders=["run", 2])  # type: ignore[list-item]

    def test_load_error_order_snapshot_and_empty_sequence(self):
        errors = ["first", "second"]
        run = LoadedImageRun("source", {}, load_errors=errors)
        errors.reverse()

        self.assertEqual(run.load_errors, ("first", "second"))
        self.assertEqual(LoadedImageRun("source", {}, load_errors=[]).load_errors, ())

    def test_load_errors_reject_unordered_text_and_nonstring_inputs(self):
        for invalid in ("one error", b"one error", {"one"}, frozenset({"one"})):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                LoadedImageRun("source", {}, load_errors=invalid)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LoadedImageRun("source", {}, load_errors=["ok", None])  # type: ignore[list-item]


class LoadedImageRunImmutabilityTests(unittest.TestCase):
    def test_attributes_mapping_and_snapshotted_sequences_are_read_only(self):
        run = LoadedImageRun(
            "source", {"a": np.ones((1,))}, ["folder"], ["warning"]
        )

        with self.assertRaises(FrozenInstanceError):
            run.primary_source = "other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            run.frames["b"] = np.ones((1,))  # type: ignore[index]
        with self.assertRaises(TypeError):
            del run.frames["a"]  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            run.source_folders.append("other")  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            run.load_errors.append("other")  # type: ignore[attr-defined]

    def test_equality_uses_object_identity_not_numpy_array_values(self):
        first = LoadedImageRun("source", {"a": np.array([1, 2])})
        second = LoadedImageRun("source", {"a": np.array([1, 2])})

        self.assertIs(first, first)
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)

    def test_repr_is_concise_and_omits_array_payloads(self):
        run = LoadedImageRun("source", {"a": np.array([987654321])})
        rendered = repr(run)

        self.assertIn("LoadedImageRun", rendered)
        self.assertIn("frame_count=1", rendered)
        self.assertIn("source_folders", rendered)
        self.assertNotIn("987654321", rendered)
        self.assertLess(len(rendered), 250)

    def test_construction_does_not_query_or_access_the_filesystem(self):
        with (
            patch("builtins.open", side_effect=AssertionError("filesystem accessed")),
            patch("os.path.exists", side_effect=AssertionError("filesystem queried")),
            patch("os.listdir", side_effect=AssertionError("filesystem queried")),
            patch("os.scandir", side_effect=AssertionError("filesystem queried")),
            patch("os.stat", side_effect=AssertionError("filesystem queried")),
            patch.object(Path, "exists", side_effect=AssertionError("filesystem queried")),
            patch.object(Path, "is_file", side_effect=AssertionError("filesystem queried")),
            patch.object(Path, "is_dir", side_effect=AssertionError("filesystem queried")),
            patch.object(Path, "stat", side_effect=AssertionError("filesystem queried")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("filesystem read")),
            patch.object(Path, "write_bytes", side_effect=AssertionError("filesystem write")),
            patch.object(Path, "read_text", side_effect=AssertionError("filesystem read")),
            patch.object(Path, "write_text", side_effect=AssertionError("filesystem write")),
        ):
            run = LoadedImageRun("synthetic/nonexistent", {"a": np.ones((1,))})

        self.assertEqual(run.primary_source, "synthetic/nonexistent")


class LoadedImageRunDependencyTests(unittest.TestCase):
    def test_contract_source_has_no_qt_or_filesystem_dependency(self):
        tree = ast.parse(CONTRACT_SOURCE.read_text(encoding="utf-8"))
        forbidden_imports = {
            "PyQt5",
            "NEAT.ui",
            "NEAT.workers",
            "pathlib",
            "os",
            "io",
            "shutil",
        }
        forbidden_io_calls = {
            "open",
            "exists",
            "is_file",
            "is_dir",
            "stat",
            "resolve",
            "absolute",
            "iterdir",
            "glob",
            "rglob",
            "listdir",
            "scandir",
            "walk",
            "read_bytes",
            "read_text",
            "write_bytes",
            "write_text",
            "load",
            "save",
            "memmap",
            "fromfile",
            "tofile",
            "unlink",
            "mkdir",
            "rmdir",
        }

        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules = [node.module]
            for module_name in imported_modules:
                self.assertFalse(
                    any(
                        module_name == prefix or module_name.startswith(prefix + ".")
                        for prefix in forbidden_imports
                    ),
                    module_name,
                )
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, {"FitsViewer", "QApplication"})
            if isinstance(node, ast.Call):
                call_name = (
                    node.func.id if isinstance(node.func, ast.Name) else None
                )
                call_attribute = (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None
                )
                self.assertNotIn(call_name, forbidden_io_calls)
                self.assertNotIn(call_attribute, forbidden_io_calls)

    def test_import_and_construction_need_no_qt_application(self):
        code = """\
import sys
import numpy as np
from NEAT.domain import LoadedImageRun
run = LoadedImageRun('synthetic-source', {'frame': np.ones((1,))})
assert run.frames['frame'].shape == (1,)
forbidden = ('PyQt5', 'NEAT.ui', 'NEAT.workers')
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules for prefix in forbidden
)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
