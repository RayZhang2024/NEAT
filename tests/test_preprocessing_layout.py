"""Headless regression tests for preprocessing folder-layout discovery."""

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from NEAT.services import preprocessing_layout
from NEAT.services.preprocessing_layout import (
    classify_standalone_summation,
    discover_classic_batch,
    immediate_child_directories,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PreprocessingLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def make_dirs(self, *relative_paths):
        paths = []
        for relative_path in relative_paths:
            path = self.root / relative_path
            path.mkdir(parents=True, exist_ok=True)
            paths.append(str(path))
        return paths

    def test_no_immediate_children_classic_batch_uses_selected_folder(self):
        discovery = discover_classic_batch(str(self.root))
        self.assertEqual(discovery.folders, [str(self.root)])
        self.assertFalse(discovery.has_child_folders)

    def test_immediate_children_count_even_when_empty_or_unrelated(self):
        children = self.make_dirs("empty", "unrelated")
        (self.root / "ordinary-file.txt").write_text("not inspected", encoding="utf-8")
        discovery = discover_classic_batch(str(self.root))
        self.assertCountEqual(discovery.folders, children)
        self.assertTrue(discovery.has_child_folders)

    def test_grandchildren_are_not_promoted_to_classic_datasets(self):
        child, = self.make_dirs("dataset/run")
        discovery = discover_classic_batch(str(self.root))
        self.assertEqual(discovery.folders, [str(self.root / "dataset")])
        self.assertNotIn(child, discovery.folders)

    def test_filesystem_enumeration_order_is_preserved_not_sorted(self):
        first, second = self.make_dirs("alpha", "zeta")
        original_listdir = os.listdir

        def ordered_listdir(path):
            if os.fspath(path) == str(self.root):
                return ["zeta", "alpha"]
            return original_listdir(path)

        with patch.object(preprocessing_layout.os, "listdir", side_effect=ordered_listdir):
            discovery = discover_classic_batch(str(self.root))
        self.assertEqual(discovery.folders, [second, first])

    def test_shared_classic_discovery_is_wired_to_all_three_call_sites(self):
        source = (REPOSITORY_ROOT / "NEAT/ui/mixins/preprocessing.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        wanted = {
            "add_outlier_images",
            "add_overlap_correction_images",
            "add_normalisation_data_images",
        }
        methods = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        }
        self.assertEqual(set(methods), wanted)
        for method_source in methods.values():
            self.assertIn("discover_classic_batch", method_source)

    def test_standalone_summation_requires_two_immediate_children(self):
        self.assertEqual(classify_standalone_summation(str(self.root)).kind, "too_few")
        self.make_dirs("only-child")
        layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "too_few")
        self.assertEqual(layout.sample_folders, [str(self.root / "only-child")])

    def test_pure_two_level_summation_layout(self):
        samples = self.make_dirs("run-a", "run-b")
        layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "two_level")
        self.assertEqual(layout.sample_folders, samples)
        self.assertEqual(layout.run_folders_by_sample, {})

    def test_pure_three_level_summation_layout_keeps_run_order(self):
        sample_a, sample_b = self.make_dirs("sample-a", "sample-b")
        run_a1, run_a2 = self.make_dirs("sample-a/run-a1", "sample-a/run-a2")
        run_b1, run_b2 = self.make_dirs("sample-b/run-b1", "sample-b/run-b2")
        original_listdir = os.listdir

        def ordered_listdir(path):
            if os.fspath(path) == str(self.root):
                return ["sample-b", "sample-a"]
            if os.fspath(path) == sample_a:
                return ["run-a2", "run-a1"]
            if os.fspath(path) == sample_b:
                return ["run-b2", "run-b1"]
            return original_listdir(path)

        with patch.object(preprocessing_layout.os, "listdir", side_effect=ordered_listdir):
            layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "three_level")
        self.assertEqual(layout.sample_folders, [sample_b, sample_a])
        self.assertEqual(layout.run_folders_by_sample[sample_a], [run_a2, run_a1])
        self.assertEqual(layout.run_folders_by_sample[sample_b], [run_b2, run_b1])

    def test_mixed_two_and_three_level_layout_is_rejected(self):
        self.make_dirs("sample-with-runs/run", "sample-without-runs")
        layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "mixed")

    def test_empty_and_unrelated_grandchildren_count_as_runs(self):
        self.make_dirs("sample-a/empty-run", "sample-a/unrelated", "sample-b/empty-run")
        layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "three_level")
        self.assertEqual(len(layout.run_folders_by_sample[str(self.root / "sample-a")]), 2)
        self.assertEqual(len(layout.run_folders_by_sample[str(self.root / "sample-b")]), 1)

    def test_three_level_sample_with_one_run_is_classified_then_later_validated(self):
        self.make_dirs("sample-a/run-a", "sample-b/run-b1", "sample-b/run-b2")
        layout = classify_standalone_summation(str(self.root))
        self.assertEqual(layout.kind, "three_level")
        self.assertEqual(len(layout.run_folders_by_sample[str(self.root / "sample-a")]), 1)

        source = (REPOSITORY_ROOT / "NEAT/ui/mixins/preprocessing.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        summation_source = ast.unparse(methods["add_summation_images"])
        batch_init_source = ast.unparse(methods["_init_summation_3level"])
        self.assertLess(
            summation_source.index("classify_standalone_summation"),
            summation_source.index("layout.kind == 'too_few'"),
        )
        self.assertLess(batch_init_source.index("output_folder"), batch_init_source.index("bad_samples"))

    def test_full_process_uses_only_immediate_children_and_keeps_empty_result(self):
        self.assertEqual(immediate_child_directories(str(self.root)), [])
        child, = self.make_dirs("one-child/grandchild")
        self.assertEqual(immediate_child_directories(str(self.root)), [str(self.root / "one-child")])
        self.assertEqual(len(immediate_child_directories(str(self.root))), 1)
        self.assertEqual(immediate_child_directories(str(self.root / "one-child")), [child])
        self.make_dirs("second-empty-child")
        children = immediate_child_directories(str(self.root))
        self.assertEqual(len(children), 2)
        worker_source = (REPOSITORY_ROOT / "NEAT/workers/preprocessing.py").read_text(
            encoding="utf-8"
        )
        worker_tree = ast.parse(worker_source)
        methods = {
            node.name: node
            for node in ast.walk(worker_tree)
            if isinstance(node, ast.FunctionDef)
        }
        maybe_sum = ast.unparse(methods["maybe_do_summation"])
        self.assertIn("immediate_child_directories", maybe_sum)
        self.assertIn("if not subfolders", maybe_sum)
        self.assertIn("for sf in subfolders", maybe_sum)
        self.assertIn("SummationWorker", maybe_sum)

    def test_summation_and_full_process_use_separate_layout_entry_points(self):
        self.assertEqual(classify_standalone_summation(str(self.root)).kind, "too_few")
        # Full Process's low-level discovery returns an empty list, while
        # classic batch discovery falls back to the selected folder.
        self.assertEqual(immediate_child_directories(str(self.root)), [])
        self.assertEqual(discover_classic_batch(str(self.root)).folders, [str(self.root)])

    def test_helper_import_is_headless_and_needs_no_qapplication(self):
        code = (
            "import sys; import NEAT.services.preprocessing_layout; "
            "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
            "for name in sys.modules); "
            "assert not any(name.startswith('NEAT.ui') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_helper_source_has_no_gui_or_content_classification_dependencies(self):
        source = (REPOSITORY_ROOT / "NEAT/services/preprocessing_layout.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        source_symbols = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        forbidden_imports = {"PyQt5", "NEAT.ui", "NEAT.workers"}
        self.assertTrue(imported.isdisjoint(forbidden_imports))
        self.assertTrue(source_symbols.isdisjoint({"PreprocessingMixin", "FitsViewer"}))
        self.assertNotIn("get_raden_tiff_stack_info", source)
        self.assertNotIn("fits", source.lower())
        self.assertNotIn("tiff", source.lower())


if __name__ == "__main__":
    unittest.main()
