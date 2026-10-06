"""Tests for the immutable, headless preprocessing result contract."""

import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from NEAT.domain import (
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PreprocessingStatusTests(unittest.TestCase):
    def test_all_final_statuses_are_representable(self):
        success = PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, 0)
        failure = PreprocessingOperationResult(
            PreprocessingStatus.FAILED, 0, errors=("failed",)
        )
        cancelled = PreprocessingOperationResult(PreprocessingStatus.CANCELLED, 0)
        self.assertEqual(
            {success.status, failure.status, cancelled.status},
            {
                PreprocessingStatus.SUCCEEDED,
                PreprocessingStatus.FAILED,
                PreprocessingStatus.CANCELLED,
            },
        )
        self.assertEqual(
            tuple(PreprocessingStatus),
            (
                PreprocessingStatus.SUCCEEDED,
                PreprocessingStatus.FAILED,
                PreprocessingStatus.CANCELLED,
            ),
        )

    def test_arbitrary_status_values_are_rejected(self):
        for status in ("running", "succeeded", None, 1):
            with self.subTest(status=status), self.assertRaises(TypeError):
                PreprocessingOperationResult(status, 0)


class ProducedOutputTests(unittest.TestCase):
    def test_path_and_optional_role_are_preserved(self):
        self.assertEqual(ProducedOutput("result/file.fits"), ProducedOutput("result/file.fits", None))
        self.assertEqual(ProducedOutput("report.csv", "report").role, "report")

    def test_empty_path_and_explicitly_empty_role_are_rejected(self):
        with self.assertRaises(ValueError):
            ProducedOutput("")
        with self.assertRaises(ValueError):
            ProducedOutput("output.fits", "")

    def test_non_string_path_or_role_are_rejected(self):
        with self.assertRaises(TypeError):
            ProducedOutput(123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ProducedOutput("output.fits", 1)  # type: ignore[arg-type]

    def test_nonexistent_paths_are_valid_without_filesystem_validation(self):
        output = ProducedOutput(str(REPOSITORY_ROOT / "does-not-exist" / "result.fits"))
        self.assertIn("does-not-exist", output.path)

    def test_construction_does_not_query_the_filesystem(self):
        with patch("os.path.exists", side_effect=AssertionError("filesystem queried")):
            output = ProducedOutput("not-checked/output.fits")
        self.assertEqual(output.path, "not-checked/output.fits")

    def test_output_descriptor_is_immutable(self):
        output = ProducedOutput("output.fits", "frame")
        with self.assertRaises(FrozenInstanceError):
            output.path = "replacement.fits"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            output.role = "report"  # type: ignore[misc]


class PreprocessingOperationResultTests(unittest.TestCase):
    def test_success_with_outputs_and_warnings(self):
        result = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            processed_count=2,
            outputs=[ProducedOutput("one.fits"), ProducedOutput("two.fits")],
            expected_count=2,
            warnings=["used fallback"],
        )
        self.assertEqual(result.outputs, (ProducedOutput("one.fits"), ProducedOutput("two.fits")))
        self.assertEqual(result.warnings, ("used fallback",))
        self.assertEqual(result.errors, ())

    def test_success_with_zero_outputs_is_valid(self):
        result = PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, 0)
        self.assertEqual(result.outputs, ())

    def test_failure_before_output_and_after_partial_output(self):
        before_output = PreprocessingOperationResult(
            PreprocessingStatus.FAILED,
            processed_count=0,
            errors=["input could not be read"],
        )
        partial = PreprocessingOperationResult(
            PreprocessingStatus.FAILED,
            processed_count=3,
            expected_count=4,
            outputs=[ProducedOutput("frame-1.fits"), ProducedOutput("frame-2.fits")],
            errors=["frame 4 failed"],
        )
        self.assertEqual(before_output.outputs, ())
        self.assertEqual(partial.processed_count, 3)
        self.assertEqual(len(partial.outputs), 2)
        self.assertEqual(partial.status, PreprocessingStatus.FAILED)

    def test_cancellation_before_and_after_outputs(self):
        before_output = PreprocessingOperationResult(PreprocessingStatus.CANCELLED, 0)
        partial = PreprocessingOperationResult(
            PreprocessingStatus.CANCELLED,
            processed_count=1,
            expected_count=3,
            outputs=[ProducedOutput("partial.fits")],
        )
        self.assertEqual(before_output.outputs, ())
        self.assertEqual(partial.outputs, (ProducedOutput("partial.fits"),))

    def test_cancelled_result_can_retain_errors(self):
        result = PreprocessingOperationResult(
            PreprocessingStatus.CANCELLED,
            1,
            errors=["earlier frame failed"],
        )
        self.assertEqual(result.errors, ("earlier frame failed",))

    def test_result_and_ordered_collections_are_immutable(self):
        result = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            1,
            outputs=[ProducedOutput("output.fits")],
            warnings=["warning"],
        )
        with self.assertRaises(FrozenInstanceError):
            result.status = PreprocessingStatus.CANCELLED  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.processed_count = 99  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            result.outputs.append(ProducedOutput("other.fits"))  # type: ignore[attr-defined]
        with self.assertRaises(TypeError):
            result.warnings[0] = "changed"  # type: ignore[index]

    def test_mutating_caller_collections_does_not_change_result(self):
        outputs = [ProducedOutput("a.fits")]
        errors: list[str] = []
        warnings = ["first"]
        result = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            0,
            outputs=outputs,
            errors=errors,
            warnings=warnings,
        )
        outputs.append(ProducedOutput("b.fits"))
        errors.append("later error")
        warnings[0] = "changed"
        warnings.append("second")
        self.assertEqual(result.outputs, (ProducedOutput("a.fits"),))
        self.assertEqual(result.errors, ())
        self.assertEqual(result.warnings, ("first",))

    def test_output_error_warning_order_is_preserved(self):
        result = PreprocessingOperationResult(
            PreprocessingStatus.FAILED,
            2,
            outputs=[ProducedOutput("second"), ProducedOutput("first")],
            errors=["error 2", "error 1"],
            warnings=["warning 2", "warning 1"],
        )
        self.assertEqual([output.path for output in result.outputs], ["second", "first"])
        self.assertEqual(result.errors, ("error 2", "error 1"))
        self.assertEqual(result.warnings, ("warning 2", "warning 1"))

    def test_status_error_consistency(self):
        with self.assertRaises(ValueError):
            PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, 0, errors=["error"])
        with self.assertRaises(ValueError):
            PreprocessingOperationResult(PreprocessingStatus.FAILED, 0)
        self.assertEqual(
            PreprocessingOperationResult(PreprocessingStatus.CANCELLED, 0).errors,
            (),
        )
        self.assertEqual(
            PreprocessingOperationResult(
                PreprocessingStatus.CANCELLED, 0, errors=["before cancellation"]
            ).errors,
            ("before cancellation",),
        )

    def test_zero_unknown_equal_and_independent_counts(self):
        zero = PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, 0)
        unknown = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED, 7, expected_count=None
        )
        equal = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED, 4, expected_count=4
        )
        fewer_outputs = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            3,
            expected_count=4,
            outputs=[ProducedOutput("only-one-artifact")],
        )
        self.assertEqual(zero.processed_count, 0)
        self.assertIsNone(unknown.expected_count)
        self.assertEqual(equal.processed_count, equal.expected_count)
        self.assertEqual(fewer_outputs.processed_count, 3)
        self.assertEqual(len(fewer_outputs.outputs), 1)

    def test_negative_counts_and_processed_greater_than_expected_are_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, -1)
        with self.assertRaises(ValueError):
            PreprocessingOperationResult(
                PreprocessingStatus.SUCCEEDED, 0, expected_count=-1
            )
        with self.assertRaises(ValueError):
            PreprocessingOperationResult(
                PreprocessingStatus.SUCCEEDED, 2, expected_count=1
            )

    def test_counts_reject_booleans_floats_strings_and_other_non_integers(self):
        invalid_values = (True, False, 1.0, "1", object())
        for value in invalid_values:
            with self.subTest(field="processed_count", value=value), self.assertRaises(TypeError):
                PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, value)  # type: ignore[arg-type]
            with self.subTest(field="expected_count", value=value), self.assertRaises(TypeError):
                PreprocessingOperationResult(
                    PreprocessingStatus.SUCCEEDED, 0, expected_count=value  # type: ignore[arg-type]
                )
        with self.assertRaises(TypeError):
            PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, None)  # type: ignore[arg-type]


class PreprocessingDomainImportTests(unittest.TestCase):
    def test_contract_module_has_an_isolated_standard_library_boundary(self):
        source_path = REPOSITORY_ROOT / "NEAT" / "domain" / "preprocessing.py"
        code = f'''\
import ast
import __future__
import dataclasses
import enum
import sys
import types
import typing
from pathlib import Path

source_path = Path({str(source_path)!r})
source = source_path.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(source_path))
allowed_imports = {{"__future__", "dataclasses", "enum", "typing"}}
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        imports = [node.module or ""]
    else:
        imports = []
    assert all(name in allowed_imports for name in imports), imports
    if isinstance(node, ast.Name):
        assert node.id not in {{"FitsViewer", "QApplication"}}, node.id
    if isinstance(node, ast.Call):
        called_name = node.func.id if isinstance(node.func, ast.Name) else None
        called_attr = node.func.attr if isinstance(node.func, ast.Attribute) else None
        assert called_name != "open", ast.unparse(node)
        assert called_attr not in {{"open", "exists", "is_file", "is_dir", "stat", "listdir", "scandir"}}, ast.unparse(node)

# Execute the dedicated source in isolation, without running NEAT package
# initializers. Imports it needs are preloaded; an audit hook rejects I/O from
# the module body itself.
module_name = "_neat_preprocessing_contract_isolated"
module = types.ModuleType(module_name)
module.__file__ = str(source_path)
sys.modules[module_name] = module
def reject_io(event, args):
    if event in {{"open", "os.listdir", "os.scandir", "os.stat"}}:
        raise AssertionError(("contract module performed filesystem I/O", event, args))
sys.addaudithook(reject_io)
exec(compile(source, str(source_path), "exec"), module.__dict__)
assert module.PreprocessingStatus.SUCCEEDED.value == "succeeded"
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in ("numpy", "PyQt5", "NEAT.ui", "NEAT.workers")
)
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_types_are_reexported_from_domain_package(self):
        code = (
            "from NEAT.domain import PreprocessingOperationResult, "
            "PreprocessingStatus, ProducedOutput; "
            "assert PreprocessingStatus.SUCCEEDED.value == 'succeeded'; "
            "assert ProducedOutput('out').path == 'out'; "
            "assert PreprocessingOperationResult(PreprocessingStatus.SUCCEEDED, 0).outputs == ()"
        )
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
