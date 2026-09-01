"""Tests for deterministic release-version validation."""

from __future__ import annotations

import unittest

from tools.check_release_version import validate_release_version


class ReleaseVersionValidationTests(unittest.TestCase):
    def test_matching_tag_and_package_version_pass(self) -> None:
        self.assertEqual(validate_release_version("v4.8.2", "4.8.2"), "4.8.2")

    def test_mismatched_tag_and_package_version_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_release_version("v4.8.2", "4.8.1")

    def test_invalid_tag_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "vX.Y.Z"):
            validate_release_version("release-4.8.2", "4.8.2")


if __name__ == "__main__":
    unittest.main()
