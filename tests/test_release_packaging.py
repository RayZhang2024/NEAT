"""Static release guards for assistant-capable CI and PyInstaller builds."""

from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.metadata import version
from pathlib import Path
import tomllib

import NEAT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTests(unittest.TestCase):
    def test_release_version_is_consistent(self) -> None:
        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(NEAT.__version__, version("NEAT"))
        self.assertEqual(project["project"]["version"], NEAT.__version__)
        self.assertIn(
            "## 4.8.1",
            (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        )

    def test_source_release_smoke_test_loads_all_approved_knowledge(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from NEAT.app import _run_release_smoke_test; "
                    "print(_run_release_smoke_test())"
                ),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertGreater(int(completed.stdout.strip()), 0)

    def test_release_workflow_installs_assistant_extras_and_smoke_tests(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(".[assistant,assistant-server]", workflow)
        self.assertIn("--release-smoke-test", workflow)
        self.assertIn("NEAT_RELEASE_SMOKE_RESULT", workflow)
        self.assertIn("tools.prepare_public_shared_access", workflow)
        self.assertIn("NEAT_SHARED_PUBLIC_ACCESS_TOKEN", workflow)
        self.assertIn('python-version: "3.13"', workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("git merge-base --is-ancestor", workflow)
        self.assertIn("does not match release tag", workflow)
        self.assertIn("tools.check_release_version", workflow)

    def test_test_workflow_installs_dependencies_used_by_full_suite(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/tests.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(".[assistant,assistant-server]", workflow)
        self.assertEqual(workflow.count('python-version: "3.13"'), 3)

    def test_spec_bundles_approved_knowledge_and_dynamic_adapters(self) -> None:
        specification = (PROJECT_ROOT / "NEAT.spec").read_text(encoding="utf-8")
        self.assertIn('"docs" / "assistant"', specification)
        self.assertIn('collect_submodules("tools")', specification)
        self.assertIn('"shared_access.json"', specification)
        self.assertIn('"NEAT/config"', specification)
        self.assertIn('"langchain-openai"', specification)
        self.assertIn('"langchain-anthropic"', specification)
        self.assertIn('"langchain-google-genai"', specification)
        self.assertIn('"langchain-chroma"', specification)
        self.assertIn('"keyring"', specification)
        self.assertIn('"NEAT"', specification)
        self.assertNotIn("collect_all(", specification)


if __name__ == "__main__":
    unittest.main()
