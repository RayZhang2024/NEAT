"""Contract tests for the single-payload Windows distributions."""

from __future__ import annotations

import hashlib
import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from tools.clean_briefcase_state import clean_briefcase_state
from tools.windows_distribution import (
    DistributionError,
    create_manifest,
    create_portable_zip,
    verify_directory,
    verify_installed_directory,
    verify_briefcase_output,
    verify_portable_zip,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WindowsDistributionTests(unittest.TestCase):
    def make_payload(self, root: Path) -> Path:
        payload = root / "NEAT"
        payload.mkdir()
        (payload / "NEAT.exe").write_bytes(b"frozen executable")
        (payload / "z-resource.dat").write_bytes(b"resource")
        (payload / "nested").mkdir()
        (payload / "nested" / "a-data.bin").write_bytes(b"nested")
        return payload

    def run_user_data_safety_script(self, script: str, local_app_data: Path) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is required for the user-data safety behavior test")
        environment = os.environ.copy()
        environment["ISSUE18_SAFETY_MODULE"] = str(
            PROJECT_ROOT / "tools/windows_user_data_safety.psm1"
        )
        environment["ISSUE18_TEST_LOCALAPPDATA"] = str(local_app_data)
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"PowerShell failed. stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def run_msi_context_script(self, script: str, *, expected_success: bool = True) -> subprocess.CompletedProcess[str]:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is required for the Windows Installer context tests")
        environment = os.environ.copy()
        environment["ISSUE18_CONTEXT_MODULE"] = str(PROJECT_ROOT / "tools/windows_msi_context.psm1")
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if expected_success:
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"PowerShell failed. stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
        else:
            self.assertNotEqual(completed.returncode, 0, msg=completed.stdout)
        return completed

    def test_user_data_test_scope_restores_preexisting_tree_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_app_data = Path(directory)
            neat_data = local_app_data / "NEAT"
            (neat_data / "assistant_cache" / "chroma").mkdir(parents=True)
            (neat_data / "empty-directory").mkdir()
            (neat_data / "assistant_settings.json").write_text(
                '{"keep":"existing settings"}', encoding="utf-8"
            )
            (neat_data / "assistant_cache" / "chroma" / "index.bin").write_bytes(
                b"existing cache data"
            )
            expected_files = {
                path.relative_to(neat_data).as_posix(): path.read_bytes()
                for path in neat_data.rglob("*")
                if path.is_file()
            }
            expected_directories = {
                path.relative_to(neat_data).as_posix()
                for path in neat_data.rglob("*")
                if path.is_dir()
            }
            script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_SAFETY_MODULE -Force
$scope = New-NeatUserDataTestScope -LocalAppDataRoot $env:ISSUE18_TEST_LOCALAPPDATA
try {
    if (Test-Path (Join-Path $scope.DataRoot "assistant_settings.json")) {
        throw "Pre-existing settings were not isolated from the test tree."
    }
    if (-not (Test-Path -LiteralPath $scope.BackupPath -PathType Container)) {
        throw "Pre-existing NEAT data was not backed up."
    }
    [System.IO.File]::WriteAllText((Join-Path $scope.DataRoot "test-sentinel.txt"), "test-owned")
} finally {
    Restore-NeatUserDataTestScope -Scope $scope
}
'''
            self.run_user_data_safety_script(script, local_app_data)

            restored_files = {
                path.relative_to(neat_data).as_posix(): path.read_bytes()
                for path in neat_data.rglob("*")
                if path.is_file()
            }
            restored_directories = {
                path.relative_to(neat_data).as_posix()
                for path in neat_data.rglob("*")
                if path.is_dir()
            }
            self.assertEqual(restored_files, expected_files)
            self.assertEqual(restored_directories, expected_directories)
            self.assertEqual(list(local_app_data.glob("NEAT.issue18-backup-*")), [])

    def test_user_data_test_scope_removes_only_its_tree_when_no_prior_data_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_app_data = Path(directory)
            script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_SAFETY_MODULE -Force
$scope = New-NeatUserDataTestScope -LocalAppDataRoot $env:ISSUE18_TEST_LOCALAPPDATA
try {
    [System.IO.File]::WriteAllText((Join-Path $scope.DataRoot "test-sentinel.txt"), "test-owned")
} finally {
    Restore-NeatUserDataTestScope -Scope $scope
}
'''
            self.run_user_data_safety_script(script, local_app_data)
            self.assertFalse((local_app_data / "NEAT").exists())
            self.assertEqual(list(local_app_data.glob("NEAT.issue18-backup-*")), [])

    def test_msi_smoke_keeps_real_user_data_sentinels_through_uninstall(self) -> None:
        smoke = (PROJECT_ROOT / "tools/windows_msi_smoke.ps1").read_text(encoding="utf-8")
        self.assertIn(
            "New-NeatUserDataTestScope -LocalAppDataRoot $env:LOCALAPPDATA", smoke
        )
        self.assertNotIn("$env:LOCALAPPDATA =", smoke)
        self.assertIn('Join-Path $neatUserData "assistant_settings.json"', smoke)
        self.assertIn(
            'Join-Path $neatUserData "assistant_cache\\chroma\\preserve-sentinel.txt"',
            smoke,
        )
        uninstall = smoke.index('"MSI uninstall"')
        self.assertGreater(smoke.index("$settingsHash ="), smoke.index("try {"))
        self.assertGreater(smoke.index("$cacheHash ="), smoke.index("try {"))
        self.assertGreater(smoke.index("per-user assistant settings sentinel"), uninstall)
        self.assertGreater(smoke.index("per-user assistant cache sentinel"), uninstall)
        self.assertIn("Restore-NeatUserDataTestScope -Scope $userDataScope", smoke)
        self.assertIn('"ALLUSERS=2"', smoke)
        self.assertIn('"MSIINSTALLPERUSER=1"', smoke)
        self.assertIn("Resolve-NeatMsiProductContext", smoke)
        self.assertIn("Get-NeatMsiProductInstances -ProductCode $identity.ProductCode", smoke)
        self.assertIn("Requested MSI properties: ALLUSERS=2; MSIINSTALLPERUSER=1", smoke)
        self.assertIn("Requested MSI uninstall properties: ALLUSERS=2; MSIINSTALLPERUSER=1", smoke)
        self.assertIn("Real per-user NEAT data root under preservation test:", smoke)
        self.assertIn("Restored original per-user NEAT data state", smoke)
        self.assertIn("ARP registrations (diagnostic only)", smoke)
        self.assertNotIn("Hive -ne \"HKCU\"", smoke)
        self.assertNotIn("$registryEntries.Count -ne 1", smoke)
        self.assertIn("$env:ProgramData", smoke)
        self.assertIn("$allUsersShortcuts", smoke)
        self.assertIn("Post-uninstall MSI ProductCode context: not registered", smoke)
        self.assertIn("ProductCode", (PROJECT_ROOT / "tools/windows_msi_context.psm1").read_text(encoding="utf-8"))
        self.assertIn("MsiEnumProductsEx", (PROJECT_ROOT / "tools/windows_msi_context.psm1").read_text(encoding="utf-8"))
        context_module = (PROJECT_ROOT / "tools/windows_msi_context.psm1").read_text(encoding="utf-8")
        self.assertIn('OpenView("SELECT `Value` FROM `Property`', context_module)
        self.assertIn('foreach ($property in @("ProductCode", "UpgradeCode"))', context_module)

    def test_start_menu_checks_require_current_user_and_reject_equivalent_all_users(self) -> None:
        script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_CONTEXT_MODULE -Force
$exe = 'C:\Program Files\NEAT\NEAT.exe'
$user = @([pscustomobject]@{ Target = $exe; Name = 'NEAT'; Path = 'C:\Users\test\Programs\NEAT.lnk' })
if (-not (Test-NeatInstalledUserShortcut -Shortcuts $user -InstalledExe $exe)) { throw 'Expected per-user link was not accepted.' }
$wrongUser = @([pscustomobject]@{ Target = 'C:\Other\NEAT.exe'; Name = 'NEAT'; Path = 'C:\Users\test\Programs\NEAT.lnk' })
if (Test-NeatInstalledUserShortcut -Shortcuts $wrongUser -InstalledExe $exe) { throw 'Wrong target was accepted for per-user link.' }
$allUsersSameTarget = [pscustomobject]@{ Target = $exe; Name = 'Application'; Path = 'C:\ProgramData\Programs\Application.lnk' }
if (-not (Test-NeatEquivalentStartMenuShortcut -Shortcut $allUsersSameTarget -InstalledExe $exe)) { throw 'All-users link to installed executable was missed.' }
$allUsersNamedNeat = [pscustomobject]@{ Target = 'C:\Other\app.exe'; Name = 'NEAT'; Path = 'C:\ProgramData\Programs\NEAT.lnk' }
if (-not (Test-NeatEquivalentStartMenuShortcut -Shortcut $allUsersNamedNeat -InstalledExe $exe)) { throw 'Named all-users NEAT link was missed.' }
$unrelated = [pscustomobject]@{ Target = 'C:\Other\app.exe'; Name = 'Another app'; Path = 'C:\ProgramData\Programs\Another app.lnk' }
if (Test-NeatEquivalentStartMenuShortcut -Shortcut $unrelated -InstalledExe $exe) { throw 'Unrelated all-users link was classified as NEAT.' }
'''
        self.run_msi_context_script(script)

    def test_msi_context_accepts_user_unmanaged_product(self) -> None:
        script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_CONTEXT_MODULE -Force
$instance = Resolve-NeatMsiProductContext -ProductCode '{01234567-89AB-CDEF-0123-456789ABCDEF}' -Instances @(
    [pscustomobject]@{ ProductCode = '{01234567-89AB-CDEF-0123-456789ABCDEF}'; Context = 'USERUNMANAGED'; Sid = 'S-1-5-21-test' }
)
if ($instance.Context -ne 'USERUNMANAGED') { throw "Unexpected context: $($instance.Context)" }
'''
        self.run_msi_context_script(script)

    def test_msi_context_accepts_user_managed_product(self) -> None:
        script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_CONTEXT_MODULE -Force
$instance = Resolve-NeatMsiProductContext -ProductCode '01234567-89ab-cdef-0123-456789abcdef' -Instances @(
    [pscustomobject]@{ ProductCode = '{01234567-89AB-CDEF-0123-456789ABCDEF}'; Context = 'USERMANAGED'; Sid = 'S-1-5-21-test' }
)
if ($instance.Context -ne 'USERMANAGED') { throw "Unexpected context: $($instance.Context)" }
'''
        self.run_msi_context_script(script)

    def test_msi_product_discovery_queries_all_contexts_with_windows_installer_api(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows Installer product enumeration is Windows-only")
        script = r'''
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_CONTEXT_MODULE -Force
$instances = @(Get-NeatMsiProductInstances -ProductCode '{01234567-89AB-CDEF-0123-456789ABCDEF}')
if ($instances.Count -ne 0) { throw "Unexpected test ProductCode registration: $($instances.Count)" }
'''
        self.run_msi_context_script(script)

    def test_msi_context_rejects_machine_missing_and_ambiguous_products(self) -> None:
        cases = [
            "@([pscustomobject]@{ ProductCode = '{01234567-89AB-CDEF-0123-456789ABCDEF}'; Context = 'MACHINE'; Sid = '' })",
            "@()",
            "@([pscustomobject]@{ ProductCode = '{01234567-89AB-CDEF-0123-456789ABCDEF}'; Context = 'USERUNMANAGED'; Sid = 'S-1' }, [pscustomobject]@{ ProductCode = '{01234567-89AB-CDEF-0123-456789ABCDEF}'; Context = 'USERMANAGED'; Sid = 'S-1' })",
        ]
        for instances in cases:
            with self.subTest(instances=instances):
                script = f'''\
$ErrorActionPreference = "Stop"
Import-Module $env:ISSUE18_CONTEXT_MODULE -Force
Resolve-NeatMsiProductContext -ProductCode '{{01234567-89AB-CDEF-0123-456789ABCDEF}}' -Instances {instances}
'''
                self.run_msi_context_script(script, expected_success=False)

    def test_manifest_is_sorted_normalized_hashed_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_payload(root)
            first = write_manifest(payload, root / "first.json")
            first_bytes = (root / "first.json").read_bytes()
            second = write_manifest(payload, root / "second.json")

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, (root / "second.json").read_bytes())
            self.assertEqual(
                [entry["path"] for entry in first["files"]],
                ["NEAT.exe", "nested/a-data.bin", "z-resource.dat"],
            )
            for entry in first["files"]:
                payload_file = payload.joinpath(*entry["path"].split("/"))
                self.assertEqual(entry["size"], payload_file.stat().st_size)
                self.assertEqual(entry["sha256"], hashlib.sha256(payload_file.read_bytes()).hexdigest())

    def test_manifest_requires_expected_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload"
            payload.mkdir()
            (payload / "other.exe").write_bytes(b"not NEAT")
            with self.assertRaisesRegex(DistributionError, "NEAT.exe"):
                create_manifest(payload)

    def test_directory_verification_detects_missing_changed_and_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_payload(root)
            manifest_path = root / "manifest.json"
            write_manifest(payload, manifest_path)
            verify_directory(payload, manifest_path)

            (payload / "z-resource.dat").write_bytes(b"changed")
            with self.assertRaisesRegex(DistributionError, "changed"):
                verify_directory(payload, manifest_path)
            (payload / "z-resource.dat").write_bytes(b"resource")
            (payload / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(DistributionError, "extra"):
                verify_directory(payload, manifest_path)
            (payload / "extra.txt").unlink()
            (payload / "z-resource.dat").unlink()
            with self.assertRaisesRegex(DistributionError, "missing"):
                verify_directory(payload, manifest_path)

    def test_installed_verification_allows_only_briefcase_msi_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_payload(root)
            manifest_path = root / "manifest.json"
            write_manifest(payload, manifest_path)
            installed = root / "installed"
            shutil.copytree(payload, installed)
            hooks = installed / "_installer"
            hooks.mkdir()
            (hooks / "run_post_install.bat").write_text("installer hook", encoding="utf-8")
            (hooks / "run_pre_uninstall.bat").write_text("uninstaller hook", encoding="utf-8")

            verify_installed_directory(installed, manifest_path)
            (hooks / "unexpected.bat").write_text("not a Briefcase hook", encoding="utf-8")
            with self.assertRaisesRegex(DistributionError, "extra"):
                verify_installed_directory(installed, manifest_path)

    def test_portable_zip_contains_exact_payload_under_one_neat_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_payload(root)
            manifest_path = root / "payload.json"
            archive_path = root / "NEAT-v1.2.3-portable.zip"
            write_manifest(payload, manifest_path)
            create_portable_zip(payload, manifest_path, archive_path)
            verify_portable_zip(archive_path, manifest_path)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["NEAT/", "NEAT/NEAT.exe", "NEAT/nested/a-data.bin", "NEAT/z-resource.dat"],
                )

    def test_portable_zip_verification_rejects_extra_or_modified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_payload(root)
            manifest_path = root / "payload.json"
            archive_path = root / "bad.zip"
            write_manifest(payload, manifest_path)
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in sorted(payload.rglob("*")):
                    if path.is_file():
                        archive.write(path, f"NEAT/{path.relative_to(payload).as_posix()}")
                archive.writestr("NEAT/extra.txt", b"extra")
            with self.assertRaisesRegex(DistributionError, "extra"):
                verify_portable_zip(archive_path, manifest_path)

    def test_briefcase_metadata_is_external_per_user_and_uses_project_version(self) -> None:
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        briefcase = project["tool"]["briefcase"]
        app = briefcase["app"]["neat"]
        windows = app["windows"]
        self.assertEqual(project["project"]["name"], "NEAT")
        self.assertEqual(
            f"{briefcase['bundle']}.{next(iter(briefcase['app']))}",
            "io.github.rayzhang2024.neat",
        )
        self.assertEqual(app["formal_name"], "NEAT")
        self.assertEqual(windows["external_package_path"], "dist/NEAT")
        self.assertEqual(windows["external_package_executable_path"], "NEAT.exe")
        self.assertIs(windows["system_installer"], False)
        self.assertIs(windows["use_full_install_path"], False)
        self.assertIs(windows["install_launcher"], True)
        self.assertNotIn("version", briefcase)
        self.assertNotIn("version", app)
        self.assertNotIn("requires", app)
        self.assertNotIn("sources", app)

    def test_project_license_uses_pep639_and_declares_repository_license(self) -> None:
        configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = configuration["project"]
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertTrue((PROJECT_ROOT / "LICENSE").is_file())
        self.assertIn("setuptools>=77.0.3", configuration["build-system"]["requires"])

    def test_briefcase_output_is_external_and_uses_expected_identity_version_and_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\nname="NEAT"\nversion="4.8.2"\n'
                '[tool.briefcase]\nproject_name="NEAT"\nbundle="io.github.rayzhang2024"\n'
                '[tool.briefcase.app.neat]\nformal_name="NEAT"\n'
                '[tool.briefcase.app.neat.windows]\nexternal_package_path="dist/NEAT"\n'
                'external_package_executable_path="NEAT.exe"\ninstall_launcher=true\n',
                encoding="utf-8",
            )
            bundle_id = "io.github.rayzhang2024.neat"
            upgrade_code = uuid.uuid5(
                uuid.NAMESPACE_DNS, ".".join(bundle_id.split(".")[::-1])
            )
            generated = root / "build" / "NEAT" / "windows" / "app"
            generated.mkdir(parents=True)
            (generated / "LICENSE.rtf").write_text("{\\rtf1 license}", encoding="utf-8")
            source = (root / "dist" / "NEAT" / "**").resolve()
            source.parent.mkdir(parents=True)
            (root / "dist" / "NEAT" / "NEAT.exe").write_bytes(b"payload")
            (root / "dist" / "NEAT-4.8.2.msi").write_bytes(b"installer")
            (generated / "neat.wxs").write_text(
                f'<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">'
                f'<Package Name="NEAT" Version="4.8.2" UpgradeCode="{upgrade_code}" '
                f'Scope="perUserOrMachine"><Shortcut Target="[INSTALLFOLDER]NEAT.exe" />'
                f'<Files Include="{source}" /></Package></Wix>',
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                verify_briefcase_output(root)
            (generated / "src").mkdir()
            with self.assertRaisesRegex(DistributionError, "unexpected runtime"):
                verify_briefcase_output(root)

    def test_clean_briefcase_state_removes_only_versioned_briefcase_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\nname="NEAT"\nversion="4.8.2"\n', encoding="utf-8"
            )
            payload = root / "dist" / "NEAT" / "NEAT.exe"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"payload must remain")
            current_msi = root / "dist" / "NEAT-4.8.2.msi"
            tagged_msi = root / "dist" / "NEAT-v4.8.2.msi"
            other_msi = root / "dist" / "NEAT-4.8.1.msi"
            for path in (current_msi, tagged_msi, other_msi):
                path.write_bytes(b"generated")
            briefcase_state = root / "build" / "NEAT" / "windows" / "app"
            briefcase_state.mkdir(parents=True)
            (briefcase_state / "stale.wxs").write_text("stale", encoding="utf-8")
            pyinstaller_state = root / "build" / "NEAT" / "analysis.toc"
            pyinstaller_state.write_text("pyinstaller state", encoding="utf-8")

            clean_briefcase_state(root)

            self.assertTrue(payload.is_file())
            self.assertTrue(other_msi.is_file())
            self.assertTrue(pyinstaller_state.is_file())
            self.assertFalse(current_msi.exists())
            self.assertFalse(tagged_msi.exists())
            self.assertFalse(briefcase_state.exists())

    def test_release_workflow_builds_payload_once_then_wraps_and_publishes_both(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count("pyinstaller --noconfirm --clean NEAT.spec"), 1)
        self.assertEqual(workflow.count("--release-smoke-test"), 1)
        self.assertLess(workflow.index("Build standalone app"), workflow.index("Smoke test packaged"))
        self.assertLess(workflow.index("Smoke test packaged"), workflow.index("Create canonical payload manifest"))
        self.assertLess(workflow.index("Create canonical payload manifest"), workflow.index("Package portable ZIP"))
        self.assertLess(workflow.index("Package portable ZIP"), workflow.index("Build per-user MSI"))
        self.assertIn("New-Item -ItemType Directory -Path $env:BRIEFCASE_HOME -Force", workflow)
        self.assertIn('"RELEASE_PORTABLE=NEAT-$tag-portable.zip"', workflow)
        self.assertIn('"RELEASE_MSI=NEAT-$tag.msi"', workflow)
        self.assertIn("tools.check_release_version", workflow)
        self.assertIn("git merge-base --is-ancestor", workflow)
        self.assertIn("tools.prepare_public_shared_access", workflow)
        self.assertIn("NEAT_SHARED_PUBLIC_ACCESS_TOKEN", workflow)
        self.assertIn("briefcase==0.4.5", workflow)
        self.assertIn("onnxruntime==1.29.0", workflow)
        self.assertIn("pyinstaller==6.22.2", workflow)
        self.assertIn("Validate silent MSI lifecycle", workflow)

    def test_dedicated_distribution_workflow_is_python313_and_non_publishing(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/distribution.yml").read_text(encoding="utf-8")
        self.assertRegex(workflow, r"(?m)^  pull_request:")
        self.assertIn('      - "pyproject.toml"', workflow)
        self.assertIn('      - "docs/RELEASE_CHECKLIST.md"', workflow)
        self.assertIn('python-version: "3.13"', workflow)
        self.assertIn("New-Item -ItemType Directory -Path $env:BRIEFCASE_HOME -Force", workflow)
        self.assertIn("briefcase==0.4.5", workflow)
        self.assertIn("onnxruntime==1.29.0", workflow)
        self.assertIn("pyinstaller==6.22.2", workflow)
        self.assertIn("--no-input", workflow)
        self.assertIn("--release-smoke-test", workflow)
        self.assertIn("windows_msi_smoke.ps1", workflow)
        self.assertIn("tools/windows_user_data_safety.psm1", workflow)
        self.assertIn('-MsiPath "dist\\$env:DISTRIBUTION_MSI"', workflow)
        self.assertIn("issue18-dummy-only", workflow)
        self.assertIn("https://ci.invalid", workflow)
        self.assertIn("NEAT-v${{ env.RELEASE_VERSION }}-portable.zip", workflow)
        self.assertIn('$filename = "NEAT-v$version.msi"', workflow)
        self.assertIn("dist/${{ env.DISTRIBUTION_MSI }}", workflow)
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        self.assertIn("python -m ruff check NEAT tests tools --select E9,F63,F7,F82", workflow)
        self.assertIn("python -m mypy NEAT/core NEAT/domain NEAT/services", workflow)
        self.assertIn("python -m pip check", workflow)
        self.assertIn("python -m build --wheel --sdist", workflow)
        self.assertNotIn("action-gh-release", workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertIn("NEAT_EXPECTED_RELEASE_VERSION = $env:RELEASE_VERSION", workflow)
        self.assertIn("Upload validated distribution artifacts", workflow)
        msi_smoke = (PROJECT_ROOT / "tools/windows_msi_smoke.ps1").read_text(encoding="utf-8")
        for marker in (
            '"/i"',
            '"/x"',
            '"/qn"',
            '"/norestart"',
            '"/l*v"',
            '"ALLUSERS=2"',
            '"MSIINSTALLPERUSER=1"',
            'Hive = $_.PSDrive.Name',
            "ExitCode -ne 0",
        ):
            self.assertIn(marker, msi_smoke)
        self.assertIn("assistant_settings.json", msi_smoke)
        self.assertIn("assistant_cache", msi_smoke)
        self.assertIn("verify-installed", msi_smoke)


if __name__ == "__main__":
    unittest.main()
