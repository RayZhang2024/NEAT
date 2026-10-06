# NEAT release checklist

1. Create an issue and implement the scoped change on a branch. Open a PR and
   let CI complete. Normal CI tests Windows Python 3.10–3.13 and separately
   builds and clean-installs both wheel and sdist artifacts on Windows Python
   3.13; the Ubuntu lane is developer-quality feedback only.
2. Merge the reviewed PR into `main`. Confirm the changelog and the single
   package version in `pyproject.toml` are ready for release.
3. Tag the already-merged commit as `vX.Y.Z`, where `X.Y.Z` exactly matches the
   installed NEAT package metadata, then push the tag.
4. The release workflow verifies both invariants before building: the tagged
   commit is contained in `main`, and the tag version matches package metadata.
   It then installs dependencies, runs tests and wheel/sdist license checks,
   prepares shared access, builds one PyInstaller folder, and runs the direct
   packaged smoke test before producing either end-user wrapper. From that
   exact validated `dist/NEAT/` payload, it creates and verifies the portable
   ZIP and builds a Briefcase external-app MSI, then validates install, launch,
   manifest correspondence and uninstall before publishing both.
5. The release assets are named `NEAT-vX.Y.Z-portable.zip` and
   `NEAT-vX.Y.Z.msi`. The portable archive contains a top-level `NEAT/`
   directory. The MSI is a per-user install with a Start Menu entry and normal
   uninstall registration.
6. Download each published asset and perform a final launch check. Test the
   exact PyInstaller candidate at `dist/NEAT/NEAT.exe`, not the source checkout.

## Manual packaged acceptance

- Launch the packaged `NEAT.exe`, open representative FITS/TIFF data, and run
  a short preprocessing → fitting → post-processing workflow.
- Verify saved outputs and image orientation, then open and exercise the AI
  Assistant. Confirm core NEAT remains usable without network access.
- Preferably test both release assets on a clean Windows user or machine.
  Install and uninstall the MSI, and extract and run the portable ZIP. Neither
  path should need Python; the MSI should not require administrator elevation.
  Uninstall must leave the user's settings, cache and data intact.

## Recovery before publication

If the workflow fails before publishing, fix the issue on a new PR, merge it
to `main`, and create a new tag for the corrected commit. Do not retag or
publish a commit that is not in `main`; a mismatched tag or package version
must be corrected before retrying.

## Packaging note

The portable ZIP and MSI are two wrappers around one smoke-tested
`dist/NEAT/` PyInstaller payload. A canonical path/size/SHA-256 manifest is
created before wrapping and checked against the ZIP and installed application
files. Briefcase is pinned to `0.4.5` for external-app MSI packaging; it does
not freeze NEAT or install a second NEAT Python environment. Its stable app
identity is `io.github.rayzhang2024.neat` (Briefcase config stores the prefix
`io.github.rayzhang2024` and app name `neat`). MSI and asset versions derive
from `[project].version`; Briefcase maps a PEP 440 version to its MSI numeric
triple, while the release asset uses the `vX.Y.Z` tag. Do not change the
identity between releases. The Windows build lanes pin PyInstaller `6.22.2`
and ONNX Runtime `1.29.0` to the locally smoke-tested Python 3.13 build set;
these are build-environment pins, not NEAT's source-install version bounds.

The MSI can be built from an already validated payload with:

```powershell
python -m tools.clean_briefcase_state
python -m briefcase package windows -p msi --no-input --adhoc-sign
```

The portable archive and MSI lifecycle are validated by the Windows
distribution workflow. Its MSI smoke silently installs to a temporary
per-user location, compares files to the payload manifest, launches
`NEAT.exe --release-smoke-test`, then silently uninstalls and checks that the
Start Menu shortcut and uninstall registration are gone while user settings
and cache sentinels remain. The installed-file comparison permits only the two
Briefcase-generated `_installer/run_post_install.bat` and
`_installer/run_pre_uninstall.bat` hooks beyond the manifest; other extra files
fail validation. On failure, retain the verbose install/uninstall logs as
workflow artifacts. For manual acceptance, use the generated MSI on a
clean Windows account and verify the same install, launch and uninstall steps;
do not substitute a machine-wide `Program Files` install.

Code signing is not required. If approved signing infrastructure is added
later, insert signing after artifact construction and validation but before
workflow artifact upload and GitHub Release publication. Do not alter the MSI's
embedded version or identity merely to rename a release asset.

The wheel and sdist smoke tests run in independent environments outside the
checkout. They validate the installed package-owned assistant knowledge and
launch splash; they do not replace the separate PyInstaller release smoke.

The local Python 3.13 package may be unable to initialise packaged ONNX
Runtime. The release smoke test verifies the supported BM25 retrieval fallback;
this does not claim that semantic ONNX retrieval is available in the package.
