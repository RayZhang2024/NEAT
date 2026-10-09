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

## Pre-release candidate validation

A pull request that updates `pyproject.toml` or this checklist triggers the
Windows Distribution workflow. It runs the unit suite, Ruff fatal-error gate,
targeted mypy checks and `pip check`; builds and validates the wheel and sdist;
builds the PyInstaller application; runs its packaged smoke test; builds the
portable ZIP and MSI from the same payload; validates the MSI install, launch
and uninstall; and uploads the candidate files as workflow artifacts. This
workflow does not publish a GitHub Release.

The distribution workflow uses dummy shared-access values at `ci.invalid`.
Its Assistant checks validate packaging, imports and bundled knowledge; they do
not validate connectivity to the production shared service.

## Manual candidate acceptance

Test both workflow artifacts on Windows, preferably on a clean user account.
Record the package filename, machine/Python state, test data, actions taken,
observed result and any error for each step:

1. Extract `NEAT-v4.8.3-portable.zip` and launch `NEAT/NEAT.exe`.
2. Install `NEAT-v4.8.3.msi` without elevation, launch from the Start Menu,
   then uninstall it.
3. Load representative experimental FITS and TIFF data. Include the original
   2,925-frame dataset when available; this real-data test has not yet been
   performed on the release candidates.
4. Run standalone Overlap Correction and inspect frame selection and output.
5. Run Full Process with numeric-only frame selection; confirm auxiliary
   `_SummedImg.fits` files are excluded and current-run sidecars are used.
6. Check Normalisation output files and metadata for correctness.
7. Run fitting and visualisation and review saved outputs.
8. Start the AI Assistant and check that its interface and bundled knowledge
   load. Production shared-service connectivity requires a separate check.
9. Confirm both packages launch on a machine without Python installed.
10. Confirm MSI uninstall preserves user settings, cache and experimental
    data, and removes the application and Start Menu entry.

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
