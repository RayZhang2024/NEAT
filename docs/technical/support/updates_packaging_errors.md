---
title: Updates, packaging and operational errors
doc_id: neat-tech-support-updates-packaging-errors
doc_type: technical_reference
functional_area: support
audience: [user, developer, support]
neat_version: 4.8.3
verified_commit: 5eb4a4b419f9a301b502502706fa04e0436b8e0a
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [pyproject.toml, NEAT.spec, NEAT/package_resources.py, .github/workflows/tests.yml, .github/workflows/distribution.yml, .github/workflows/release.yml, NEAT/app.py, NEAT/ui/main_window.py, tools/windows_distribution.py, tools/clean_briefcase_state.py, tools/windows_msi_smoke.ps1, tools/windows_msi_context.psm1, tools/assistant_openai.py]
source_symbols: [UpdateCheckWorker, FitsViewer.start_update_check, FitsViewer._on_update_check_finished, describe_openai_error]
test_paths: [tests/test_assistant_openai.py, tests/test_package_resources.py, tests/test_release_packaging.py, tests/test_windows_distribution.py]
---

# Updates, packaging and operational errors

## Supported Python and installation surfaces

The package metadata declares Python `>=3.10,<3.14`; the documented GUI
platform remains Windows. This range applies to Python package/source installs
and developer checkouts. The Windows standalone release is a separate
self-contained PyInstaller artifact and continues to build with Python 3.13.
Ubuntu CI provides development-quality unit, lint and type feedback, not a
claim of supported Linux GUI operation.

Normal CI runs the full unit suite on Windows with Python 3.10, 3.11, 3.12 and
3.13. Its Windows 3.13 artifact job builds both a wheel and an sdist, inspects
their contents, then installs each into a separate clean environment outside
the checkout. The smoke probes verify installed imports, metadata, runtime
resources, headless scientific imports and dependency health. Source-checkout
tests are not treated as proof that either artifact is complete.

## Source installation and entry points

The package version is `4.8.3`. Core installation declares PyQt5, Matplotlib,
NumPy, SciPy, Astropy, pandas, openpyxl, h5py, psutil and Pillow. The assistant
dependencies are optional:

```text
python -m pip install -e ".[assistant]"
```

The console command `neat` and `python -m NEAT.app` both call the same GUI
entry point. A missing core package, for example `h5py`, causes an import-time
`ModuleNotFoundError`; installing the project dependencies into the active
environment is the corrective action.

## Update checking

NEAT queries GitHub's latest-release API in a background thread with a six
second URL timeout. Version comparison removes a leading `v` and compares the
first dotted numeric sequence as an integer tuple.

Startup checks:

- are enabled by default;
- begin 1.5 seconds after window construction;
- are skipped if disabled or successfully attempted in the previous 24 hours;
- do not prompt again for a version the user chose to skip.

A manual check reports network errors in a dialog. A newer release offers Open
Download Page, Skip This Version and Later. NEAT only opens the release page;
it does not download, install or replace application files.

## Windows end-user releases

Tags matching `v*` run the GitHub release workflow on Python 3.13. It installs
the project with the `assistant` and `assistant-server` extras, plus pinned
build-only `pyinstaller==6.22.2`, `onnxruntime==1.29.0`, and
`briefcase==0.4.5`; it runs the unit suite and keeps the wheel/sdist packaging
checks. These build pins do not narrow NEAT's general source-install
compatibility. After shared-access preparation it builds the
one-folder application from `NEAT.spec` exactly once and runs
`dist/NEAT/NEAT.exe --release-smoke-test` before wrapping it.

That exact smoke-tested `dist/NEAT/` directory supplies both release formats:

- `NEAT-vX.Y.Z-portable.zip` contains `NEAT/NEAT.exe` and the complete
  one-folder payload. Extract and run it without installing Python.
- `NEAT-vX.Y.Z.msi` installs the external PyInstaller payload per user with a
  Start Menu entry and Windows uninstall registration. Briefcase does not
  rebuild NEAT, resolve its application dependencies, or install a second
  Python runtime.

Before wrapping, `tools.windows_distribution` writes a deterministic manifest
sorted by normalized payload-relative path with each file's size and SHA-256.
The portable ZIP and MSI-installed application files are compared to this
manifest. The MSI smoke also runs the installed executable's release smoke,
then uninstalls and checks removal of application files, Start Menu shortcut
and uninstall registration while preserving per-user settings/cache data.
Verbose MSI install/uninstall logs are uploaded when lifecycle validation
fails.

The stable Briefcase application identity is
`io.github.rayzhang2024.neat`; Briefcase configuration expresses it as the
`io.github.rayzhang2024` bundle prefix plus app name `neat`. Per-user scope,
the short install path, and Start Menu launcher are explicit settings. The
MSI version comes from `[project].version` in `pyproject.toml`, and Briefcase
derives the numeric MSI version triple from it. Only the finished MSI asset
filename is renamed to match the `vX.Y.Z` release tag.

The dedicated Windows distribution workflow uses Python 3.13 and dummy
shared-access values to build and exercise both formats; it cannot publish a
GitHub Release. Code signing is not required. If approved signing
infrastructure is added later, sign after construction and validation of both
wrappers but before artifact upload and publication.

The spec relies on PyInstaller's standard scientific/GUI hooks rather than
recursively collecting dependency test suites. It includes selected runtime
DLLs, provider metadata, dynamic assistant modules, keyring backends and the
approved `NEAT/knowledge` package resources. Normal wheels and sdists declare
the same knowledge Markdown and launch splash through explicit setuptools
package-data rules. The executable is windowed
(`console=False`) and is collected as a directory, not a single-file
executable.

Installed runtime knowledge is resolved through
`NEAT.package_resources.assistant_knowledge_root()`; repository evaluation
questions and developer guidance remain under `docs/assistant` and are not
runtime dependencies.

## Assistant package verification and retrieval fallback

The release smoke mode is invoked with `--release-smoke-test`. It makes no API
request and records whether the package can load:

- OpenAI, Anthropic, Google and OpenAI-compatible provider adapters;
- the operating-system keyring integration;
- every bundled approved knowledge section; and
- the local BM25 retrieval fallback.

The semantic Chroma/ONNX retriever is optional. Its index is stored in the
user's local application cache rather than beside the executable. If ONNX
Runtime cannot load, its embedding model is unavailable, or the first-use
model download fails, NEAT continues with approved-source BM25 retrieval. The
assistant therefore remains usable, although semantic matching quality may be
lower. The locally built Python 3.13 release candidate exercised this fallback;
the official GitHub release workflow builds and retests on Python 3.13.

The local Chroma embedding model may require a first-use download. Offline
operation does not require that download because BM25 works only from the
bundled approved documents.

## Provider credentials and diagnostics

Personal API keys are stored through the operating-system keyring when entered
in Assistant Settings. Provider-specific environment variables, including
`OPENAI_API_KEY`, are also supported for development and managed deployments.
Keys are not stored in the repository or ordinary GUI settings. Local OpenAI-
compatible servers such as Ollama and LM Studio can be configured without an
API key when the server does not require one.

Safe error mapping distinguishes:

- missing or rejected keys;
- insufficient billing quota versus transient rate limiting;
- model permission or model-not-found errors;
- timeout and connection/firewall errors; and
- an unexpected exception type.

When available, the OpenAI request ID is displayed, but credentials and raw
provider response bodies are not echoed.

## Manual release verification

- Install the MSI on a clean Windows account without elevation, verify its
  Start Menu launch and uninstall entry, run the packaged smoke test, and
  uninstall it. Confirm settings/cache files survive.
- Extract the portable ZIP and launch `NEAT/NEAT.exe` without installing
  Python.
- Ask one grounded question and verify three displayed citations.
- Test first launch without network and with no API key.
- Test authentication, quota, timeout and unavailable-model messages.
- Confirm core analysis remains usable when the assistant cannot start.

## Retrieval questions

- How does NEAT check for updates?
- Does NEAT install an update automatically?
- Why does source NEAT have the assistant but the standalone build may not?
- Why does NEAT report that h5py is missing?
- What do OpenAI authentication, quota and model-access errors mean?
