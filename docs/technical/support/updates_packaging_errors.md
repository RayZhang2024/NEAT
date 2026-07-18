---
title: Updates, packaging and operational errors
doc_id: neat-tech-support-updates-packaging-errors
doc_type: technical_reference
functional_area: support
audience: [user, developer, support]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: not-required
source_paths: [pyproject.toml, NEAT.spec, .github/workflows/release.yml, NEAT/app.py, NEAT/ui/main_window.py, tools/assistant_openai.py]
source_symbols: [UpdateCheckWorker, FitsViewer.start_update_check, FitsViewer._on_update_check_finished, describe_openai_error]
test_paths: [tests/test_assistant_openai.py]
---

# Updates, packaging and operational errors

## Source installation and entry points

The package version is `4.8.0`. Core installation declares PyQt5, Matplotlib,
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

## Windows standalone release

Tags matching `v*` run the GitHub release workflow on Python 3.11. It installs
the project with the `assistant` and `assistant-server` extras plus PyInstaller,
runs the unit suite, builds from `NEAT.spec`, runs a non-interactive packaged
smoke test, compresses the one-folder `dist/NEAT` directory and publishes the
zip.

The spec relies on PyInstaller's standard scientific/GUI hooks rather than
recursively collecting dependency test suites. It includes selected runtime
DLLs, provider metadata, dynamic assistant modules, keyring backends and the
approved `docs/assistant` knowledge directory. It bundles the launch splash
and uses `docs/icon/NEAT.ico` when present. The executable is windowed
(`console=False`) and is collected as a directory, not a single-file
executable.

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
the official GitHub release workflow builds and retests on Python 3.11.

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

- Launch the packaged executable on a clean Windows machine.
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
