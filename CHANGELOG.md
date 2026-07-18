# Changelog

All notable user-facing changes to NEAT are recorded here.

## 4.8.1 - 2026-07-18

### Changed

- Official standalone downloads now enable NEAT shared AI access automatically;
  users no longer need to configure a service URL or access token.
- Shared-access labels describe a limited daily request allowance without
  advertising a fixed number in the interface.
- Complete per-computer shared-service settings continue to override the
  bundled public configuration for development and testing.

### Packaging and security

- Release automation injects the limited public-client service credential from
  a GitHub Actions secret and stops the build when it is missing.
- The packaged smoke test verifies that automatic shared access is configured.
- The server OpenAI API key remains exclusively on the hosted laptop and is
  never written to the release package.
- The bundled public-client token is intentionally extractable; the atomic
  server-side daily allowance and OpenAI project budget remain the cost
  controls.

## 4.8.0 - 2026-07-18

### Added

- Dockable NEAT AI Assistant with grounded answers and verified source links.
- Dedicated **AI Assistant** menu, settings window and **Explain current
  screen** action.
- Personal model access for OpenAI, Anthropic, Google Gemini, DeepSeek and
  Kimi/Moonshot, plus manually configured OpenAI-compatible endpoints.
- Local-model support through Ollama and LM Studio, including automatic model
  discovery and no-key localhost operation.
- Optional hosted shared access with a global daily request allowance.
- Curated software, UI, preprocessing, fitting, mapping, post-processing,
  troubleshooting and known-limitations knowledge for assistant retrieval.
- Local helpful/not-helpful feedback recording with path redaction.
- Rectangle and polygon mask-editing tools and a mask-applied preview.
- Uncertainty-planning estimator and additional fitting controls and tests.
- Full Process data and open-beam folder selection controls.

### Changed

- Normalisation defaults to adjacent-frame half-window `m = 0`; validated
  ranges are `0-100` for spatial half-window `n` and `0-10` for `m`.
- Positive spike cleaning uses a `10x` threshold relative to the surrounding
  `5x5` mean, excluding the centre pixel.
- Clean expands to a `7x7` neighbourhood when the initial neighbourhood has no
  valid replacement value.
- FITS and TIFF image orientation is handled consistently across loading,
  processing and writing.
- Fitting-window, fixed-parameter uncertainty, phase-definition and `d_110`
  naming behaviour have been clarified and tested.
- Assistant indexes, settings and feedback are written to per-user application
  data rather than the source or executable directory.
- Assistant retrieval falls back to bundled BM25 search if the optional ONNX
  semantic runtime or first-use model download is unavailable.

### Packaging and security

- Windows release builds include assistant providers, keyring support, Chroma,
  ONNX Runtime and the approved knowledge base.
- CI installs and tests both desktop-assistant and shared-server dependencies.
- The packaged release has a non-network import and knowledge-base smoke test.
- API keys and hosted-service secrets remain outside committed settings and
  release artifacts.

### Known limitations

- Shared access is disabled unless the service URL and access token are
  explicitly configured for the installation.
- Local LLM answer quality and performance depend on the selected model and
  hardware; users should validate models against representative NEAT questions.
- Scientific interpretation remains subject to the assumptions and review
  boundaries documented in the assistant knowledge base.
