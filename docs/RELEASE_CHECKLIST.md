# NEAT 4.8.2 release checklist

## Automated checks completed locally

- [x] Version is `4.8.2` in the package and project metadata.
- [x] Full unit suite passes (`171` tests).
- [x] Ruff critical-error checks pass.
- [x] Mypy checks for `NEAT/core` pass.
- [x] `git diff --check` passes.
- [x] No API-key-shaped secrets were found in release sources.
- [x] Windows one-folder package builds successfully.
- [x] Packaged smoke test loads provider adapters, keyring and all `149`
  approved knowledge sections.
- [x] Packaged BM25 retrieval fallback returns results without an API call.
- [x] Packaged smoke test confirms automatic shared access is configured.

## Manual acceptance checks required before tagging

Test the exact candidate at `dist/NEAT/NEAT.exe`, not the source checkout.

- [ ] Launch NEAT and confirm the splash screen, main window and **AI
  Assistant** menu open normally.
- [ ] Open representative FITS/TIFF data and perform one short preprocessing,
  fitting and post-processing workflow. Confirm saved image orientation and
  output folders are correct.
- [ ] Open **AI Assistant > Settings**. Confirm no personal key is displayed in
  plain text and a saved/environment key is reported only by availability.
- [ ] Ask at least three known NEAT questions. Confirm the answers are grounded,
  citations open the correct bundled sections, and unsupported details are not
  invented. One good test is **Explain current screen** from a fitting screen.
- [ ] If personal cloud access is part of the release, test one provider with a
  low-cost request. This is a real, billable API call.
- [ ] If shared access is part of the release, keep the laptop server and
  Tailscale running and test the freshly downloaded ZIP on a clean Windows user
  without shared-access environment variables. Confirm shared access is enabled
  automatically and the global daily quota message is correct.
- [ ] If local-model support is part of the release, test one installed Ollama
  or LM Studio model. This is optional when no supported local server is
  available.
- [ ] Disconnect the network and verify core NEAT functions still work. The AI
  assistant should show a useful provider error; bundled document retrieval
  must not crash NEAT.
- [ ] Preferably unzip and test the candidate under a different Windows user or
  a clean Windows machine without the source virtual environment.
- [ ] Review `CHANGELOG.md`, the README release link and the user-manual AI
  section for public wording.

## Known release-candidate observation

The locally built Python 3.13 package cannot initialize packaged ONNX Runtime,
so it uses the tested BM25 document-retrieval fallback. This does not disable
the assistant. The official GitHub release workflow builds with Python 3.11
and repeats the packaged smoke test; semantic retrieval remains optional in
either case.

## Publication gate

Only after the manual checks pass:

1. commit the reviewed release files;
2. push the release commit;
3. create and push tag `v4.8.2`;
4. wait for the GitHub release workflow to pass;
5. download the produced `NEATv4.8.2.zip` and perform a final launch check;
6. publish/announce the release.
