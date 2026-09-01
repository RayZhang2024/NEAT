# NEAT release checklist

1. Create an issue and implement the scoped change on a branch. Open a PR and
   let CI complete.
2. Merge the reviewed PR into `main`. Confirm the changelog and the single
   package version in `pyproject.toml` are ready for release.
3. Tag the already-merged commit as `vX.Y.Z`, where `X.Y.Z` exactly matches the
   installed NEAT package metadata, then push the tag.
4. The release workflow verifies both invariants before building: the tagged
   commit is contained in `main`, and the tag version matches package metadata.
   It then installs dependencies, runs tests, prepares shared access, builds
   with PyInstaller, smoke-tests the packaged executable, creates the ZIP,
   uploads the artifact, and publishes the GitHub release.
5. Download the published ZIP and perform a final launch check. Test the exact
   candidate at `dist/NEAT/NEAT.exe`, not the source checkout.

## Recovery before publication

If the workflow fails before publishing, fix the issue on a new PR, merge it
to `main`, and create a new tag for the corrected commit. Do not retag or
publish a commit that is not in `main`; a mismatched tag or package version
must be corrected before retrying.

## Packaging note

The local Python 3.13 package may be unable to initialise packaged ONNX
Runtime. The release smoke test verifies the supported BM25 retrieval fallback;
this does not claim that semantic ONNX retrieval is available in the package.
