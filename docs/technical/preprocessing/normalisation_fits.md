---
title: Classic FITS/TIFF folder normalisation
doc_id: neat-tech-preprocessing-normalisation-classic
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: e62c089a8e62f29875fc1a5232b5688363bdac5d
status: code-verified
instrument_applicability: [classic image folders]
scientific_review: pending
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py, NEAT/services/preprocessing_normalisation.py]
source_symbols: [PreprocessingMixin.normalise_images, NormalisationWorker, normalise_loaded_image_runs, normalise_classic_frame, validate_normalisation_windows]
test_paths: [tests/test_preprocessing_workers.py, tests/test_preprocessing_normalisation.py]
---

# Classic FITS/TIFF folder normalisation

## Purpose and UI parameters

This block divides each sample frame by a spatially averaged open-beam signal,
with optional averaging across adjacent open-beam frames and shutter-count
scaling.

| UI value | Code meaning | Default |
|---|---|---:|
| spatial half-window `n` | square width is `2n+1` | 10 (21×21) |
| adjacent-frame half-window `m` | uses up to `2m+1` open-beam frames | 0 |
| base name | output prefix | `normalised` |

Both values must be integers. The permitted ranges are `0 <= n <= 100` and
`0 <= m <= 10`; invalid values prevent worker construction or UI launch.

## Pairing and scale

The headless service requires equal numbers of sample and open-beam runs. Within
each ordered pair it processes only the lexicographically sorted intersection
of filename suffixes. Extra open-beam frames do not enter the temporal window.

From the first raw-directory-order `_ShutterCount.txt` in each folder, it parses
the second token of line 1. The multiplicative scale is:

```text
scale = open_beam_count / sample_count, when sample_count > 0
```

Missing or malformed shutter data falls back to `scale = 1`. A nonpositive or
NaN sample count also yields scale 1, but follows the successful-reader message
path. Open-beam zero, negative and non-finite values are not pre-rejected;
the ordinary ratio and final NaN/Inf-to-zero handling are retained.

## Algorithm

For sample frame `i` in the sorted common-suffix list, open-beam frames from
`max(0,i-m)` to `min(last,i+m)` are summed. At stack ends, fewer frames are
used. Sample frames are never temporally summed. The pure
`normalise_classic_frame` helper preserves the original float64 open-beam
accumulation and float32 sample/output dtype ordering.

For each pixel, an integral-image calculation obtains the open-beam sum in the
clipped `(2n+1)×(2n+1)` neighborhood. Border neighborhoods are rescaled to the
full nominal area. Let `K` be the number of adjacent open-beam frames actually
included, `A=(2n+1)^2`, `S` the sample pixel and `B_scaled` the border-adjusted
local open-beam sum:

```text
normalised = K × A × S / B_scaled × scale
```

This is equivalent to division by the local mean open-beam response after
compensating for the number of temporally summed open-beam frames. NaN and
infinite outputs are converted to zero.

## Validation and outputs

Sample and combined open beam must have identical shape, and both dimensions
must be at least `2n+1`. Violations mark the worker failed/incomplete. Output is
`<base>_<suffix>.fits` as `float32`. Multiple run pairs may overwrite the same
filename; each successful write still counts, and structured output descriptors
retain repeated paths in production order. Sample spectra and shutter sidecars
are copied as `Run<index>_<original-name>`, with all Spectra files before all
ShutterCount files in raw directory order. A copy error is a warning, not a
frame failure.

The headless service returns `PreprocessingOperationResult`: `expected_count`
is all supplied sample frames, and `processed_count` counts frames that pass
the legacy post-write deletion step. The worker adapter immediately deletes a
successfully written suffix from its legacy mutable sample dictionary. At the
end of an entered run, it clears remaining sample frames, collects garbage,
then copies sidecars and announces `Run done.`. The service never mutates the
read-only `LoadedImageRun.frames` mapping. If legacy deletion raises after a
FITS write, the file remains recorded in `outputs` but that frame does not
advance processed count or progress. Shared mutable image dictionaries across
different run entries (or between sample and open beam) are outside the
supported compatibility contract.

## Progress, performance and cancellation

The open-beam spatial average uses integral images, avoiding a direct window
loop per pixel. Progress divides successful outputs by the original number of
sample frames, so suffix/shape/window skips can prevent 100%. The worker pauses
five seconds after each entered run while still running and reports process
memory at completion. Stop is checked between runs and frames, not within a
frame. An inner-frame stop still allows entered-run clear, sidecars and `Run
done.`; a later run boundary then reports `User stopped the process.`. Final
status is sampled after sidecars and pacing, so cancellation during either can
still produce `CANCELLED`. Early no-run/count-mismatch aborts omit the normal
completion summary but retain worker memory reporting and one `finished`.

## Known limitations

- Only common suffixes are used. Missing counterparts make the final success
  flag false, although successfully matched outputs may already be written.
- Zero open-beam neighborhoods become nonfinite during division and are
  silently converted to zero.
- Shutter parsing uses only the first row and silently degrades to scale 1.
- The previous string-suffix-versus-integer `1500/2500` branches are unreachable
  for valid string suffixes and are not part of the headless service.
- The worker assumes its output directory already exists; the UI creates it.
- Golden values were captured by running the original worker at
  `376be7d471bfe47caa53445a6391799898fe0ef1` on deterministic synthetic
  arrays. Focused tests now protect `n=0,m=0`, border rescaling, multi-frame
  temporal averaging, shutter fallback/non-finite behavior, output orientation,
  collisions, progress and destructive worker-state timing. Exact float32
  baseline values are asserted where reproducible.
- The physical appropriateness of the spatial/temporal averaging and shutter
  scale needs scientific review.

## Retrieval questions

- What do `n` and `m` mean in normalisation?
- How is the shutter-count scale calculated?
- Why were some sample frames skipped or output as zero?
- Are sample frames also summed across the adjacent-frame window?
