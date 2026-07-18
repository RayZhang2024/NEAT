---
title: Classic FITS/TIFF folder normalisation
doc_id: neat-tech-preprocessing-normalisation-classic
doc_type: technical_reference
functional_area: preprocessing
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [classic image folders]
scientific_review: pending
source_paths: [NEAT/ui/mixins/preprocessing.py, NEAT/workers/preprocessing.py]
source_symbols: [PreprocessingMixin.normalise_images, NormalisationWorker]
test_paths: [tests/test_preprocessing_workers.py]
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

The worker requires equal numbers of sample and open-beam run objects. Within a
pair it processes only the intersection of filename suffixes.

From the first `_ShutterCount.txt` in each folder, it parses column 2 of line
1. The multiplicative scale is:

```text
scale = open_beam_count / sample_count, when sample_count > 0
```

Any missing or malformed sidecar, or a nonpositive sample count, falls back to
`scale = 1`.

## Algorithm

For sample frame `i`, open-beam frames from `max(0,i-m)` to
`min(last,i+m)` are summed. At stack ends, fewer frames are used. Sample frames
are never temporally summed.

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
`<base>_<suffix>.fits` as `float32`. Sample spectra and shutter sidecars are
copied as `Run<index>_<original-name>`.

After writing, each processed sample image is removed from the in-memory run
dictionary to reduce memory use.

## Progress, performance and cancellation

The open-beam spatial average uses integral images, avoiding a direct window
loop per pixel. Progress divides successful outputs by the original number of
sample frames, so suffix/shape/window skips can prevent 100%. The worker pauses
five seconds after each run and reports process memory at completion. Stop is
checked between runs and frames.

## Known limitations

- Only common suffixes are used. Missing counterparts make the final success
  flag false, although successfully matched outputs may already be written.
- Zero open-beam neighborhoods become nonfinite during division and are
  silently converted to zero.
- Shutter parsing uses only the first row and silently degrades to scale 1.
- Two garbage-collection branches compare string suffixes with integer 1500
  and 2500 and are therefore likely unreachable.
- The worker assumes its output directory already exists; the UI creates it.
- A focused test verifies local division and shutter scaling for `n=0, m=0`.
  Border windows and multi-frame temporal averaging remain untested.
- The physical appropriateness of the spatial/temporal averaging and shutter
  scale needs scientific review.

## Retrieval questions

- What do `n` and `m` mean in normalisation?
- How is the shutter-count scale calculated?
- Why were some sample frames skipped or output as zero?
- Are sample frames also summed across the adjacent-frame window?
