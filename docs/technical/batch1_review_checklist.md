# Batch 1 domain-review checklist

Use this checklist to review scientific intent separately from the
code-verified implementation. Add answers or corrections beneath each item.
Once resolved, the affected page can advance from `code-verified` to
`domain-reviewed`.

## Data and instrument scope

- [ ] Which classic image-folder conventions are IMAT-specific?
  no sure
- [x] Are `.fits/.fit` and ordinary single-frame `.tif/.tiff` intended to be
  treated equivalently in all classic preprocessing blocks?
  yes
- [x] Must generated FITS products preserve any source headers or acquisition
  metadata?
  optional
- [x] Is the exact five-digit Full Process suffix rule intended?
  that's the usual way for naming, but can be relaxed to non-five-digit name
  Implemented: Full Process now accepts a nonempty final name component.

## Summation

- [x] Under what acquisition conditions may runs be added pixel by pixel?
  The runs are identical in all settings, on the same object.
- [x] Is summing column 2 of the complete shutter-count table correct?
  Yes, this will be used for subsequent overlap correction
- [x] Should spectra files remain separate, be combined, or should one
  authoritative spectrum be selected?
  Remain separate
- [x] Should unequal shutter-array lengths and malformed runs abort the whole
  summation rather than continue?
  abort
  Implemented as pre-write validation.

## Clean

- [x] Are all values `<= 0` physically invalid at this stage of the pipeline?
  yes, invalid
- [x] Is a sequential 5×5 positive-neighbor mean the intended replacement?
  yes
- [x] What should happen when a bad pixel has no valid neighbor?
  Search the 5×5 neighborhood first, then expand to 7×7.
- [x] Should high positive spikes also be detected?
  Yes. A finite positive pixel at least 10 times its 5×5 positive-neighbor
  mean, excluding itself, is replaced by that mean.

## Overlap correction

- [ ] What detector/instrument and acquisition mode does this correction apply
  to?
  Not sure
- [x] Confirm the ToF segment boundary threshold `0.0001` and its unit.
  Yes, unit is second
- [x] Confirm the singleton segment interval `1e-5` and its unit.
  Yes, unit is second
- [x] Confirm that shutter values should be filtered at `>1000`.
  Yes
- [ ] Which shutter-count column should be used? The current UI flattens all
  nonzero table values.
  Not sure
- [x] Confirm `p = cumulative_intensity / shutter_count` and
  `image / (1-p)`.
  Yes
- [ ] Should `p >= 1` be rejected instead of clamped to a denominator of
  `1e-10`?
  Not sure
- [x] Is a fixed 512×512 image requirement intended?
  This is due to the current MCP detector used on IMAT contains 512x512 pixels

## Classic normalisation

- [x] Confirm the local open-beam spatial-averaging equation.
  Yes
- [x] Confirm that only open-beam frames, not sample frames, are summed across
  the adjacent-frame window.
  Yes
- [x] Confirm the first-line, second-column shutter-count convention and the
  scale `open_beam/sample`.
  Yes
- [ ] Should missing shutter metadata abort rather than use scale 1?
  Not sure
- [ ] What should zero/open-beam-free neighborhoods produce?
  Not sure
- [x] Define recommended and permitted ranges for `n` and `m`.
  recommended n=10, m=0, ranges n between 0 and 100, m between 0 and 10
  Implemented in both UI paths and worker construction.

## RADEN normalisation

- [x] Confirm that ToF bins/min/max/unit matching is sufficient.
  yes
- [x] Confirm preference of `pulses_with_data` over `pulses`.
  yes
- [x] Confirm the pulse scale `open_beam_pulses/sample_pulses`.
  yes
- [x] Confirm which `.stat`, `.json` and `.log` files must accompany output.
  yes
- [x] Confirm whether the local spatial/temporal equation is valid for RADEN.
  Yes

## Filtering and pipeline order

- [x] Confirm that every nonzero mask value means “keep original intensity.”
  The value of the nonzero mask should be 1
  Implemented as strict binary-mask validation.
- [x] Confirm whether discarded pixels should be zero, NaN, or represented by
  separate mask metadata.
  zero
- [x] Confirm the intended order:
  Summation → Clean → Overlap Correction → Normalisation.
  Yes
- [x] Define when Full Process may safely skip overlap correction.
  Full process should not skip overlap correction
  Implemented: missing sidecars or incomplete correction aborts the pipeline.
- [ ] Define the minimum output-completeness checks before fitting.
  Not sure

## Implementation decisions to resolve

- [x] Should any skipped frame make a worker/pipeline return a failure state?
  Yes
  Implemented with worker `succeeded` and `failed_frames` state for the revised blocks.
- [x] Should output directories be required to be empty or versioned?
  No
- [x] Should Stop be forwarded from Full Process to its active child worker?
  Not sure
  Implemented as an engineering safety improvement; cancellation remains cooperative.
- [x] Should `OverlapCorrectionWorker` rename its input attribute so it no
  longer shadows the `run()` method?
  Not sure
  Implemented because the shadowed method could prevent normal thread execution.
- [x] Which missing focused tests should be implemented before RAG approval?
  Not sure
  Added tests for shutter-length abort, relaxed suffixes, normalisation ranges,
  binary masks, worker completion state, required overlap and Stop forwarding.
