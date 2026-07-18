---
title: Classic FITS/TIFF image-folder loading
doc_id: neat-tech-loading-classic-images
doc_type: technical_reference
functional_area: loading
audience: [user, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [classic image folders]
scientific_review: pending
source_paths: [NEAT/workers/batch.py, NEAT/ui/mixins/fitting.py]
source_symbols: [load_image_file, write_fits_image_file, ImageLoadWorker, FittingMixin.handle_fits_run_loaded]
test_paths: [tests/test_image_io_orientation.py]
---

# Classic FITS/TIFF image-folder loading

## Selection and naming

The fitting image-folder loader selects `.fits`, `.fit`, `.tiff` and `.tif`.
Each basename must end in an underscore and a 1–10 digit identifier. Files are
lexicographically sorted, then stored by suffix string. A duplicate suffix
overwrites the earlier frame with a warning.

The resulting dictionary is converted to the fitting image sequence in sorted
suffix order. Filename suffixes therefore define spectral/frame order.

## Orientation

FITS and TIFF arrays are flipped vertically at load so the in-memory row order
matches the ImageJ-style display convention used by NEAT. `write_fits_image_file`
flips the row axis again when saving. Tests verify FITS display orientation and
round-trip behavior.

## Spectra association

After images load, the fitting UI searches for a spectra text file and may ask
the user to select one. Its first numeric column becomes the classic ToF array.
If no usable spectra exists, the UI enables manual wavelength-anchor mode.

## Validation and limitations

Unreadable, unsupported and incorrectly named files are skipped rather than
aborting the folder. Image shapes are not consolidated by the loader; later
operations can fail when frames differ. Source FITS headers are not retained in
the in-memory list or generated processing outputs.

## Retrieval questions

- How does NEAT order a FITS image folder?
- Why was an image skipped or overwritten?
- Why does NEAT flip FITS and TIFF images vertically?
- What happens when no spectra file accompanies the images?

