---
title: Mapping ROI, boxes, steps and result coordinates
doc_id: neat-tech-mapping-geometry
doc_type: technical_reference
functional_area: mapping
audience: [user, scientist, developer]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [image stacks]
scientific_review: pending
source_paths: [NEAT/ui/mixins/fitting.py, NEAT/workers/batch.py]
source_symbols: [FittingMixin.batch_fit_edges, FittingMixin.batch_fit, BatchFitEdgesWorker.run, BatchFitWorker.run]
test_paths: [tests/test_fitting_headless.py]
---

# Mapping ROI, boxes, steps and result coordinates

Mapping uses a half-open ROI `x[min,max), y[min,max)`. Box width/height and
step X/Y must be positive integers. Default UI values are 20×20 pixels, step
5×5, with interpolation enabled.

Top-left box positions are:

```text
y = min_y ... max_y-box_height, increment step_y
x = min_x ... max_x-box_width, increment step_x
```

The number of boxes is the product of the two corresponding integer-division
counts. A box must fit completely inside the ROI. Results are stored at:

```text
center_row = top + box_height//2
center_col = left + box_width//2
```

For even box dimensions this is the lower/right of the four geometric central
pixels. Output `x` is array column and `y` is array row.

Every mapping spectrum is a pixel sum, not a mean. This differs from the
interactive yellow ROI, which divides by pixel count. Scientific review must
decide whether that distinction is intended, especially when box sizes change.

## Retrieval questions

- How does NEAT choose mapping box positions?
- Are mapping ROI maxima inclusive?
- At which pixel is a box result stored?
- Does mapping sum or average the pixels?

