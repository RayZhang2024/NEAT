# NEAT Assistant — Reviewed Preprocessing Details

Knowledge-base version: NEAT 4.8.2
Approval: reviewed technical guidance for user support

These sections describe reviewed NEAT behaviour. They are intended to answer
precise usage questions without exposing implementation-only detail.

## Summation compatibility requirements

Summation combines corresponding FITS frames pixel by pixel. Every selected run
must contain the same number of frames and the same filename-suffix keys.
Mixed folder depths, missing frames, duplicate suffixes or incompatible image
shapes cause the operation to stop rather than silently pair ambiguous files.

The output contains one summed image for each common frame key. Use Summation
only for compatible repeated runs of the same acquisition.

## Clean invalid-pixel replacement

Clean handles nonfinite or otherwise invalid pixels using valid neighbouring
pixels:

1. calculate the mean of the surrounding 5×5 neighbourhood, excluding the
   center pixel;
2. if that neighbourhood contains no valid value, retry with a 7×7
   neighbourhood; and
3. if neither neighbourhood contains a valid value, leave the original pixel
   unchanged.

Neighbourhoods are clipped at image boundaries. Replacement never wraps to the
opposite side of an image.

## Clean positive-spike rule

A finite positive pixel is treated as a spike when it is at least ten times
the positive finite mean of its surrounding 5×5 neighbourhood. The center
pixel itself is excluded from that mean.

An identified spike is replaced with the same neighbouring mean. The rule is
specifically for large positive outliers; it is not a general smoothing filter.

## Normalisation window defaults and ranges

Normalisation uses two integer controls:

- `n`, Window half: spatial averaging over `(2n+1) × (2n+1)` pixels. The
  supported range is 0–100.
- `m`, Adjacent: averaging over `(2m+1)` wavelength frames. The supported range
  is 0–10.

The reviewed default is `m=0`, meaning only the current wavelength frame is
used. Increasing either value adds smoothing and can remove genuine spatial or
wavelength detail, so inspect the resulting spectrum.

## Preprocessing binary masks

Filtering uses a binary FITS mask with the same rows and columns as every data
frame. Mask value 1 keeps a pixel and value 0 excludes it. A mismatched or
non-binary preprocessing mask is rejected.

This is distinct from the Data Post-Processing mask operation, which warns
about a non-binary mask before applying it.

## Cancellation and output safety

Preprocessing workers use cooperative cancellation. A running file read,
calculation or write may need to reach a safe check before it stops. A Stop
request therefore may not take effect instantly.

NEAT validates required inputs before starting and reports whether a stage
completed, failed or was cancelled. Do not treat the existence of an output
folder alone as proof that every intended frame was processed.
