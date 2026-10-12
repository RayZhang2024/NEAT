# M5A-1 local visual pre-fitting review

M5A-1 is an opt-in visual rehearsal, ending at **user review**. It makes no fit,
map, phase-table edit, selected-ROI change, display-range change or saved file.
It does not establish crystallographic identity or fitness for strain analysis.
The older `d_fit`/factor-of-two physical-interpretation caveat remains unresolved.

## Data and trust boundaries

- Ordinary Start behavior remains M4 read-only preflight followed by metadata-only
  planning. Local visual review requires a configured vision model **and** a fresh
  checked box in the Agent window for that run. A missing model produces an
  explicit unavailable state and no invented visual observation.
- The separate local vision adapter accepts only an explicit numeric loopback
  `http://127.0.0.1:<port>` or `http://127.0.0.1:<port>/v1` endpoint (IPv6 `::1`
  is also accepted). It sends OpenAI-compatible `/chat/completions` requests
  with image content parts of type `image_url` containing `data:image/png;base64`
  URLs, and `response_format: {"type":"json_object"}`. A server/model must
  actually support multimodal input and this JSON response format. Text-only
  servers fail validation. No credentials are sent to this endpoint.
- No vision PNG is added to M4's canonical planning state or cloud adapter.
  M4 cloud consent covers metadata only and cannot enable image transmission.
  The local transport disables proxies and redirects. Request and response
  bodies, image bytes and untrusted model text are not logged.
- Frame evidence comes from preflight `get_image_preview()` at up to 512 pixels
  per side. Original-to-preview coordinates are retained. The proposed ROI is
  validated as a half-open original-pixel rectangle. `get_roi_spectrum()` then
  extracts the actual measured spectrum with NEAT's existing arithmetic.
  Rendering the full wavelength source does not truncate the underlying data.
- Phase candidates are from NEAT's built-in `PHASE_DATA`; theoretical dashed
  marker positions come from `calculate_theoretical_bragg_edges()`. The model
  does not calculate those positions and cannot introduce a custom phase/HKL.
  Per-edge fitting windows are intentionally not proposed in this first slice;
  existing fitting windows remain untouched.

## Bounded sequence

There are at most **three** local vision requests and no agent loop:

1. Up to three real frame PNGs → a proposed ROI, short reason, confidence
   category and ambiguity flags.
2. One actual measured ROI-spectrum PNG plus trusted phase-edge catalogue →
   shortlist of at most three built-in phases.
3. Up to three measured-spectrum/phase-overlay PNGs → provisional phase, up
   to five candidate HKLs with visual reasons, a finite in-range global
   **display** interval, alternatives and unresolved checks. The model may
   report uncertainty instead of choosing a phase.

Each PNG is at most 1 MB; a serialized request is at most 4 MB; a response is
at most 12 KB. Each request times out after 45 seconds and the visual stage
has a 150-second total budget. Stop cancels an active HTTP task where the local
server cooperates; local server-side generation may continue independently.
The Qt worker remains tracked until it exits, and results from a stopped,
stale or closing run are discarded. Captured stack handles are released at
worker retirement.

The Agent window shows the real ROI overlay and measured spectrum with each
inspected phase overlay. The user can reject the proposal or adjust the ROI
rectangle for a **preview only**. Adjusting withdraws the old spectrum and
phase hypothesis because they describe the prior ROI; rerun visual review for
current spectral evidence. No review control writes to the fitting tab.

## Windows GUI smoke procedure (opt-in)

1. Start a compatible local multimodal server, identify the exact vision model
   ID and numeric loopback `/v1` endpoint. Confirm that it accepts image URL
   data parts and returns JSON objects. Do not configure an OpenAI cloud URL.
2. Launch this branch's NEAT build. Load an image stack with a valid wavelength
   axis. Open **AI Agent → Agent settings…**, save local vision model and
   endpoint. The M4 planner can remain Offline.
3. In the Agent window, check **For this run, send bounded image and spectrum
   previews to my local vision model**, then press **Start Autonomous
   Analysis**. The check box resets for the next run.
4. Confirm the frame/ROI overlay uses original pixel coordinates, the spectrum
   corresponds to that ROI, the phase overlays have dashed theoretical markers
   and labels, and the proposal explains its uncertainty. Confirm the main
   fitting tab and its phase/window/ROI settings have not changed.
5. Press Stop during a second run; confirm `Stopping…` remains visible until
   the worker retires and no late proposal appears. Change the dataset during
   a run and confirm a stale result is rejected.
6. Try an ambiguous or intentionally different dataset. The model must be able
   to return uncertainty, not automatically repeat a prior phase or ROI.

For the October 10 rehearsal data, blind the model to the earlier Fe_bcc,
211/200/110 and 2.18–4.44 Å human selections. Record model-visible previews,
selected ROI/phase/edges, rejected calls, time and uncertainty first; compare
with the reference only afterward. A real local model smoke is manual and
never part of CI. CI uses injected fake inspectors with actual synthetic PNGs.
