# M5A-1 OpenAI visual pre-fitting review

M5A-1 is a read-only visual rehearsal that ends at **user review**. It makes no
fit, map, phase-table edit, selected-ROI change, display-range change or saved
file. It does not establish crystallographic identity or fitness for strain
analysis. The legacy `d_fit`/factor-of-two physical-interpretation caveat
remains unresolved.

## One model and automatic OpenAI evidence transfer

- The OpenAI model ID and Agent-only API key saved in **Agent settings** are
  shared by metadata planning and visual review. There is no second vision-model
  configuration. The optional loopback text planner remains available, but
  visual review is only offered for OpenAI Cloud mode.
- In OpenAI mode, pressing **Start Autonomous Analysis** authorizes the
  disclosed run-level transfers. When preflight has enough evidence, the run
  sends one canonical, metadata-only planning request. A normal image-stack
  visual review can send up to **7 PNGs in 3 vision requests**: up to 3 actual
  frame previews, 1 measured ROI-spectrum plot, and up to 3 trusted
  phase-overlay plots. Together with metadata planning, the run has at most 4
  model requests. Saving settings or opening the request preview sends nothing.
  Local-model mode stays on its configured loopback endpoint and sends no
  visual evidence to OpenAI; model failure does not substitute an offline plan.
- If an image stack has no selected NEAT ROI, M4 metadata planning is skipped
  rather than weakening its preflight validator or claiming a plan. With a
  valid wavelength axis and bounded image previews, the OpenAI model can run a
  visual-first ROI discovery path: it inspects up to 3 real frames, suggests a
  bounded ROI, and NEAT extracts that ROI's spectrum with `get_roi_spectrum()`
  before continuing with the same phase-shortlist and overlay comparison. If
  the image previews or wavelength calibration are unusable, the run explains
  why it cannot continue. This direct path does not report an M4 planning
  result.
- The OpenAI endpoint is fixed to `https://api.openai.com/v1/chat/completions`.
  The Agent keyring key is sent in the authorization header only. Requests set
  `store: false`; redirects, proxy environment settings and retries are disabled.
  Request/response bodies, previews and model text are not logged.
- The visual request may contain the bounded image/spectrum preview PNGs, a
  short task prompt and the configured model ID. PNGs retain the scientific
  labels needed for review (wavelength, intensity, phase, HKL and ROI
  coordinates where relevant), but omit dataset IDs, run IDs and revision
  numbers. The request does not upload the image stack, source files, file
  paths/names, sample title, credentials in the request body, or unbounded
  arrays.
  Each PNG is at most 1 MB, each serialized request at most 4 MB and each response
  at most 12 KB. A request is limited to 45 seconds and the visual stage to 150
  seconds total.

## Scientific and trust boundaries

- Frame evidence comes from NEAT's bounded `get_image_preview()` operation, up
  to 512 pixels on each side. Original-to-preview coordinates are retained.
  The proposed ROI is validated as a half-open rectangle in original pixels.
  `get_roi_spectrum()` extracts the measured spectrum using NEAT's existing
  arithmetic. Rendering the full wavelength source does not truncate the data.
- Phase candidates come only from NEAT's built-in `PHASE_DATA`. Theoretical
  dashed marker positions come from `calculate_theoretical_bragg_edges()`; the
  model cannot invent a phase/HKL or calculate a marker position. This visual
  proposal is not a deterministic phase identification or an edge fit.
- Per-edge fitting windows are not proposed in this milestone. Existing fitting
  windows, mapping settings and the user's fitting configuration stay untouched.
- The model output is validated against bounded schemas and NEAT's scientific
  allowlists. An invalid response is rejected without changing application
  state. Ambiguity can be returned instead of a phase/ROI proposal.
- The worker retains its captured dataset handle only until thread retirement.
  Stop cooperatively cancels an active HTTP task, stale dataset revisions reject
  results, and shutdown waits for tracked workers. A provider may continue
  server-side generation after local cancellation.

The Agent window shows the proposed ROI overlay and measured spectrum with
trusted phase overlays. A user may reject the proposal or adjust the ROI
rectangle. Adjusting the ROI withdraws the prior spectrum and phase hypothesis
because they describe the earlier ROI. **Start Autonomous Analysis** then
reuses the adjusted ROI only within the Agent window, re-extracts its spectrum,
and regenerates the phase overlays and recommendations. The override is bound
to the dataset revision and discarded after a dataset change or explicit
rejection. No review control writes to the fitting tab.

## Windows GUI smoke procedure (opt-in)

1. In **AI Agent → Agent settings**, configure one OpenAI model ID and Agent API
   key. The same model is used for planning and vision. No API key is required
   for CI or offline tests.
2. Load an image stack with a valid wavelength axis and start **Autonomous
   Analysis**. In OpenAI mode, the run automatically sends the disclosed
   canonical dataset summary and bounded evidence PNGs. If the model is
   unavailable or returns unusable output, the run stops with the reported
   reason; no deterministic offline plan is substituted.
3. Confirm the rendered frame/ROI overlay uses original pixel coordinates, the
   spectrum corresponds to that ROI, and any phase overlays use trusted NEAT
   markers. Confirm the main fitting tab and its phase/window/ROI settings have
   not changed.
4. Stop during a second run; confirm `Stopping…` remains visible until the
   worker retires and no late proposal appears. Change the dataset during a run
   and confirm a stale result is rejected.
5. Try an ambiguous or intentionally different dataset. The model must be able
   to return uncertainty, not automatically repeat a prior phase or ROI.

For the October 10 rehearsal data, blind the model to earlier Fe_bcc, 211/200/110
and 2.18–4.44 Å human selections. Record the actual model-visible previews,
proposed ROI/phase/edges, rejected calls, time and uncertainty first; compare
with the reference only afterward. Real API smoke tests are opt-in and are not
part of CI. Automated tests use injected fake models and synthetic PNGs only.
