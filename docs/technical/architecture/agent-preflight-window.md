# AI Agent window: one-click preflight and planning (M3)

## User-visible behaviour

The independent, non-modal AI Agent window keeps its Overview and Visual
Evidence panels side by side and does not replace the AI Assistant dock.
Opening the window is side-effect-free. **Start Autonomous Analysis** creates
one parent run and automatically advances through local preflight and the
configured LLM planning stage. In OpenAI mode, image-stack runs then review
bounded visual evidence with that same model. Starting an OpenAI run sends the
disclosed minimized dataset summary and, for image stacks, bounded evidence
PNGs automatically. Local-model requests stay on loopback and do not send
visual evidence to OpenAI. If no LLM is configured or the model fails, the run
shows an error and does not substitute an offline plan. No second preview or
continue action is required. The dataset check is an internal stage of this
single action; the window does not expose a separate diagnostic button, plan
focus selector, or detailed dataset log. It keeps a short dataset summary and
the measured visual evidence. The window permits only one active stage at a
time; Stop applies to whichever stage is running.

The planning terminal message is **“Plan prepared — scientific execution is
not yet implemented.”** The plan and visual proposal are read-only. M5A-1 adds
LLM-assisted visual review, but performs no edge fitting, mapping, phase
identification, physical interpretation, report generation, or source-data
changes. A missing optional dependency, invalid dataset, insufficient evidence,
stale revision, cancellation, and internal failure each have explicit
non-success outcomes.

## Components and data boundary

- `NEAT.domain.agent_planning` defines typed plan contracts, terminal states,
  allowlisted proposal types, and explicit state/text/evidence bounds.
- `NEAT.services.agent_planning` builds and runs the actual LangGraph
  `StateGraph` with nodes `validate_preflight -> propose_read_only_plan ->
  validate_plan -> terminal`. Invalid preflight and missing evidence take a
  finite blocked route. The built-in proposal is deterministic and offline.
  A future proposal seam cannot dispatch tools: plan validation rejects
  unsupported step names, unsupported status, oversized output, inconsistent
  permission classification, and evidence IDs not issued by that preflight.
- `NEAT.ui.agent_planning_worker` is a tracked `QThread` that receives only a
  bounded planning context. It never receives a dataset handle, image stack,
  preview PNG, spectrum array, widget, source path, credential, or provider.
- `NEAT.ui.agent_window` chains planning only after the preflight completion
  callback is current and the preflight thread has finished. It guards every
  callback with run identity, worker identity, dataset ID, and revision.
- `NEAT.ui.main_window` registers the planner worker with the existing
  `WindowThreadInventory`. Window Stop, dataset invalidation, and application
  shutdown request cooperative cancellation and keep ownership until native
  thread exit. A stale dataset clears displayed evidence and the plan.

The graph receives only bounded JSON-compatible scalar values: parent and
preflight run IDs, dataset ID and revision, mode/readiness/axis enums,
capability booleans, evidence reference strings, summary/warning strings, and
bounded typed plan steps. The context builder copies these from the successful
Issue #55 preflight; it does not copy `PreflightResult` arrays or PNG data into
the planning worker. Evidence references are opaque allowlisted IDs such as
`image:frame:2` and `spectrum:plot`; they never identify a file or carry the
evidence content.

LangGraph and LangSmith are loaded lazily from the separate `NEAT[agent]`
extra. A baseline NEAT install can import the domain/service and open the Agent
window without either dependency. If a user starts analysis without the
required graph dependency, the window reports the missing dependency; it never
substitutes an offline plan or reports a successful plan. The planning graph
has no provider adapters or API-key access. The phase-comparison, visual-location
retry, and local-fitting loops also use bounded LangGraph state graphs. No graph
state is persisted during ordinary runs. Tests may inject an in-memory
checkpointer; SQLite durability and restart/resume are future work.

LangSmith tracing is off by default and is separately enabled in Agent
settings. Its API key is stored in the OS keyring, separate from the Agent's
OpenAI key. When enabled, the planning, phase-comparison, location-selection,
and local-fitting graphs send traces to the selected LangSmith agent and
environment using the configured regional endpoint. The first successful trace
creates the agent automatically. A workspace ID is optional for API keys linked
to multiple workspaces. Direct model requests are recorded as `llm` child runs
under graph nodes; the initial reference-ROI request is an LLM run in the same
agent. LLM runs contain the request and response, including bounded PNG previews
for visual calls. LangSmith receives graph inputs, outputs, metadata, errors,
and runtime information without client-side redaction; these may include
dataset-derived values, ROI coordinates, spectra, or rendered previews. Graph
and node names identify the workflow stages. The settings connection test
submits one synthetic trace with a fixed test payload and records its separate
model request as an LLM run when tracing is enabled. When
disabled, each graph explicitly suppresses inherited LangSmith tracing settings
from the process environment.

## Worker, revision, and shutdown rules

The Qt owner thread captures the scientific dataset handle and selected ROI
for Step 0. The existing preflight worker releases its handle before
completion. Only then is a safe planning context constructed and passed to a
second tracked worker. The planner checks a cooperative stop event and the
thread-safe dataset revision clock at graph-node boundaries and before
publishing its result. It has a four-node fixed graph and no retry or
refinement loop.

Starting again while either worker is active is ignored and the controls stay
disabled. The internal preflight entry point also rejects overlapping work.
Closing the Agent window requests a stop, discards current displayed
results, invalidates that run's callback IDs, and hides the reusable window.
The main window's shutdown veto waits for both worker roles to retire before
data cleanup. Planning results contain no references to old image stacks; the
preflight result itself retains only the bounded evidence already displayed
by the Agent window and is cleared on stale data or close.

## Future autonomy and human checkpoints

The binding future policy is designed for one initial action after the user
sets a goal and permission envelope:

- Routine operations already inside granted permissions advance without a
  prompt. The controller may validate, evaluate, and replan within at most
  three refinement rounds per identified issue.
- A meaningful, pre-authorised noncritical decision offers a real ten-second
  interruption window, then continues only if that exact operation is inside
  the run-level permission grant. This is not a delay on every node or tool.
- Critical scientific decisions, destructive or file-overwriting operations,
  out-of-scope permissions, sensitive/cloud transmission, and unresolved
  privacy or safety concerns always require explicit approval or stop. Silence
  or a timeout never approves them. Cloud image/plot transmission is denied
  by default.
- Pause, Resume, and Stop belong to deterministic controller code. Stop is
  cooperative and remains visibly in progress until safe termination.

M3 did not implement the ten-second timer, Pause/Resume, approval UI, durable
checkpoints, autonomous execution, or live model reasoning. M5A-1 adds only
read-only model-assisted review; it does not implement execution or timed
approval. Those require later reviewed milestones. Scientific review
of `d_fit` / CSV factor-of-two / HKL-denominator conventions remains a separate
gate before fitting or physical claims can be automated.

## Manual smoke test

1. Open **AI Agent** and confirm the window is side by side, the AI Assistant
   remains available, and opening the Agent starts no analysis. With no LLM
   selected, press **Start Autonomous Analysis** and confirm the window explains
   that no LLM is configured; it must not present an offline substitute plan.
2. Configure one OpenAI model and Agent API key, load an image stack, and press
   **Start Autonomous Analysis once**. Confirm the disclosed minimized summary
   and bounded evidence transfer behavior, automatic preflight → planning →
   visual review, and that the plan and visual suggestions remain read-only.
3. Repeat with a local loopback model. Confirm requests stay local and no visual
   evidence is sent to OpenAI. With an unavailable model or invalid output,
   confirm the run ends with a reason instead of a deterministic fallback.
4. Repeat with no ROI, invalid wavelength axis, empty data, or a spectrum-only
   import. Confirm no phase, edge, or fitted value is invented and the visual
   stage is skipped when image-stack evidence is unavailable.
5. Stop during a stage, reload or clear the dataset, and close/reopen the Agent
   window. Confirm stale evidence is discarded, Stop drains safely, the window
   can be reused, and NEAT shutdown waits for worker retirement. Confirm no
   LangSmith request while tracing is off (the default), fitting, mapping,
   ten-second countdown, or source write. As a separate optional check, enable
   LangSmith with a user-supplied key and confirm graph nodes have visible LLM
   child runs containing the request and response, including bounded previews
   for visual calls.

## M4 structured planner and M5A-1 visual review

M4 adds a structured-output proposer behind the existing LangGraph proposal
seam. The no-model mode requires no model, key, HTTP client, or cloud access;
starting autonomous analysis in that mode reports that no LLM is configured.
Optional `NEAT[agent]` adds LangGraph, LangSmith, and OS keyring integration.
`NEAT[agent-llm]` adds the bounded HTTPX model transport alongside those
dependencies. The RAG Assistant
settings file and `NEAT AI Assistant` keyring service are never read by the
Agent. Agent model selection is stored separately in the user's
`NEAT/agent_model_settings.json`; secrets use the separate `NEAT AI Agent`
keyring service. The optional `OPENAI_API_KEY` fallback is off by default and
can only be enabled explicitly in Agent settings. A keyring value takes
precedence. Neither credentials nor cloud permission are stored in settings.

The supported choices are no model, a strictly loopback local OpenAI-compatible
endpoint (`127.0.0.1`, `[::1]`, or `localhost`), and OpenAI
cloud. The exact model ID must be entered by the user; no cloud model default,
probe, discovery, or fallback provider is manufactured. Local requests disable
proxy environment variables and redirects; `localhost` is pinned to IPv4
loopback without DNS, and other remote hosts are rejected. There is no
local-to-cloud failover. A local server must support OpenAI-compatible chat
completions with JSON-schema structured output; unsupported servers fail
clearly instead of falling back to unrestricted text. A manually activated
connection test sends one small test-only request, not NEAT data, and does not
start an analysis run.

Cloud mode requires a configured OpenAI model. Pressing Start authorizes the
run's disclosed transfers; saving settings alone sends nothing. For an eligible
image-stack run with a selected NEAT ROI, OpenAI receives the canonical
minimized planning summary, followed by M5A-1 visual review. The visual stage
can send up to 7 bounded PNGs in 3 requests: up to 3 image previews, 1 measured
ROI-spectrum plot, and up to 3 trusted phase-overlay plots. If there is no
selected ROI but the image stack, bounded previews, and wavelength calibration
are valid, M4 planning is skipped without weakening its validator or claiming a
plan; M5A-1 instead starts a visual-first ROI-discovery path and then extracts
the measured spectrum with the read-only observation API. Unusable previews or
calibration stop that path with a reason. Limited or invalid data never triggers
an offline substitute, while stale or cancelled preflight sends no request.
Local-model mode stays on loopback and does not send visual evidence to OpenAI.
Changing settings during a run affects only a future run. Missing keys and
provider failures produce a safe error with the available reason; no Assistant
key is inherited.

The user-facing **Preview Data Sent** is generated from the canonical planning
request schema. The planning request payload is restricted to the schema
version, one enumerated objective, dataset mode, three capability booleans,
axis-validity, ROI-presence, measured-spectrum-evidence availability,
calibration-provenance-known, non-finite-intensity-present, a closed set of
trusted warning codes, and available enumerated review-step types. It omits
all image bytes and previews, spectrum arrays and values, wavelengths,
intensities, paths/names, sample titles, free-form preflight text, stable
dataset/run IDs, credentials, and local files. LLM output is limited to an
ordered list of allowed step types; NEAT attaches trusted rationales and
evidence references locally. The LLM does not interpret unseen scientific
images or spectra and cannot establish phases, Bragg-edge locations, or
strain.

The planning stage issues at most **one model request**, with zero automatic
retries. For OpenAI image-stack runs, M5A-1 adds at most three bounded visual
requests, also without retries. The complete planning request body, including
schema and prompts,
is capped at **4,096 ASCII bytes**; this is also the conservative upper bound
of **4,096 input tokens**. Provider output is capped at **384 tokens** and
**4,096 response bytes**. The per-operation HTTP timeout is **10 seconds** and
the transport has a **15-second total deadline** that cancels the asynchronous
request inside the tracked worker. Local slow models may be rejected at these
bounds. The UI reports token use only if returned by the provider; no estimated
cost is shown. Cloud requests set OpenAI's `store=false`; this asks the provider
not to store the request but does not guarantee zero provider-side retention.
A stopped run requests transport cancellation and remains owned until its
tracked worker retires.

Visual explanations use separate bounds: `reason` permits 2,048 characters,
each `edge_reasons` entry permits 512, and short text entries permit 240.
The system instructions and response-schema descriptions communicate these
limits; local validation checks them after normalizing ordinary whitespace.
An over-limit explanation is rejected explicitly, never silently truncated.
The existing 12,000-byte response cap and three-request budget still apply.
Analysis details show the exact sanitized validation error and any allowlisted
phase proposed but not accepted. The fitting status distinguishes a response
validation failure from an uncertain scientific comparison. Rejected model
content is not echoed in diagnostics, and no extra model request is made.

Adjusting or rejecting a visual proposal withdraws its fit plan, outcome,
matrix cells, curves, numeric summary, and selectable spatial ROI overlay.
Dataset invalidation also clears the numeric summary. An invalid ROI edit
leaves the current evidence intact until a valid adjustment is previewed.
Spatial overlays project half-open original-pixel bounds outwards into
preview pixels before converting to inclusive drawing bounds. This keeps
valid subpixel ROIs drawable without changing their extraction coordinates;
rejected rectangles remain rejected even when clipped for display.

M4 planner graph runs use the same explicit LangSmith opt-in. The model
provider and LangSmith are separate network recipients. When tracing is
enabled, the structured planning request and response appear as an LLM run
under the planning node. Graph inputs, outputs, metadata, errors, and runtime
information are uploaded without client-side redaction; these may contain
dataset-derived values. Model failures or rejected output end the planning stage with a reason.
No deterministic offline plan is substituted for a failed LLM response.

## M4 manual smoke test (optional, no live key needed for CI)

1. Select **No LLM (analysis unavailable)** and press **Start Autonomous
   Analysis**. Confirm the Agent immediately reports that no LLM is configured
   and does not substitute an offline plan. The separate preflight action remains
   local-only.
2. In **Agent Model Settings**, configure a loopback local server and exact
   model ID. Confirm settings opening does not probe it; use **Test
   Connection…** only as an explicit, separate action. Then run a synthetic
   dataset and confirm the request remains local and structured.
3. For an optional cloud smoke, install `NEAT[agent-llm]`, save a
   user-supplied OpenAI key in the Agent keyring, choose an exact model ID, and
   inspect **Preview Data Sent**. Starting analysis in OpenAI mode sends the
   disclosed minimized summary and bounded visual evidence for image stacks.
   Use only data authorized for this transfer. LangSmith remains off unless
   separately enabled in Agent settings.
4. Do not treat the plan as scientific interpretation or execution. Confirm
   incomplete preflight stops LLM planning with a reason and no offline substitute,
   and verify no fitting, mapping, report generation, phase identification, or
   timed approval occurs.
