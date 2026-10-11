# AI Agent window: one-click preflight and planning (M3)

## User-visible behaviour

The independent, non-modal AI Agent window keeps its Overview and Visual
Evidence panels side by side and does not replace the AI Assistant dock.
Opening the window is side-effect-free. **Start Autonomous Analysis** creates
one parent run and automatically advances through the existing local Step 0
preflight and the offline deterministic LangGraph plan. No second preview or
continue action is required. **Start Preflight** remains a separate diagnostic
action. The window permits only one active stage at a time; Stop applies to
whichever stage is running.

M3's terminal success message is **“Plan prepared — scientific execution is
not yet implemented.”** The plan is a read-only proposal. M3 performs no LLM
reasoning, edge fitting, mapping, phase identification, physical
interpretation, report generation, or source-data changes. A missing optional
dependency, invalid dataset, insufficient evidence, stale revision,
cancellation, and internal failure each have explicit non-success outcomes.

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

LangGraph is loaded lazily from the separate `NEAT[agent]` extra. A baseline
NEAT install can import the domain/service and open the Agent window without
LangGraph. Its diagnostic preflight remains available. If the user starts the
one-click run without the extra, preflight still completes and the window says
that planning requires `[agent]`; it never reports a successful plan. The
graph has no provider adapters or API-key access. Its invocation disables
LangSmith tracing even if tracing was enabled globally for another process
task. No graph state is persisted during ordinary runs. Tests may inject an
in-memory checkpointer; SQLite durability and restart/resume are future work.

## Worker, revision, and shutdown rules

The Qt owner thread captures the scientific dataset handle and selected ROI
for Step 0. The existing preflight worker releases its handle before
completion. Only then is a safe planning context constructed and passed to a
second tracked worker. The planner checks a cooperative stop event and the
thread-safe dataset revision clock at graph-node boundaries and before
publishing its result. It has a four-node fixed graph and no retry or
refinement loop.

Starting again while either worker is active is ignored and the controls stay
disabled. Starting diagnostic preflight during an autonomous run is also
blocked. Closing the Agent window requests a stop, discards current displayed
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

M3 neither grants permissions nor implements the ten-second timer,
Pause/Resume, approval UI, durable checkpoints, autonomous execution, or live
model reasoning. Those require later reviewed milestones. Scientific review
of `d_fit` / CSV factor-of-two / HKL-denominator conventions remains a separate
gate before fitting or physical claims can be automated.

## Manual smoke test

1. On a normal install without `[agent]`, open **AI Agent** and confirm the
   window is side by side, the AI Assistant remains available, and opening the
   window starts no analysis. Run **Start Preflight** and confirm it still
   shows observed local evidence.
2. Install `NEAT[agent]`, load an image stack, select a valid ROI, and press
   **Start Autonomous Analysis once**. Confirm the status advances from
   preflight to planning without another click; the plan shows only
   preflight-supported, read-only proposals and evidence IDs. It ends with the
   explicit execution-unavailable message.
3. Repeat with no ROI, invalid wavelength axis, empty data, or a spectrum-only
   import. Confirm the plan says needs-input or blocked and does not invent an
   image, spectrum, phase, or fitted value.
4. Stop during either stage, reload or clear the dataset, and close/reopen the
   Agent window. Confirm stale evidence is discarded, Stop drains safely, the
   window can be reused, and NEAT shutdown waits for worker retirement.
5. Confirm there is no model/provider/network request, fitting or mapping
   action, ten-second countdown, fake approval control, or automatic source
   data write.
