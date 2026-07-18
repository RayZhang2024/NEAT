# Batch 6 architecture and support review checklist

## How this batch should be reviewed

Batch 6 is primarily an engineering, product-governance and institutional
security review. It is not reasonable to expect one scientific developer to
approve every item.

Suggested ownership:

- **NEAT developer:** worker inventory/cancellation, settings versioning,
  packaging, smoke tests, knowledge-loader changes and automated evaluation.
- **NEAT product owner/support lead:** default assistant visibility, whether
  the assistant is a supported release feature, diagnostic-log behaviour and
  selection of user-safe help pages.
- **Facility IT/cybersecurity:** update-check network access, approved model
  provider, API-key provisioning, offline embedding-model distribution and
  clean-machine deployment.
- **Data protection/information governance:** typed sensitive information,
  local feedback retention/deletion/access and diagnostic-log contents.
- **Scientific/domain reviewers:** mandatory-human-review boundaries and final
  approval of user-facing technical guidance.

The existing technical documents provide code-derived facts. Those facts do
not constitute institutional approval.

## Code-derived current state

- GUI settings and custom phases are separate per-user JSON files with no
  schema version or migration mechanism. Read/write failures are suppressed.
- The assistant dock is created and shown by default, subject to the saved
  visibility setting.
- Shutdown uses an explicit worker list and cooperative stop flags.
  `OpenBeamLoadWorker` remains a known cancellation gap.
- Completion payloads are not yet standardized across every worker family.
- Assistant dependencies and approved knowledge are not fully bundled by the
  current standalone release workflow.
- The local semantic embedding model may download on first use.
- Update checks contact GitHub and are enabled by default.
- Only an allow-listed subset of screen context is sent to OpenAI, but text
  typed directly by a user is not generally sanitized.
- Optional feedback is retained locally in
  `%LOCALAPPDATA%/NEAT/assistant_feedback.jsonl`; no retention or deletion
  policy is defined.
- The default provider/model behaviour is configurable in code, but facility
  approval and a change/fallback policy are not defined.
- Technical pages are not automatically ingested. The assistant currently
  loads only its explicit curated knowledge files.

## Startup and state

- [ ] Confirm which settings should be per-user, per-instrument or per-project.
- [ ] Decide whether settings load/save failures should be visible to users.
- [ ] Add versioning and migration rules for both JSON settings files.
- [ ] Confirm the assistant should be created and visible by default.

## Workers and shutdown

- [ ] Inventory every transient worker attribute in `cleanup_resources`.
- [ ] Add cooperative stop support to `OpenBeamLoadWorker`.
- [ ] Define a common success/cancel/error completion result.
- [ ] Test closing during each loading, preprocessing, fitting and assistant job.
- [ ] Decide whether long shutdown waits need a progress/cancel dialog.

## Releases and errors

- [ ] Decide whether the AI assistant is part of the supported standalone app.
- [ ] If yes, install and bundle all assistant dependencies and approved knowledge.
- [ ] Decide how the local embedding model is provisioned for offline/restricted sites.
- [ ] Run a clean-machine packaged assistant smoke test.
- [ ] Add a persistent, privacy-reviewed diagnostic log or explicit export tool.
- [ ] Verify update checks comply with facility network/security policy.

## Assistant safety and privacy

- [ ] Review every allow-listed application-context field.
- [ ] Define guidance for sensitive text typed directly into questions.
- [ ] Review local feedback retention, deletion and access expectations.
- [ ] Confirm model/provider approval for facility use.
- [ ] Review the six routes and mandatory-human-review boundaries.
- [ ] Confirm the default model and fallback/change policy.

## Technical-document ingestion

- [ ] Complete scientific review of Batches 1-5.
- [x] Separate observed defects from behavior users should be taught.
  Curated assistant pages distinguish reviewed behaviour, user-facing warnings
  and known limitations.
- [x] Select user-safe pages rather than ingesting all code detail automatically.
  Five curated technical/limitations files are loaded; `docs/technical` remains
  excluded.
- [x] Mark selected pages `domain-reviewed` and `approved-for-rag`.
  Curated pages carry an explicit reviewed/approved statement.
- [x] Extend the explicit knowledge loader and evaluation datasets.
  The loader now includes the five curated files and the retrieval set contains
  50 questions.
- [ ] Re-run retrieval, grounded-answer and pilot feedback evaluations.
  Automated retrieval completed: lexical and semantic Hit@3 are both 100%.
  Routing accuracy and escalation recall are both 100%; all automated tests
  pass. Billable live-answer evaluation and a new user pilot remain pending.
