---
title: AI assistant retrieval, privacy, feedback and evaluation
doc_id: neat-tech-support-assistant-architecture
doc_type: technical_reference
functional_area: assistant
audience: [user, developer, support, scientist]
neat_version: 4.8.0
verified_commit: 628c767ef44186e4301454f24a54fbc05ad71233
status: code-verified
instrument_applicability: [general]
scientific_review: pending
source_paths: [NEAT/ui/assistant_panel.py, tools/assistant_retrieval.py, tools/assistant_semantic_retrieval.py, tools/assistant_router.py, tools/assistant_query_pipeline.py, tools/assistant_answering.py, tools/assistant_service.py, tools/assistant_openai.py, tools/assistant_feedback.py, tools/assistant_feedback_summary.py, tools/assistant_answer_evaluation.py]
source_symbols: [AssistantDockWidget, collect_neat_context, ChromaSemanticRetriever, QuestionRouter, RoutedRetriever, GroundedPromptBuilder, NEATAssistantService, record_assistant_feedback]
test_paths: [tests/test_assistant_retrieval.py, tests/test_assistant_semantic_retrieval.py, tests/test_assistant_router.py, tests/test_assistant_answering.py, tests/test_assistant_openai.py, tests/test_assistant_panel.py, tests/test_assistant_feedback.py, tests/test_assistant_feedback_summary.py, tests/test_assistant_answer_evaluation.py]
---

# AI assistant retrieval, privacy, feedback and evaluation

## Purpose and boundary

The docked assistant answers questions about using NEAT. It is a support layer,
not part of the scientific calculation engine. It does not run fitting,
validate a scientific conclusion, edit settings or inspect raw image data.

The request path is:

```text
question + allow-listed screen context
  -> deterministic safety/intent route
  -> local semantic retrieval from approved Markdown
  -> grounded prompt with recent chat turns
  -> OpenAI chat model
  -> answer + application-owned citations/review flag
```

Retrieval and the model request run in a `QThread`. Only one request can be
active per panel.

## Approved knowledge boundary

`load_knowledge_base` loads exactly:

- `docs/assistant/faq.md`;
- `docs/assistant/troubleshooting.md`;
- `docs/assistant/parameter_reference.md`;
- `docs/assistant/preprocessing_technical.md`;
- `docs/assistant/fitting_technical.md`;
- `docs/assistant/mapping_technical.md`;
- `docs/assistant/postprocessing_technical.md`; and
- `docs/assistant/known_limitations.md`.

Markdown is divided into level-two and level-three heading sections. Each
section has a deterministic filename-and-anchor source ID. The larger
`docs/technical` collection remains excluded by code. Its reviewed,
user-relevant material is represented by the curated technical files above;
creating or changing an internal technical page does not automatically expose
it to users.

The production GUI uses local LangChain Chroma semantic retrieval and the
Chroma default ONNX `all-MiniLM-L6-v2` embedding. The index is stored under
`.assistant_cache/chroma` and is fingerprinted/rebuilt when approved knowledge
changes. A dependency-free BM25 retriever remains available as an evaluation
baseline.

## Routing and grounded generation

Before retrieval, deterministic rules select:

- how-to;
- parameter explanation;
- troubleshooting;
- scientific interpretation;
- bug report; or
- mandatory escalation.

Scientific-interpretation and escalation routes set
`requires_human_review=True`. Escalation also pins the approved escalation
section into the retrieved results.

The answer prompt receives route-specific policy, approved source blocks,
allow-listed application context and at most six recent user/assistant turns.
It instructs the model not to invent NEAT behavior and to state when sources
are insufficient.

Citations are constructed from retrieved metadata after generation. The model
does not control source identifiers, route or review status.

## Data sent to OpenAI

The allow-listed application context contains:

- NEAT version and active top-level module; and, only on the fitting tab,
- selected phase;
- wavelength minimum/maximum;
- macro-pixel width/height;
- fixed state of `s`, `t` and `eta`; and
- selected theoretical edge d-spacing when numeric.

The request also sends the typed question, retrieved approved passages and up
to 12 stored UI history messages (six conversation turns). It does not collect
raw images or file paths. Chat history exists only in memory and Clear removes
it.

This boundary does not sanitize arbitrary sensitive information that a user
types into the question itself. Users must not paste confidential experiment
details, credentials or raw data into the chat.

## API configuration

`OPENAI_API_KEY` must exist in the NEAT process environment. The default model
is `gpt-5.6-luna`; `NEAT_ASSISTANT_MODEL` overrides it. Calls use the Responses
API with low reasoning effort, a 60-second timeout and up to two retries.

The local embedding engine may download its model on first use. Therefore the
assistant is not fully offline even before considering the OpenAI request.

## Local optional feedback

Helpful/Not helpful feedback is appended as JSON Lines at:

```text
%LOCALAPPDATA%/NEAT/assistant_feedback.jsonl
```

Each record contains the rating, question, citations, route, review flag, NEAT
version, timestamp and random feedback ID. It excludes the generated answer
and application context. Windows-style paths typed in the question are
replaced and questions are limited to 2000 characters.

Other sensitive text typed by the user is not generally redacted. Feedback is
local and optional, but its path is displayed after saving.

The summary tool reports overall and latest-per-question ratings, unresolved
not-helpful questions, route coverage and commonly cited sources. It makes no
OpenAI request.

## Evaluation

Separate controlled datasets test retrieval, routing and live answers:

- retrieval: Hit@1, Hit@3 and mean reciprocal rank;
- routing: expected route and human-review flag;
- answer automation: expected-source citation and escalation alignment;
- human answer review: correctness, groundedness, clarity and scientific
  scope.

Live evaluation requires an explicit `--confirm-live` acknowledgement because
it makes billable API calls. Reports save after every question and bind to both
the model name and knowledge fingerprint before resuming.

Lexical key-point matching is a heuristic, not scientific validation. Human
review remains required before knowledge changes are approved.

## Approval path for technical documents

The new code-derived technical pages should not be added wholesale. For each
functional block:

1. resolve its domain-review checklist;
2. correct any code-derived behavior that should not be taught as intended;
3. mark selected pages `domain-reviewed`;
4. choose user-safe sections and mark them `approved-for-rag`;
5. extend the explicit knowledge-file loader or generate curated assistant
   material from those approved sections;
6. add expected-source retrieval questions; and
7. rerun retrieval, routing, answer and real-use feedback evaluation.

## Retrieval questions

- What information does the NEAT assistant send to OpenAI?
- Which documents can the assistant currently retrieve?
- Are citations generated by the language model?
- Where is assistant feedback stored and what does it contain?
- How are technical documents approved before assistant use?
