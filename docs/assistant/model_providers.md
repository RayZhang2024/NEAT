# Personal LLM providers and custom models

NEAT shared access remains fixed to the centrally configured OpenAI model and
the global daily allowance. The options in this document apply only to **Use my
own API key**.

## First-class suppliers and local models

NEAT provides curated model choices for:

- OpenAI
- Anthropic
- Google Gemini
- DeepSeek
- Kimi / Moonshot
- Local model through Ollama or LM Studio

Each supplier has a separate credential entry in the operating-system
credential manager. A key saved for OpenAI is never reused for DeepSeek, Kimi,
or a custom endpoint.

DeepSeek uses `https://api.deepseek.com`. The curated models are
`deepseek-v4-flash` and `deepseek-v4-pro`.

Kimi uses `https://api.moonshot.ai/v1`. The curated models are `kimi-k2.6` and
`kimi-k3`.

Provider model catalogues change frequently. Select **Enter model ID manually**
to use a model that is available to the user's account but not yet in NEAT's
curated list. The connection test makes one small, potentially billable API
request.

## Local model through Ollama or LM Studio

Local mode runs inference on the user's own computer and does not require an
API key. In **AI Assistant Settings**:

1. select **Local model (Ollama / LM Studio)**;
2. start Ollama or start the LM Studio local server;
3. select **Detect local models**;
4. choose one of the detected model IDs;
5. select **Test connection**, then save.

NEAT checks only the two standard loopback endpoints:

- Ollama: `http://localhost:11434/v1`
- LM Studio: `http://localhost:1234/v1`

Detection runs in the background so the NEAT interface remains responsive. If
the server is running but no model appears, first download a model in Ollama or
load a model in LM Studio, then detect again. A model ID may also be entered
manually.

Questions, retrieved NEAT documentation excerpts, conversation history and safe
screen context remain on the computer in local mode. The selected local server
must remain running while the assistant is used. Raw images and automatically
collected file paths are still excluded.

NEAT prefers local semantic retrieval. If its optional ONNX embedding runtime
or first-use embedding-model download is unavailable, it automatically falls
back to the bundled lexical retriever. Both modes search the same approved NEAT
documents; the fallback allows local-model support to remain usable on offline
or restricted machines.

For a computer with about 16 GB system RAM and a 6 GB laptop GPU, begin with a
quantised 4B to 8B instruct model. Larger models may run slowly or spill into
system memory. Answer quality and instruction following vary by model, so use
the existing NEAT pilot questions before recommending a local model to users.

## Advanced OpenAI-compatible endpoint

The advanced supplier accepts:

- an exact model ID;
- an API base URL;
- an independently stored API key.

Remote endpoints must use HTTPS. HTTP is accepted only for `localhost`,
`127.0.0.1`, or `::1`, allowing local services such as Ollama, LM Studio, and
vLLM. Local endpoints may omit the API key if their server does not require one.

NEAT rejects endpoint URLs containing embedded usernames, passwords, query
strings, or fragments. Before saving, confirm that the hostname in the privacy
notice belongs to the intended provider. Compatibility is not guaranteed:
providers may advertise an OpenAI-style API while imposing different parameter,
message, tool-call, or response constraints.

## Data sent in personal-key mode

NEAT sends the user's question, approved retrieved NEAT excerpts, limited
conversation history, and selected safe screen context directly to the chosen
provider. Raw images and automatically collected file paths are not sent.
Users remain responsible for the provider's terms, data location, institutional
approval, API charges, and model access permissions.
