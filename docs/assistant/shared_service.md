# NEAT shared AI service

The shared service lets NEAT users ask support questions through one centrally
funded OpenAI project without distributing its OpenAI API key.

## Request policy

- The allowance is 20 logical requests across all users per UTC day.
- Invalid and unauthorised requests do not consume the allowance.
- A validated request consumes one allowance before the OpenAI call starts.
- A provider failure still consumes that allowance because a billable upstream
  attempt may have occurred.
- Retrying the same completed request ID returns its cached response without
  consuming another allowance.
- Personal API-key mode does not use the shared allowance.

## Security boundary

The hosted server owns `OPENAI_API_KEY`. Desktop clients never receive it.
The server accepts a structured NEAT question, safe scalar UI context, and
limited conversation history. It performs approved-document retrieval itself,
so it is not a general-purpose OpenAI proxy.

The initial implementation uses a limited service access token. This is
appropriate for a controlled pilot. A public deployment should replace the
shared token with institutional sign-in or individual short-lived user tokens.

Remote desktop clients require HTTPS. Plain HTTP is accepted only for
`localhost` and `127.0.0.1` development.

## Install the server

From the NEAT repository:

```powershell
python -m pip install -e ".[assistant,assistant-server]"
```

Create a long random service token:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Configure the server process:

```powershell
$env:OPENAI_API_KEY = "server OpenAI key"
$env:NEAT_SHARED_SERVICE_TOKEN = "generated limited service token"
$env:NEAT_SHARED_OPENAI_MODEL = "gpt-5.6-luna"
$env:NEAT_SHARED_DAILY_LIMIT = "20"
```

Run a local instance:

```powershell
python -m tools.assistant_shared_server
```

The default address is `http://127.0.0.1:8765`. The health endpoint is:

```text
GET /health
```

Production hosting should terminate HTTPS in front of the service and keep the
OpenAI key and service token in the hosting platform's secret manager.

## Configure a desktop client

For a local pilot:

```powershell
$env:NEAT_SHARED_SERVICE_URL = "http://127.0.0.1:8765"
$env:NEAT_SHARED_ACCESS_TOKEN = "generated limited service token"
python -m NEAT.app
```

The AI Settings window will then enable **Use NEAT shared access**.

For a remote service, use its HTTPS URL. Never place `OPENAI_API_KEY` on client
computers.

## Quota storage

The initial quota backend uses an atomic SQLite transaction and is safe for one
server process. Its local state is stored under `.assistant_server/` by default
and is excluded from Git.

For a no-hosting-cost controlled pilot, the service can run on a Windows laptop
and be exposed through Tailscale Funnel without router port forwarding. Follow
the [laptop hosting guide](laptop_hosting.md).

If the service is later run as multiple containers or server instances, replace
SQLite with a shared Redis or PostgreSQL quota backend before scaling.
