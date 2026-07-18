# Host the NEAT assistant on a Windows laptop

This is the no-hosting-cost arrangement used by NEAT public shared access.
The laptop runs the existing single-instance FastAPI service and retains the
atomic 20-request UTC daily allowance in its local SQLite database.

## Availability and security

- The laptop must remain powered on, awake, and connected to the internet.
- The assistant is unavailable while Windows is restarting or the laptop is off.
- Keep the server bound to `127.0.0.1`; do not forward port 8765 on the router.
- Tailscale Funnel provides the public HTTPS endpoint.
- The OpenAI key stays in Windows Credential Manager on the host laptop.
- The separate NEAT public-client token is included in official downloads and
  must be treated as extractable. It protects the endpoint from incidental
  traffic but does not identify or authenticate an individual user.
- The server-side global daily quota and the OpenAI project budget are the cost
  controls; one public user can consume the shared allowance before others.
- Check institutional security policy before offering the service publicly.

## 1. Install and sign in to Tailscale

Install the official Tailscale Windows client and sign in using the account that
will own the laptop's public Funnel hostname.

Confirm installation in PowerShell:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" status
```

## 2. Install the NEAT server dependencies

From the NEAT repository and its virtual environment:

```powershell
python -m pip install -e ".[assistant,assistant-server]"
```

## 3. Save the server secrets securely

```powershell
python -m tools.configure_assistant_laptop
```

Enter the server OpenAI API key when prompted. Input is hidden. The command
creates a separate random NEAT access token and saves both secrets in Windows
Credential Manager. Copy the displayed access token to a password manager; it
is displayed only so it can be configured on approved client computers.

Check that both credentials exist without displaying them:

```powershell
python -m tools.configure_assistant_laptop --status
```

## 4. Start and test the local server

```powershell
python -m tools.assistant_shared_server
```

In a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

The result should report `status` as `ok` and `daily_limit` as `20`.

After the local health check succeeds, register the server to start after you
sign in to Windows:

```powershell
powershell -ExecutionPolicy Bypass -File tools\register_assistant_startup.ps1
```

The task runs under your Windows account so it can read the credentials saved
in your Credential Manager. It does not contain either secret. The task waits
30 seconds after sign-in before launching the server and restarts it after an
unexpected exit.

After restarting Windows, allow approximately one minute and check:

```powershell
Get-ScheduledTask -TaskName "NEAT Shared Assistant Server"
Invoke-RestMethod http://127.0.0.1:8765/health
```

The task should be `Running` and the health response should be `ok`. A
persistent supervisor checks the local health endpoint every 10 seconds and
relaunches the server after a failure. Startup diagnostics are stored locally
in:

```text
%LOCALAPPDATA%\NEAT\logs\assistant_shared_server.log
%LOCALAPPDATA%\NEAT\logs\assistant_shared_server.stderr.log
```

An HTTP 502 from the public address normally means Tailscale is reachable but
the local server is not listening on port 8765. Check the scheduled task,
local health endpoint and these logs in that order.

## 5. Publish only the local service through HTTPS

Run Funnel in the background:

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" funnel --bg 8765
& "$env:ProgramFiles\Tailscale\tailscale.exe" funnel status
```

The status output provides a stable address similar to:

```text
https://your-laptop.your-tailnet.ts.net
```

Do not include `/health` in the client service URL.

## 6. Configure official public downloads

Official release packages receive the Funnel address and separate public-client
token during the GitHub Actions build. Add the server's service token as the
GitHub repository Actions secret `NEAT_SHARED_PUBLIC_ACCESS_TOKEN`. The release
workflow supplies the public URL and prepares the bundled configuration before
PyInstaller runs. Never add the OpenAI API key to the workflow or a client.

Before a local candidate build, prepare the same extractable configuration from
the current computer's `NEAT_SHARED_SERVICE_URL` and
`NEAT_SHARED_ACCESS_TOKEN`, then build:

```powershell
python -m tools.prepare_public_shared_access
pyinstaller --noconfirm --clean NEAT.spec
```

Users of an official download can select **Use NEAT shared access** immediately;
they do not configure an endpoint or token. Complete per-computer environment
settings still override the bundled configuration for development and testing.

## Stop public access

```powershell
& "$env:ProgramFiles\Tailscale\tailscale.exe" funnel reset
```

This removes the public route without deleting the OpenAI key or quota data.

To remove automatic server startup:

```powershell
powershell -ExecutionPolicy Bypass -File tools\register_assistant_startup.ps1 -Remove
```

To remove the locally saved server credentials:

```powershell
python -m tools.configure_assistant_laptop --remove
```

## Backup

Back up `.assistant_server/quota.sqlite3` if retaining quota history matters.
The Chroma index in `.assistant_server/chroma` can be rebuilt from the approved
documentation and does not need to be backed up.
