# OX-Alpha Windows Quickstart

Everything below runs from PowerShell and needs **no `cd`** and **no bare
`bash`** (PowerShell's `bash` is the WSL relay and cannot run these scripts —
the `.cmd` wrappers locate real Git Bash at `C:\Program Files\Git\bin\bash.exe`).

Working folder: `C:\ox-alpha-src\ox-alpha-ultimate`. Replace it below if your
copy lives elsewhere.

---

## 1. Open a FRESH PowerShell (important)

Start menu → "PowerShell" → open it. Do **not** open it from an old window
(`powershell` typed inside a stale session inherits that session's environment,
including an outdated `DHAN_STATIC_IP`, which makes preflight fail egress).

Optional: activate the venv for plain-`python` commands
(`cd C:\ox-alpha-src\ox-alpha-ultimate`, then `.venv\Scripts\Activate.ps1`).
The `.cmd` launchers do not require it.

## 2. One-time key entry (hidden prompts)

```powershell
C:\ox-alpha-src\ox-alpha-ultimate\setup-live.cmd
```

- Accepts **no arguments** — keys are entered at its hidden prompts.
- Enter: Dhan client ID, **today's fresh Dhan access token** (Dhan web console →
  My Profile → DhanHQ Trading APIs → Generate Access Token), your machine
  public IP `49.43.232.235`, and the audit key (leave blank to auto-generate).
- Writes `~/.ox_secrets.env` (permissions 600). Re-run it any morning the token
  changes.

## 3. Readiness check

```powershell
C:\ox-alpha-src\ox-alpha-ultimate\start-live.cmd preflight
```

Expect `PREFLIGHT: 0 FAIL`. If it FAILs on egress IP, you are in a stale
session — close the window and open a fresh one from the Start menu.

## 4. Closed-market data-and-pattern run (market shut = no fills)

```powershell
C:\ox-alpha-src\ox-alpha-ultimate\start-live.cmd dhan
```

What the boot output should show:

- `Secure boot complete; autonomous execution=True, strategies=0`
- Health `OBSERVATION` (`validated_strategies=0`) — expected in week 1;
  entries only appear after strategies earn real OOS edge.
- Real history fetch for the configured symbols, then `auto_train_on_boot`
  training over real candles (it re-runs on the first tick after 18:00 IST).
- No trades can fill while the market is closed. Leave it running to collect
  results, or stop with **Ctrl+C** (graceful shutdown).

## 5. Inspect what training found

From the repo folder (needs the folder for the default config), or anywhere by
passing the config path:

```powershell
cd C:\ox-alpha-src\ox-alpha-ultimate
python scripts/analyze_patterns.py
```

(Equivalent from anywhere:
`python C:\ox-alpha-src\ox-alpha-ultimate\scripts\analyze_patterns.py C:\ox-alpha-src\ox-alpha-ultimate\config.yaml`)

Read-only and offline: per-candidate pooled OOS trades / ret / pf / score /
rejection cause, grouped by status, plus the gate-breakdown summary. If it says
"no training results", run step 4 once first.

## 6. Status / stop

```powershell
C:\ox-alpha-src\ox-alpha-ultimate\start-live.cmd status
```

Stop the running session with **Ctrl+C** in its window. Never run two live
sessions at once.

---

## Incident / hygiene rules

- **Tokens never go on the command line, into files, or into chat** — only into
  the hidden prompts of `setup-live.cmd` (one-time) or `start-daily.cmd`'s
  masked daily-token prompt. Every launcher refuses credential-like arguments.
- **If a token was ever pasted anywhere exposed: regenerate it** in the Dhan web
  console and clear your PowerShell history
  (`Clear-History`; delete `$env:APPDATA\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`).
- **Fresh token each daily launch.** Daily Dhan access tokens expire (~24 h);
  re-run `setup-live.cmd` with a new token, or use `start-daily.cmd`, which
  prompts masked and keeps the token session-only.
- **KILL.flag semantics.** A `KILL.flag` appears only after an autonomous halt
  (e.g. a daily-loss-cap breach or an unreconciled broker state). The agent then
  refuses to boot until an operator reconciles the broker book, flattens any
  untracked position manually, and deletes the flag — never delete it without
  that reconciliation, and never restart while uncertain capital is open.
