# ============================================================================
# START_DAILY.ps1 - daily-token launcher for ox-alpha (recommended flow)
#
# Security model: NO TOTP secret, NO PIN, and NO token is ever stored on this
# machine - not in files, not in the registry, not in PowerShell history.
# You paste a fresh 24-hour Dhan access token each morning; it lives only in
# this console session's environment and is scrubbed when the agent exits.
#
# Generate today's token: dhan.co web -> My Profile -> DhanHQ Trading APIs
# -> Generate Access Token (the site will ask your PIN/TOTP - that happens
# in the BROWSER, so the secret never touches this PC).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File START_DAILY.ps1
#   (or just double-click start-daily.cmd)
# ============================================================================

$ErrorActionPreference = "Stop"

# ── 1. Client ID ────────────────────────────────────────────────────────────
$ClientId = $env:DHAN_CLIENT_ID
if (-not $ClientId) {
    $ClientId = Read-Host "Enter DHAN_CLIENT_ID (session-only, not saved)"
}
$ClientId = $ClientId.Trim()
if (-not $ClientId) { Write-Host "Client ID is required." -ForegroundColor Red; exit 1 }

# ── 2. Paste today's token (masked input) ───────────────────────────────────
Write-Host ""
Write-Host "Paste TODAY'S Dhan access token (input hidden; valid ~24h):" -ForegroundColor Cyan
$secure = Read-Host -AsSecureString "Access token"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $Token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
$Token = $Token.Trim().Trim('"').Trim("'")
if (-not $Token) { Write-Host "No token entered." -ForegroundColor Red; exit 1 }
if ($Token.ToUpper().StartsWith("DUMMY")) { Write-Host "Placeholder token rejected." -ForegroundColor Red; exit 1 }

# ── 3. Preflight: validate the token against a read-only endpoint ───────────
Write-Host "Validating token against Dhan /fundlimit ..." -ForegroundColor Cyan
$headers = @{
    "access-token" = $Token
    "client-id"    = $ClientId
    "Accept"       = "application/json"
}
try {
    $null = Invoke-RestMethod -Uri "https://api.dhan.co/v2/fundlimit" `
        -Headers $headers -TimeoutSec 10
    Write-Host "Token VALID - session confirmed." -ForegroundColor Green
}
catch {
    $detail = ""
    if ($_.Exception.Response) {
        $code = [int]$_.Exception.Response.StatusCode
        if ($code -eq 401) { $detail = " (401: token expired or invalid - regenerate it)" }
        elseif ($code -eq 403) { $detail = " (403: check static-IP registration for $env:DHAN_STATIC_IP)" }
        else { $detail = " (HTTP $code)" }
    }
    Write-Host "Token validation FAILED$detail - agent not started." -ForegroundColor Red
    exit 1
}

# ── 4. Session-only environment (Process scope: dies with this window) ──────
# Deliberately NOT [Environment]::SetEnvironmentVariable(..., "User"/"Machine")
$env:DHAN_CLIENT_ID    = $ClientId
$env:DHAN_TOKEN        = $Token
$env:DHAN_ACCESS_TOKEN = $Token

# ── 5. Static-IP sanity, then launch ────────────────────────────────────────
python verify_ip.py
if ($LASTEXITCODE -ne 0) { Write-Host "Static-IP check failed - agent not started." -ForegroundColor Red; exit 1 }

try {
    python run.py run
}
finally {
    # ── 6. Scrub secrets from the session on the way out ────────────────────
    Remove-Item Env:DHAN_TOKEN        -ErrorAction SilentlyContinue
    Remove-Item Env:DHAN_ACCESS_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:DHAN_CLIENT_ID    -ErrorAction SilentlyContinue
    Write-Host "Session secrets scrubbed." -ForegroundColor Green
}
