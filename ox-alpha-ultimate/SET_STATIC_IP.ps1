# One-time host setup for the ox-alpha static IP (SEBI / Dhan requirement).
# The real IP is deliberately NOT stored in this file anymore: enter it at
# runtime or pass -Ip. It lives only in host environment variables and in
# your broker's dashboard - never in config.yaml or source control.

param([string]$Ip = "")
if (-not $Ip) { $Ip = Read-Host "Enter the static IP registered with your broker" }

[System.Environment]::SetEnvironmentVariable("DHAN_STATIC_IP", $Ip, "User")
$env:DHAN_STATIC_IP = $Ip
Write-Host "DHAN_STATIC_IP set to $env:DHAN_STATIC_IP (User + Process)"

try {
  $egress = (Invoke-RestMethod -Uri "https://api.ipify.org?format=json" -TimeoutSec 5).ip
  Write-Host "Current egress IP (as seen by the internet): $egress"
  if ($egress -eq $Ip) { Write-Host "MATCH - ready for broker registration" -ForegroundColor Green }
  else { Write-Host "MISMATCH - register the egress IP instead, or run this on the static-IP host." -ForegroundColor Yellow }
} catch { Write-Host "Egress check failed: $_" -ForegroundColor Yellow }

python verify_ip.py
