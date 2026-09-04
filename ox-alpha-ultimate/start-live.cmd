@echo off
REM start-live.cmd - Windows launcher for scripts/live.sh via REAL Git Bash.
REM
REM Usage:  start-live.cmd <verb>
REM   verbs: dhan  choice  binance  live-test  verify-all  preflight  track
REM          status  paper  promax-smoke
REM
REM PowerShell's built-in bash often resolves to the WSL relay, which cannot
REM run these scripts.  This wrapper locates Git for Windows instead and
REM fails with a clear message when it is absent.
REM
REM Credentials are NEVER accepted as arguments.  For the masked daily-token
REM flow use start-daily.cmd; one-time keys go into setup-live.sh prompts.
setlocal EnableExtensions
cd /d "%~dp0"

set "GITBASH="
for %%G in ("%ProgramFiles%\Git\bin\bash.exe" "%ProgramFiles(x86)%\Git\bin\bash.exe" "%LOCALAPPDATA%\Programs\Git\bin\bash.exe") do (
  if exist "%%~fG" set "GITBASH=%%~fG"
)
if not defined GITBASH (
  echo ERROR: Git Bash not found.
  echo Open "Git Bash" from the Start menu instead, or install Git for Windows
  echo from https://git-scm.com/download/win
  echo Note: Windows' built-in bash resolves to the WSL relay and cannot run
  echo scripts/live.sh - do not use it.
  exit /b 3
)

if "%~1"=="" (
  echo Usage: start-live.cmd ^<verb^>
  echo   verbs: dhan  choice  binance  live-test  verify-all  preflight  track
  echo          status  paper  promax-smoke
  echo Credentials are never accepted as arguments.  One-time keys go into the
  echo hidden prompts of setup-live.sh; the daily token goes into start-daily.cmd.
  exit /b 2
)
if not "%~2"=="" (
  echo ERROR: start-live.cmd accepts at most one verb - extra arguments refused.
  echo If you pasted a token here: REGENERATE it in the Dhan web console and clear
  echo your shell history.  Tokens belong only in the hidden prompts of setup-live.
  exit /b 2
)

set "VERB=%~1"
set "ALLOWED= dhan choice binance live-test verify-all preflight track status paper promax-smoke "
echo %ALLOWED% | findstr /i /c:" %VERB% " >nul
if errorlevel 1 (
  echo ERROR: unknown or credential-like verb: %VERB%
  echo Tokens never belong on the command line.  Regenerate anything you pasted
  echo and enter keys only via setup-live.sh prompts or start-daily.cmd.
  exit /b 2
)

echo Starting live.sh %VERB% via: %GITBASH%
"%GITBASH%" scripts/live.sh %VERB%
exit /b %ERRORLEVEL%
