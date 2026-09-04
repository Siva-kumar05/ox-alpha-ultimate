@echo off
REM setup-live.cmd - Windows one-time key entry via REAL Git Bash.
REM
REM Runs the hidden-input prompts of scripts/setup-live.sh, which write
REM ~/.ox_secrets.env (chmod 600).  Works from ANY folder - this wrapper
REM changes to the repo itself.  PowerShell's built-in bash is the WSL
REM relay and cannot run these scripts; Git for Windows is located here.
REM
REM Credentials are entered ONLY at the script's hidden prompts - this
REM wrapper accepts ZERO arguments.  Re-run it any time a key changes,
REM and every morning to enter the fresh daily Dhan token.
setlocal EnableExtensions
cd /d "%~dp0"

if not "%~1"=="" (
  echo ERROR: setup-live.cmd accepts no arguments - keys are entered at the
  echo hidden prompts inside.  If you pasted a token or key here: REGENERATE it
  echo in the Dhan web console and clear your shell history.  Tokens belong only
  echo in the hidden prompts, never on the command line, in files, or in chat.
  exit /b 2
)

set "GITBASH="
for %%G in ("%ProgramFiles%\Git\bin\bash.exe" "%ProgramFiles(x86)%\Git\bin\bash.exe" "%LOCALAPPDATA%\Programs\Git\bin\bash.exe") do (
  if exist "%%~fG" set "GITBASH=%%~fG"
)
if not defined GITBASH (
  echo ERROR: Git Bash not found.
  echo Install Git for Windows from https://git-scm.com/download/win or open
  echo "Git Bash" from the Start menu and run:  bash scripts/setup-live.sh
  echo Note: Windows' built-in bash resolves to the WSL relay and cannot run
  echo these scripts - do not use it.
  exit /b 3
)

echo Entering live credentials via hidden prompts ...
"%GITBASH%" scripts/setup-live.sh
exit /b %ERRORLEVEL%
