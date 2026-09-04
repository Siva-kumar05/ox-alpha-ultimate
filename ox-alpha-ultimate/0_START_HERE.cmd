@echo off
REM 0_START_HERE.cmd - single double-click entry point.  No typing, no cd, no bash.
REM
REM Double-click this file in Explorer (the leading 0 sorts it to the top), then
REM press ONE number and Enter:
REM
REM   [1] Enter keys once            setup-live.cmd hidden prompts -> ~\.ox_secrets.env
REM   [2] Readiness check            start-live.cmd preflight -> expect 0 FAIL
REM   [3] Analysis run on live dhan  start-live.cmd dhan - week-1 boots into
REM                                  OBSERVATION: real history + training, no entries
REM                                  until a strategy shows real OOS edge on real data
REM   [4] Status                     start-live.cmd status
REM   [5] Open WINDOWS_QUICKSTART.md the step-by-step guide
REM   [0] Exit
REM
REM Accepts ZERO command-line arguments.  Tokens are never typed or pasted here -
REM they go only into the hidden prompts of menu item [1].  If you pasted a token
REM onto any command line, regenerate it in the Dhan web console and clear your
REM shell history.
setlocal EnableExtensions
cd /d "%~dp0"

if not "%~1"=="" (
  echo ERROR: 0_START_HERE.cmd accepts no arguments - double-click it in Explorer
  echo and press one menu digit.  If you pasted a token or key here: REGENERATE it
  echo in the Dhan web console and clear your shell history.  Tokens belong only
  echo in the hidden prompts of menu item [1], never on the command line, in files,
  echo or in chat.
  exit /b 2
)

:menu
cls
echo.
echo  ================================================================
echo    OX-ALPHA  live session menu
echo    repo: %~dp0
echo  ----------------------------------------------------------------
echo    [1] Enter keys once           hidden prompts - fresh daily Dhan token
echo    [2] Readiness check           preflight - expect 0 FAIL
echo    [3] Analysis run on live dhan week-1 OBSERVATION - fetches real history,
echo        trains on real data, places no entries until a strategy shows real
echo        OOS edge.  After market hours nothing can fill - the safe closed-
echo        market analysis run.  Stop anytime with Ctrl+C.
echo    [4] Status                    positions / strategies / health
echo    [5] Open WINDOWS_QUICKSTART.md
echo    [0] Exit
echo  ================================================================
echo.
set "MENU_CHOICE="
set /p "MENU_CHOICE=Press one number then Enter: "
if "%MENU_CHOICE%"=="1" goto opt_keys
if "%MENU_CHOICE%"=="2" goto opt_preflight
if "%MENU_CHOICE%"=="3" goto opt_run
if "%MENU_CHOICE%"=="4" goto opt_status
if "%MENU_CHOICE%"=="5" goto opt_guide
if "%MENU_CHOICE%"=="0" goto end
echo Invalid choice - press 0 to 5.
pause
goto menu

:opt_keys
echo.
echo  [1] Key entry via setup-live.cmd
echo  You will be prompted for three values with hidden input, then they are
echo  stored in %USERPROFILE%\.ox_secrets.env
echo  Generate the DHAN_TOKEN fresh in the Dhan app in this launch window first -
echo  yesterday's token is dead and the launcher now refuses it.
echo.
call "%~dp0setup-live.cmd"
echo.
pause
goto menu

:opt_preflight
echo.
echo  [2] Readiness check - this runs the real preflight, no keys needed.
echo  Expected: PREFLIGHT: 0 FAIL
echo.
call "%~dp0start-live.cmd" preflight
echo.
pause
goto menu

:opt_run
if exist "%USERPROFILE%\.ox_secrets.env" goto run_ok
echo.
echo  [3] No key file found at %USERPROFILE%\.ox_secrets.env
echo  Run menu item [1] first to enter your keys and a fresh daily Dhan token.
pause
goto menu
:run_ok
echo.
echo  [3] Starting the live Dhan session via start-live.cmd dhan
echo  Week-1 config boots into OBSERVATION mode with validated strategies 0 -
echo  that is expected and correct.  Let it fetch real history and train, then
echo  press Ctrl+C to stop the session when you are done looking.
echo.
call "%~dp0start-live.cmd" dhan
echo.
echo  Session ended - returning to the menu.
pause
goto menu

:opt_status
echo.
echo  [4] Status via start-live.cmd status
echo.
call "%~dp0start-live.cmd" status
echo.
pause
goto menu

:opt_guide
echo.
echo  Opening WINDOWS_QUICKSTART.md ...
start "" "%~dp0WINDOWS_QUICKSTART.md"
pause
goto menu

:end
echo Goodbye.
exit /b 0
