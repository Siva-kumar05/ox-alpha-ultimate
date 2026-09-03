@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "START_DAILY.ps1"
pause
