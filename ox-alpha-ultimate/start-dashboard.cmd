@echo off
cd /d "%~dp0"
start "ox-alpha dashboard" /B ".venv\Scripts\python.exe" -m streamlit run dashboard.py
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8501"
