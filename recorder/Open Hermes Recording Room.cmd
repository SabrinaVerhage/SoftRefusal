@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8768/
.venv\Scripts\python.exe -m avoidance.cli chat --config configs/hermes4-14b-remote.json --out runs/hermes-chat --port 8768
pause
