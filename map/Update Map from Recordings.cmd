@echo off
rem Rebuilds data\recordings.js from the cloud GPU recording room (../recorder), then opens the map.
cd /d "%~dp0..\recorder"
".venv\Scripts\python.exe" -m avoidance.cli map --runs runs/gpu-chat --out ../map/data/recordings.js
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "%~dp0Open UMAP Prototype.cmd"
