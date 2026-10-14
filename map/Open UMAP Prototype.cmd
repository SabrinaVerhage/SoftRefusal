@echo off
setlocal
title UMAP Prototype
pushd "%~dp0"
echo Opening the UMAP prototype...

if exist "..\recorder\.venv\Scripts\python.exe" (
    "..\recorder\.venv\Scripts\python.exe" "launch.py" %*
    goto result
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "launch.py" %*
    goto result
)
where python >nul 2>nul
if not errorlevel 1 (
    python "launch.py" %*
    goto result
)
echo Python could not be found. Install Python or use the Recording Room environment.
pause
popd
exit /b 1

:result
set "UMAP_LAUNCH_EXIT=%errorlevel%"
if not "%UMAP_LAUNCH_EXIT%"=="0" (
    echo.
    echo The prototype could not open. The explanation is above.
    pause
)
popd
exit /b %UMAP_LAUNCH_EXIT%
