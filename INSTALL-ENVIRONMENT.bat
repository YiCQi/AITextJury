@echo off
cd /d "%~dp0"
title ZeroAIBench - Install environment
echo ==================================================
echo   ZeroAIBench - First-time setup (run once)
echo   Creates a private environment and installs deps.
echo   Your existing Python/conda is NOT touched.
echo ==================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo [!] Python not found. Install Python 3.10+ from
  echo     https://www.python.org/downloads/
  echo     and tick "Add Python to PATH" during install.
  echo.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1"
if errorlevel 1 (
  echo.
  echo [x] Setup failed. Please open an issue with the error text above.
  pause
  exit /b 1
)
echo.
echo [OK] Setup complete! From now on, double-click: START-WORKBENCH.bat
echo      Optional: INSTALL-LM-DETECTORS.bat enables 3 local-LM detectors.
pause
