@echo off
cd /d "%~dp0"
title ZeroAIBench - Install local-LM detectors
echo ==================================================
echo   Optional add-on: local-LM detectors (~2-3 GB)
echo   Unlocks: LLM-Perplexity / Fast-DetectGPT / Binoculars
echo   Run once; a demo calibration follows automatically.
echo ==================================================
echo.
if not exist "apps\api\.venv\Scripts\python.exe" (
  echo [!] Base environment missing. Run INSTALL-ENVIRONMENT.bat first.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1" -Ml
if errorlevel 1 (
  echo [x] Install failed. Please open an issue with the error text above.
  pause
  exit /b 1
)
echo.
echo Downloading models and running first calibration
echo (3-10 minutes on first run - keep this window open)...
cd apps\api
.venv\Scripts\python.exe -m zeroaibench.cli calibrate --dataset demo
echo.
echo [OK] Done! The detectors activate automatically on the next analysis.
echo      Usage unchanged: double-click START-WORKBENCH.bat
pause
