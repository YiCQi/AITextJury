@echo off
cd /d "%~dp0"
title ZeroAIBench Workbench
if not exist "apps\api\.venv\Scripts\python.exe" (
  echo [!] Environment not installed yet - installing now, first run only...
  echo.
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1"
  if errorlevel 1 (
    echo [x] Setup failed. Run INSTALL-ENVIRONMENT.bat to see the error.
    pause
    exit /b 1
  )
)
echo.
echo   ZeroAIBench is starting... your browser will open
echo   at http://localhost:8000
echo   ------------------------------------------------------------
echo   * Keep this window open  = workbench is running
echo   * Close this window      = stop the workbench
echo   * Next time              = just double-click this file again
echo   ------------------------------------------------------------
start "" cmd /c "timeout /t 7 >nul & start http://localhost:8000"
cd apps\api
.venv\Scripts\python.exe -m zeroaibench
pause
