@echo off
cd /d "%~dp0"
title ZeroAIBench - Create desktop shortcut
echo Creating a "ZeroAIBench" shortcut on your desktop...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut([IO.Path]::Combine($ws.SpecialFolders['Desktop'], 'ZeroAIBench.lnk'));" ^
  "$lnk.TargetPath = '%~dp0START-WORKBENCH.bat';" ^
  "$lnk.WorkingDirectory = '%~dp0';" ^
  "$lnk.Description = 'ZeroAIBench - open AI-text detection workbench';" ^
  "$lnk.Save()"
if errorlevel 1 (
  echo [x] Failed - the desktop may not be writable.
) else (
  echo [OK] Shortcut created! Double-click it on your desktop from now on.
)
echo.
echo Tip: uninstalling ZeroAIBench = delete the folder. Nothing is left behind.
pause
