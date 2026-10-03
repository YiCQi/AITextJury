@echo off
cd /d "%~dp0"
title ZeroAIBench - 创建桌面快捷方式
echo 正在桌面创建 "ZeroAIBench 工作�? 快捷方式...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut([IO.Path]::Combine($ws.SpecialFolders['Desktop'], 'ZeroAIBench 工作�?lnk'));" ^
  "$lnk.TargetPath = '%~dp0启动工作�?bat';" ^
  "$lnk.WorkingDirectory = '%~dp0';" ^
  "$lnk.Description = 'ZeroAIBench - open AI-text detection workbench';" ^
  "$lnk.Save()"
if errorlevel 1 (
  echo [x] 创建失败（可能没有桌面权限）�?) else (
  echo [OK] 已创建！以后从桌面双击即可启动工作台�?)
echo.
echo 提示：卸�?ZeroAIBench 非常简�?- 整个文件夹删除即可，不残留任何系统设置�?pause
