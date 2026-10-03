@echo off
cd /d "%~dp0"
title ZeroAIBench 工作台
if not exist "apps\api\.venv\Scripts\python.exe" (
  echo [!] 还没有安装环境，先自动安装（约 1-3 分钟，只需这一次）...
  echo.
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1"
  if errorlevel 1 (
    echo [x] 安装失败。请先双击 "安装环境.bat" 查看具体错误。
    pause
    exit /b 1
  )
)
echo.
echo   ZeroAIBench 正在启动... 浏览器将自动打开 http://localhost:8000
echo   ------------------------------------------------------------
echo   * 保持本窗口开着 = 工作台运行中
echo   * 关闭本窗口    = 停止工作台
echo   * 下次使用       = 直接双击本文件即可
echo   ------------------------------------------------------------
start "" cmd /c "timeout /t 7 >nul & start http://localhost:8000"
cd apps\api
.venv\Scripts\python.exe -m zeroaibench
pause
