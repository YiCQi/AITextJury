@echo off
cd /d "%~dp0"
title ZeroAIBench - 安装本地大模型检测器
echo ==================================================
echo   可选增强包：本地大模型检测器（装完约多占 2-3 GB�?echo   解锁 3 个检测器�?echo     LLM-Perplexity / Fast-DetectGPT / Binoculars
echo   （HF Classifier 属于自带模型，不在此列）
echo   只需运行一次；装完自动做一次示例校准�?echo ==================================================
echo.
if not exist "apps\api\.venv\Scripts\python.exe" (
  echo [!] 基础环境还没安装。请先双�?"安装环境.bat"�?  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1" -Ml
if errorlevel 1 (
  echo [x] 安装失败。请把上方错误信息截图，到项目主页提 Issue�?  pause
  exit /b 1
)
echo.
echo 正在下载模型并做首次校准（首次约 3-10 分钟，请勿关窗）...
cd apps\api
.venv\Scripts\python.exe -m zeroaibench.cli calibrate --dataset demo
echo.
echo [OK] 全部完成！检测器会在下次分析时自动启用�?echo      以后使用照旧：双�?"启动工作�?bat"
pause
