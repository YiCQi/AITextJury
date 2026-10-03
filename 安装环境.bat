@echo off
cd /d "%~dp0"
echo ==================================================
echo   ZeroAIBench  安装环境（首次使用只需运行一次）
echo   自动完成：检查 Python - 创建隔离环境 - 安装依赖
echo   不会改动你电脑上已有的 Python / conda 环境
echo ==================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo [!] 未找到 Python。请先安装 Python 3.10 或更高版本:
  echo     https://www.python.org/downloads/
  echo     安装时务必勾选 "Add Python to PATH"!
  echo.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1"
if errorlevel 1 (
  echo.
  echo [x] 安装失败。请把上方红色/错误信息截图，到项目主页提 Issue。
  pause
  exit /b 1
)
echo.
echo [OK] 安装完成！以后使用只需双击： 启动工作台.bat
echo      （想启用 Binoculars 等本地大模型检测器，见 README 的 -Ml 说明）
pause
