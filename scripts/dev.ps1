#requires -Version 5.1
<#
.SYNOPSIS
  Start the ZeroAIBench workbench (backend + web UI) and open it in the browser.

.DESCRIPTION
  Opens two PowerShell windows - one for the API (:8000), one for the Vite dev
  server (:5173) - so you can read each log and stop each side independently
  (close the window = stop the process). Then opens http://localhost:5173.

  Prerequisite: scripts\setup.ps1 has been run at least once.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $repo 'apps\api\.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    throw "venv not found. Run scripts\setup.ps1 first."
}
if (-not (Test-Path (Join-Path $repo 'apps\web\node_modules'))) {
    Write-Warning "apps\web\node_modules missing - running setup for the web UI ..."
    Push-Location (Join-Path $repo 'apps\web')
    npm install --no-audit --no-fund
    Pop-Location
    if ($LASTEXITCODE -ne 0) { throw "npm install failed; see setup.ps1 output for registry hints." }
}

Write-Host "Starting backend  -> http://localhost:8000  (window 1)" -ForegroundColor Cyan
Start-Process powershell.exe -WorkingDirectory (Join-Path $repo 'apps\api') `
    -ArgumentList '-NoExit', '-Command', '.venv\Scripts\python.exe -m zeroaibench'

Write-Host "Starting web UI   -> http://localhost:5173  (window 2)" -ForegroundColor Cyan
Start-Process powershell.exe -WorkingDirectory (Join-Path $repo 'apps\web') `
    -ArgumentList '-NoExit', '-Command', 'npm run dev'

Write-Host "Waiting for Vite to boot ..." -ForegroundColor DarkCyan
Start-Sleep -Seconds 6
Start-Process 'http://localhost:5173'
Write-Host "UI opening in your browser. Stop either side by closing its window." -ForegroundColor Green
