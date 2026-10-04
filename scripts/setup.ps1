#requires -Version 5.1
<#
.SYNOPSIS
  AITextJury one-command setup (Windows).

.DESCRIPTION
  Creates an isolated virtualenv under apps\api\.venv and installs the
  backend + (optionally) the native-LM detector stack. Your system / conda
  Python is never touched, so this cannot break any existing environment —
  and a broken global environment cannot break AITextJury.

  Idempotent: rerun any time; an existing venv is reused and deps refreshed.

.PARAMETER Ml
  Also install torch + transformers (~2 GB download) to enable the local-LM
  detectors (LLM-Perplexity, Fast-DetectGPT, Binoculars, HF classifier).
  You can add this later by rerunning with -Ml.

.EXAMPLE
  scripts\setup.ps1          # core stack, ~1 minute
  scripts\setup.ps1 -Ml      # + native detectors
#>
[CmdletBinding()]
param([switch]$Ml)

$ErrorActionPreference = 'Stop'
$repo  = Split-Path -Parent $PSScriptRoot
$api   = Join-Path $repo 'apps\api'
$venv  = Join-Path $api  '.venv'
$tuna  = 'https://pypi.tuna.tsinghua.edu.cn/simple'

# ------------------------------------------------------------- helpers
function Test-PythonCmd([string]$Exe, [string[]]$PyArgs) {
    # Returns "major.minor" if this python works, else $null.
    try {
        $out = & $Exe @PyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($LASTEXITCODE -eq 0 -and $out) { return "$out" }
    } catch { }
    return $null
}

function Resolve-Python {
    # NOTE: cannot name the param $args - that's a PowerShell automatic
    # variable (function-argument splat), which silently breaks splatting.
    foreach ($cand in @('python', 'py -3')) {
        $parts = @($cand -split ' ')
        $exe   = $parts[0]
        $rest  = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() }
        $ver   = Test-PythonCmd -Exe $exe -PyArgs $rest
        if ($ver) { return @{ Exe = $exe; Args = $rest; Ver = $ver } }
    }
    throw "Python 3.10+ not found. Install it from https://python.org (tick 'Add to PATH'), or skip Python entirely with: docker compose up"
}

function Pip-Install([string[]]$PipArgs) {
    $py = Join-Path $venv 'Scripts\python.exe'
    & $py -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">> pip on the default index failed; retrying once via the Tsinghua mirror ..." -ForegroundColor Yellow
        & $py -m pip @PipArgs -i $tuna
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed on both the default index and the Tsinghua mirror. Check your network / set HTTP_PROXY, then rerun."
        }
    }
}

# ------------------------------------------------------------- 1. python
$pyInfo = Resolve-Python
if ([version]$pyInfo.Ver -lt [version]'3.10') {
    throw "Python 3.10+ required, found $($pyInfo.Ver). Nothing was changed."
}
Write-Host "[1/4] Python $($pyInfo.Ver) found ($($pyInfo.Exe))." -ForegroundColor Green

# ------------------------------------------------------------- 2. venv
if (Test-Path (Join-Path $venv 'Scripts\python.exe')) {
    Write-Host "[2/4] venv already exists, reusing apps\api\.venv" -ForegroundColor Green
} else {
    Write-Host "[2/4] creating isolated venv at apps\api\.venv ..." -ForegroundColor Cyan
    & $pyInfo.Exe @($pyInfo.Args + @('-m', 'venv', $venv))
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed." }
}

# ------------------------------------------------------------- 3. backend
Write-Host "[3/4] installing backend dependencies ..." -ForegroundColor Cyan
Pip-Install @('install', '--upgrade', 'pip')
if ($Ml) {
    Pip-Install @('install', '-e', "${api}[ml]")
    Write-Host "     + torch/transformers (native-LM detectors enabled)" -ForegroundColor DarkCyan
} else {
    Pip-Install @('install','-e', $api)
}

# ------------------------------------------------------------- 4. web
Write-Host "[4/4] web UI ..." -ForegroundColor Cyan
$nodeOk = $false
try { node -v | Out-Null; if ($LASTEXITCODE -eq 0) { $nodeOk = $true } } catch { }
if (-not $nodeOk) {
    Write-Host "     Node.js not found - skipping web UI install." -ForegroundColor Yellow
    Write-Host "     The API + CLI still work (http://localhost:8000), but for the UI either" -ForegroundColor Yellow
    Write-Host "     install Node 18+ from https://nodejs.org or use: docker compose up" -ForegroundColor Yellow
} else {
    Push-Location (Join-Path $repo 'apps\web')
    try {
        npm install --no-audit --no-fund
    } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "     npm install failed. If your network blocks npmjs:" -ForegroundColor Yellow
        Write-Host '       npm config set registry https://registry.npmmirror.com   (then rerun)' -ForegroundColor Yellow
    } else {
        Write-Host "     web deps ready." -ForegroundColor Green
    }
}

# ------------------------------------------------------------- summary
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Run the workbench:   scripts\dev.ps1   (opens http://localhost:5173)" -ForegroundColor White
if (-not $Ml) {
    Write-Host "  Later, to enable Binoculars / Fast-DetectGPT / LM-Perplexity:" -ForegroundColor DarkGray
    Write-Host "    scripts\setup.ps1 -Ml     (~2 GB, first use also downloads GPT-2 models)" -ForegroundColor DarkGray
}
Write-Host "  Or one-window mode:  docker compose up" -ForegroundColor DarkGray
