#!/usr/bin/env bash
# AITextJury one-command setup (Linux / macOS).
#
#   ./scripts/setup.sh             core stack (~1 min)
#   ./scripts/setup.sh --with-ml   + torch/transformers for the local-LM
#                                  detectors (~2 GB download)
#
# Creates an isolated virtualenv at apps/api/.venv. Your system Python is
# never touched. Idempotent - rerun any time.

set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${AITEXTJURY_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "python3 not found. Install Python 3.10+ or use: docker compose up"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || { echo "Python 3.10+ required."; exit 1; }
echo "[1/4] $($PY --version) found."

venv="$repo/apps/api/.venv"
if [ -d "$venv/bin" ]; then
    echo "[2/4] venv already exists, reusing $venv"
else
    echo "[2/4] creating isolated venv at $venv ..."
    "$PY" -m venv "$venv"
fi

echo "[3/4] installing backend dependencies ..."
vpip() { "$venv/bin/python" -m pip "$@"; }
vpip install --upgrade pip
if [ "${1:-}" = "--with-ml" ]; then
    vpip install -e "$repo/apps/api[ml]"
    echo "     + torch/transformers (native-LM detectors enabled)"
else
    vpip install -e "$repo/apps/api"
fi

echo "[4/4] web UI ..."
if command -v node >/dev/null 2>&1; then
    (cd "$repo/apps/web" && npm install --no-audit --no-fund) \
        || echo "     npm install failed. If npmjs is blocked: npm config set registry https://registry.npmmirror.com"
else
    echo "     Node.js not found - skipping web UI."
    echo "     API + CLI still work (http://localhost:8000); for the UI install Node 18+ or use docker compose up."
fi

echo ""
echo "Setup complete."
echo "  Run the workbench:   ./scripts/dev.sh        (opens http://localhost:5173)"
[ "${1:-}" = "--with-ml" ] || echo "  Later, native LM detectors: ./scripts/setup.sh --with-ml   (~2 GB)"
echo "  Or one-terminal mode: docker compose up"
