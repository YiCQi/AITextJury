#!/usr/bin/env bash
# Start the ZeroAIBench workbench (backend :8000 + web UI :5173).
# Prerequisite: ./scripts/setup.sh has run at least once.
# Logs: data/dev-backend.log, data/dev-web.log   Stop: Ctrl-C (fg) or kill the PIDs printed below.

set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/apps/api/.venv/bin/python"

[ -x "$py" ] || { echo "venv not found - run ./scripts/setup.sh first."; exit 1; }
[ -d "$repo/apps/web/node_modules" ] || { echo "web deps missing - run ./scripts/setup.sh first."; exit 1; }

mkdir -p "$repo/data"

echo "Starting backend  -> http://localhost:8000   (log: data/dev-backend.log)"
(cd "$repo/apps/api" && exec "$py" -m zeroaibench) > "$repo/data/dev-backend.log" 2>&1 &
BACKEND_PID=$!
echo "  backend pid: $BACKEND_PID"

echo "Starting web UI   -> http://localhost:5173   (log: data/dev-web.log)"
(cd "$repo/apps/web" && exec npm run dev) > "$repo/data/dev-web.log" 2>&1 &
WEB_PID=$!
echo "  web pid: $WEB_PID"

echo "Waiting for Vite to boot ..."
sleep 6
if command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:5173
elif command -v open    >/dev/null 2>&1; then open http://localhost:5173
else echo "Open http://localhost:5173 in your browser."; fi

echo ""
echo "Stop with:  kill $BACKEND_PID $WEB_PID"
wait
