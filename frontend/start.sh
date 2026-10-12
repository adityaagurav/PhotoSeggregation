#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec npx vite --host 0.0.0.0 --port 5173 "$@"
