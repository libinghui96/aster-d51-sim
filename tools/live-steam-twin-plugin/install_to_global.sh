#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST_DIR="${HOME}/.gemini/config/plugins/live-steam-twin"
mkdir -p "${DST_DIR}"
rsync -a --exclude='install_to_global.sh' --exclude='.git' --exclude='__pycache__' "${SRC_DIR}/" "${DST_DIR}/"
chmod +x "${DST_DIR}/hooks/"*.py "${DST_DIR}/skills/"*/scripts/*.py
echo "[OK] Installed live-steam-twin global plugin to ${DST_DIR}"
