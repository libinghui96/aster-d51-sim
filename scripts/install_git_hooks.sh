#!/usr/bin/env bash
# Install automatic Git pre-commit, post-commit & pre-push hooks in aster-d51-sim
# so any update to docs/LESSONS_LEARNED.md or tools/live-steam-twin-plugin/ automatically
# updates, commits, and pushes ~/.gemini/config/plugins/live-steam-twin (libinghui96/live-steam-twin).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="${REPO_ROOT}/.git/hooks"
mkdir -p "${HOOKS_DIR}"

cat > "${HOOKS_DIR}/pre-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
if git diff --cached --name-only | grep -qE '^(docs/LESSONS_LEARNED\.md|tools/live-steam-twin-plugin/)'; then
  python3 "${ROOT}/scripts/11_sync_lessons_to_plugin.py" --stage-workspace
fi
EOF

cat > "${HOOKS_DIR}/post-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
python3 "${ROOT}/scripts/11_sync_lessons_to_plugin.py" --commit-plugin || true
EOF

cat > "${HOOKS_DIR}/pre-push" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
python3 "${ROOT}/scripts/11_sync_lessons_to_plugin.py" --push-plugin || true
EOF

chmod +x "${HOOKS_DIR}/pre-commit" "${HOOKS_DIR}/post-commit" "${HOOKS_DIR}/pre-push"
echo "[OK] Installed pre-commit, post-commit & pre-push auto-sync hooks in ${HOOKS_DIR}"
