#!/usr/bin/env bash
# Installs every Golfie Python package (editable) plus the frontend's
# node_modules. Run once after cloning, and again after pulling changes
# that add a new package dependency.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Installing Python packages (editable)"
for pkg in golfie_core golfie_cv golfie_physics golfie_render; do
  echo "    pip install -e packages/$pkg"
  pip install --break-system-packages -e "packages/$pkg"
done

echo "    pip install -e apps/web/backend"
pip install --break-system-packages -e apps/web/backend

echo "==> Installing test dependencies"
pip install --break-system-packages pytest httpx

echo "==> Installing frontend dependencies"
(cd apps/web/frontend && npm install)

echo "==> Generating the synthetic demo session (for the Three.js driving range)"
python3 scripts/generate_sample_session.py

echo
echo "Done. Next steps:"
echo "  Backend:  cd apps/web/backend && uvicorn golfie_api.main:app --reload --port 8000"
echo "  Frontend: cd apps/web/frontend && npm run dev"
echo "  Tests:    pytest tests/"
