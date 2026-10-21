#!/usr/bin/env bash
# One-command production deploy for the CareerPilot frontend (Netlify).
#
# Why this exists: the Netlify site is deployed via the Netlify CLI, NOT a GitHub hook,
# so `git push` alone does NOT update the live site. Run this after pushing frontend
# changes. It builds a static export with the correct API base baked in, verifies the
# build, then deploys that exact output to production.
#
# Usage:
#   scripts/deploy_web.sh                         # build + deploy to production
#   scripts/deploy_web.sh --preview               # deploy a draft (preview URL, not prod)
#   NEXT_PUBLIC_API_BASE=https://my-api scripts/deploy_web.sh   # override the backend URL
#
# Requires: the Netlify CLI (`netlify`) authenticated to this site (see `netlify status`).
set -euo pipefail

# Repo root = parent of this script's dir, so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="$ROOT/web"

API_BASE="${NEXT_PUBLIC_API_BASE:-https://careerpilot-seeh.onrender.com}"
PROD_FLAG="--prod"
[ "${1:-}" = "--preview" ] && PROD_FLAG=""

echo "→ Building frontend with NEXT_PUBLIC_API_BASE=$API_BASE"
( cd "$WEB" && NEXT_PUBLIC_API_BASE="$API_BASE" npm run build )

OUT="$WEB/out"
echo "→ Verifying build output in $OUT"
[ -f "$OUT/index.html" ] || { echo "✗ build produced no out/index.html — aborting"; exit 1; }
if ! grep -rq "$API_BASE" "$OUT"; then
  echo "✗ API base '$API_BASE' not found in build — refusing to deploy a frontend that can't reach the backend"
  exit 1
fi
echo "✓ API base baked in"

echo "→ Deploying to Netlify ${PROD_FLAG:-(preview)}"
( cd "$ROOT" && netlify deploy $PROD_FLAG --dir web/out --no-build )

echo "✓ Done. Live at https://careerpilot-raghu298.netlify.app"
