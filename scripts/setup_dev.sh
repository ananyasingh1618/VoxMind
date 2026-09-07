#!/usr/bin/env bash
# One-shot bootstrap for Mode A (infrastructure-only Docker + local API/frontend).
# See docs/development.md for what this does and does not do, and for Modes B/C.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Starting infrastructure (Postgres, Redis, MinIO)…"
docker compose -f "$ROOT_DIR/docker/docker-compose.yml" up -d

echo "==> Waiting for Postgres to become healthy…"
until docker compose -f "$ROOT_DIR/docker/docker-compose.yml" exec -T postgres pg_isready -U voxmind >/dev/null 2>&1; do
  sleep 1
done

echo "==> Setting up backend virtualenv…"
cd "$ROOT_DIR/apps/api"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -e ".[dev]"

if [ ! -f .env ]; then
  echo "==> Creating apps/api/.env from .env.example (generating JWT_SECRET_KEY)…"
  cp .env.example .env
  SECRET="$(openssl rand -hex 32)"
  # Portable in-place sed for both GNU and BSD sed.
  sed -i.bak "s/^JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${SECRET}/" .env && rm -f .env.bak
fi

echo "==> Running database migrations…"
alembic upgrade head

echo "==> Setting up frontend…"
cd "$ROOT_DIR/apps/web"
npm install
if [ ! -f .env ]; then
  cp .env.example .env
fi

cat <<'EOF'

==> Done. Infrastructure is running. To start the app:

  Terminal 1: cd apps/api && source .venv/bin/activate && uvicorn voxmind.main:app --reload
  Terminal 2: cd apps/web && npm run dev

Then open http://localhost:5173
EOF
