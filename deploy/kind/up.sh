#!/usr/bin/env bash
# Bring up the whole stack on a local kind cluster. Run from the repo root:
#   ./deploy/kind/up.sh            # create cluster (if missing), build, deploy
# Then: http://localhost/docs
set -euo pipefail
CLUSTER=dograg
ENVOY_GATEWAY_VERSION=v1.10.0

# 1. Postgres + Ollama run on the Mac; pods reach them via host.docker.internal
docker compose -f deploy/compose/docker-compose.yml --env-file .env up -d --wait postgres
curl -sf http://localhost:11434/ >/dev/null || { echo "Ollama is not running (ollama serve)"; exit 1; }

# 2. Cluster (with localhost:80 mapped in)
kind get clusters | grep -qx "$CLUSTER" || kind create cluster --config deploy/kind/cluster.yaml

# 3. Envoy Gateway (Gateway API CRDs + controller)
kubectl apply --server-side -f "https://github.com/envoyproxy/gateway/releases/download/${ENVOY_GATEWAY_VERSION}/install.yaml"
kubectl wait --timeout=5m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available

# 4. Images
docker build -t dograg-web:v1 services/rag
docker build -f services/agent/Dockerfile -t dog-agent:v1 .
kind load docker-image dograg-web:v1 dog-agent:v1 --name "$CLUSTER"

# 5. DB credentials -> Secret, read from .env (never committed, never printed)
DB_ENV=$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?(DB_NAME|DB_USER|DB_PASSWORD)[[:space:]]*=[[:space:]]*[\"']?([^\"']*)[\"']?[[:space:]]*$/\2=\3/p" .env)
[ "$(printf '%s\n' "$DB_ENV" | grep -c '=')" -eq 3 ] || { echo ".env must define DB_NAME, DB_USER and DB_PASSWORD"; exit 1; }
kubectl create secret generic dograg-db --from-env-file=<(printf '%s\n' "$DB_ENV") \
  --dry-run=client -o yaml | kubectl apply -f -

# 6. App + entry point
kubectl apply -k deploy/k8s
kubectl rollout restart deployment/dograg-web-deploy deployment/dog-agent-deploy
kubectl rollout status deployment/dograg-web-deploy --timeout=5m
kubectl rollout status deployment/dog-agent-deploy --timeout=2m
kubectl wait --timeout=2m gateway/dog-gateway --for=condition=Programmed

echo "Ready: http://localhost/docs"
