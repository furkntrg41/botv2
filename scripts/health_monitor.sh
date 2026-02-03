#!/usr/bin/env bash
set -euo pipefail

HEALTH_URL=${HEALTH_URL:-"http://localhost:8080/health"}
LOG_FILE=${LOG_FILE:-"/opt/trading-bot/logs/healthcheck.log"}
TIMEOUT=${TIMEOUT:-5}

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if curl -fsS --max-time "$TIMEOUT" "$HEALTH_URL" > /tmp/healthcheck.json; then
  echo "$TS OK $(cat /tmp/healthcheck.json)" >> "$LOG_FILE"
  exit 0
else
  echo "$TS FAIL" >> "$LOG_FILE"
  exit 1
fi
