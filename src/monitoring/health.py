"""Lightweight healthcheck server for liveness/readiness signals."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class HealthState:
    """Shared health state exposed by the HTTP server."""

    status: str = "starting"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_bar_time: datetime | None = None
    last_order_time: datetime | None = None
    last_error: str | None = None
    strategy: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    position_open: bool = False
    position_size: float | None = None
    last_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "last_bar_time": self._dt(self.last_bar_time),
            "last_order_time": self._dt(self.last_order_time),
            "last_error": self.last_error,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "position_open": self.position_open,
            "position_size": self.position_size,
            "last_price": self.last_price,
        }

    @staticmethod
    def _dt(value: datetime | None) -> str | None:
        return value.isoformat() if value else None


class HealthServer:
    """Minimal HTTP server for health endpoints."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, state: HealthState | None = None) -> None:
        self._host = host
        self._port = port
        self._state = state or HealthState()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> HealthState:
        return self._state

    def start(self) -> None:
        if self._server is not None:
            return

        server = self._build_server()
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def _build_server(self) -> ThreadingHTTPServer:
        snapshot = self._snapshot

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path not in ("/health", "/healthz", "/ready"):
                    self.send_response(404)
                    self.end_headers()
                    return

                payload = json.dumps(snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        return ThreadingHTTPServer((self._host, self._port), Handler)
