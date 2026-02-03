"""Telegram notifications for trade events."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger


@dataclass
class TelegramNotifier:
    """Send messages to a Telegram bot/chat if configured."""

    token: Optional[str] = None
    chat_id: Optional[str] = None
    min_interval: float = 1.0
    _last_sent: float = 0.0

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN"),
            chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            min_interval=float(os.getenv("TELEGRAM_MIN_INTERVAL", "1.0")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> None:
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_sent < self.min_interval:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
                return
            self._last_sent = now
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Telegram send error: {exc}")
