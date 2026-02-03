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
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        min_interval = float(os.getenv("TELEGRAM_MIN_INTERVAL", "1.0"))
        notifier = cls(token=token, chat_id=chat_id, min_interval=min_interval)
        if notifier.enabled:
            logger.success(f"✅ Telegram notifier enabled (chat={chat_id[:10]}...)")
        else:
            logger.warning("⚠️  Telegram notifier disabled (missing token or chat_id)")
        return notifier

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> None:
        if not self.enabled:
            logger.debug(f"Telegram disabled, skipping: {message}")
            return

        now = time.time()
        if now - self._last_sent < self.min_interval:
            logger.debug(f"Telegram rate limit, skipping: {message}")
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
                return
            logger.info(f"📨 Telegram sent: {message[:50]}...")
            self._last_sent = now
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Telegram send error: {exc}")
