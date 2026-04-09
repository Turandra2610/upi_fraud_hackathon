"""
client.py — Python client wrapper for the UPI Fraud environment server.
Used by inference.py and testing scripts to interact with the environment.
"""

import os
from typing import Any, Dict, Optional

import requests


class UPIFraudClient:
    """HTTP client for the UPI Fraud OpenEnv server."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("ENV_BASE_URL", "http://localhost:7860")).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if task:
            payload["task"] = task
        if seed is not None:
            payload["seed"] = seed
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()
