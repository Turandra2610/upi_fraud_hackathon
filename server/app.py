"""
server/app.py — FastAPI OpenEnv-compliant server for UPI Fraud Pattern Investigator
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.upi_project_environment import UPIFraudEnvironment

app = FastAPI(
    title="UPI Fraud Pattern Investigator",
    description="OpenEnv-compliant RL environment for UPI fraud detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per session)
_env = UPIFraudEnvironment()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — used by HF Space ping."""
    return {"status": "ok", "service": "upi-fraud-investigator"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.
    Required by OpenEnv spec — must return 200.
    """
    try:
        result = _env.reset(task=request.task, seed=request.seed)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.
    Returns observation, reward (0.0–1.0), done flag, and info.
    """
    try:
        result = _env.step(request.action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return current environment state."""
    try:
        return _env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "name": "single-transaction-classify",
                "difficulty": "easy",
                "description": "Classify one UPI transaction as fraud/legitimate/needs-review",
                "max_steps": 5,
            },
            {
                "name": "account-compromise-detect",
                "difficulty": "medium",
                "description": "Detect if an account is compromised (mule pattern) from 20 transactions over 7 days",
                "max_steps": 10,
            },
            {
                "name": "fraud-ring-investigate",
                "difficulty": "hard",
                "description": "Map a fraud ring across 15 accounts — find originator and cash-out endpoint",
                "max_steps": 20,
            },
        ]
    }


@app.get("/")
def root():
    """Root endpoint — returns environment info."""
    return {
        "name": "UPI Fraud Pattern Investigator",
        "version": "1.0.0",
        "description": "OpenEnv RL environment for Indian UPI fraud detection",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }
