"""
server/app.py — FastAPI OpenEnv-compliant server for UPI Fraud Pattern Investigator
Endpoints: POST /reset, POST /step, GET /state, GET /tasks,
           GET /health, GET /metadata, GET /schema, POST /mcp
"""

import os
import uvicorn
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.upi_project_environment import UPIFraudEnvironment

app = FastAPI(
    title="UPI Fraud Pattern Investigator",
    description="OpenEnv-compliant RL environment for detecting fraud patterns in Indian UPI transactions.",
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
# Required OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — openenv validate requires status='healthy'."""
    return {"status": "healthy", "service": "upi-fraud-investigator"}


@app.get("/metadata")
def metadata():
    """Environment metadata — required by openenv validate."""
    return {
        "name": "UPI Fraud Pattern Investigator",
        "description": (
            "An OpenEnv RL environment where an agent investigates UPI transaction streams "
            "to detect fraud patterns unique to Indian digital payments. "
            "India processes 14 billion UPI transactions per month — fraud is exploding."
        ),
        "version": "1.0.0",
        "tasks": ["single-transaction-classify", "account-compromise-detect", "fraud-ring-investigate"],
        "difficulty_levels": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema():
    """Action/observation/state schemas — required by openenv validate."""
    return {
        "action": {
            "type": "object",
            "description": "Agent decision with confidence and reasoning",
            "properties": {
                "decision": {"type": "string", "enum": ["fraud", "legitimate", "needs-review"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "fraud_type": {"type": "string", "nullable": True},
                "reasoning": {"type": "string"},
                "is_compromised": {"type": "boolean"},
                "mule_pattern_detected": {"type": "boolean"},
                "suspicious_txn_ids": {"type": "array", "items": {"type": "string"}},
                "ring_detected": {"type": "boolean"},
                "originator_account": {"type": "string", "nullable": True},
                "cashout_endpoint": {"type": "string", "nullable": True},
                "flagged_accounts": {"type": "array", "items": {"type": "string"}},
                "money_flow_summary": {"type": "string"},
            },
        },
        "observation": {
            "type": "object",
            "description": "UPI transaction data for analysis",
            "properties": {
                "task": {"type": "string"},
                "transaction": {"type": "object", "description": "Single UPI transaction (easy task)"},
                "account": {"type": "object", "description": "Account history with 20 transactions (medium task)"},
                "network": {"type": "object", "description": "15-account fraud network (hard task)"},
                "context_hint": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "description": "Current episode state",
            "properties": {
                "task": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "total_reward": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rewards_history": {"type": "array", "items": {"type": "number"}},
            },
        },
    }


@app.post("/mcp")
def mcp(request: Dict[str, Any] = {}):
    """MCP JSON-RPC endpoint — required by openenv validate."""
    method = request.get("method", "")
    req_id = request.get("id", 1)

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment and start a new episode",
                    "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}}},
                },
                {
                    "name": "step",
                    "description": "Submit an agent action and receive a reward",
                    "inputSchema": {"type": "object", "properties": {"action": {"type": "object"}}},
                },
                {
                    "name": "state",
                    "description": "Get current environment state",
                    "inputSchema": {"type": "object"},
                },
            ]
        }
    elif method == "tools/call":
        tool_name = request.get("params", {}).get("name", "")
        tool_args = request.get("params", {}).get("arguments", {})
        if tool_name == "reset":
            env_result = _env.reset(task=tool_args.get("task"), seed=tool_args.get("seed"))
            result = {"content": [{"type": "text", "text": str(env_result)}]}
        elif tool_name == "step":
            env_result = _env.step(tool_args.get("action", {}))
            result = {"content": [{"type": "text", "text": str(env_result)}]}
        elif tool_name == "state":
            env_result = _env.state()
            result = {"content": [{"type": "text", "text": str(env_result)}]}
        else:
            result = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}
    else:
        result = {"message": "UPI Fraud Pattern Investigator MCP endpoint ready"}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


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
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/metadata", "/schema", "/mcp"],
    }


# ---------------------------------------------------------------------------
# main() — required by openenv validate (project.scripts entry point)
# ---------------------------------------------------------------------------

def main():
    """Entry point for 'uv run server' and openenv serve."""
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
