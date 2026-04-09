"""
models.py — Typed Pydantic models for UPI Fraud Pattern Investigator
All request/response models used by the OpenEnv server and inference client.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Transaction data models
# ---------------------------------------------------------------------------

class UPITransaction(BaseModel):
    txn_id: str
    sender_vpa: str
    receiver_vpa: str
    amount: float
    timestamp: str                          # ISO 8601
    device_id: str
    location: str                           # city or lat/lon string
    merchant_category: str
    upi_app: str                            # e.g. PhonePe, GPay, Paytm
    is_new_payee: bool
    time_since_last_txn_minutes: Optional[float] = None
    daily_txn_count: Optional[int] = None
    weekly_txn_count: Optional[int] = None
    amount_deviation_pct: Optional[float] = None  # % deviation from user avg
    label: Optional[str] = None            # ground truth (hidden from agent)


class AccountHistory(BaseModel):
    account_vpa: str
    transactions: List[UPITransaction]
    account_age_days: int
    kyc_verified: bool
    linked_bank: str


class FraudNetwork(BaseModel):
    accounts: List[AccountHistory]
    total_transactions: int
    time_window_days: int


# ---------------------------------------------------------------------------
# Agent action models
# ---------------------------------------------------------------------------

class EasyAction(BaseModel):
    """Action for Task 1: single transaction classification."""
    decision: Literal["fraud", "legitimate", "needs-review"]
    confidence: float = Field(ge=0.0, le=1.0)
    fraud_type: Optional[str] = None       # e.g. "phishing", "mule", "sim-swap"
    reasoning: str


class MediumAction(BaseModel):
    """Action for Task 2: account compromise detection."""
    is_compromised: bool
    confidence: float = Field(ge=0.0, le=1.0)
    mule_pattern_detected: bool
    suspicious_txn_ids: List[str]
    reasoning: str


class HardAction(BaseModel):
    """Action for Task 3: fraud ring investigation."""
    ring_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    originator_account: Optional[str] = None
    cashout_endpoint: Optional[str] = None
    flagged_accounts: List[str]
    money_flow_summary: str
    reasoning: str


# ---------------------------------------------------------------------------
# Observation models
# ---------------------------------------------------------------------------

class EasyObservation(BaseModel):
    task: Literal["single-transaction-classify"] = "single-transaction-classify"
    transaction: UPITransaction
    context_hint: str = ""


class MediumObservation(BaseModel):
    task: Literal["account-compromise-detect"] = "account-compromise-detect"
    account: AccountHistory
    context_hint: str = ""


class HardObservation(BaseModel):
    task: Literal["fraud-ring-investigate"] = "fraud-ring-investigate"
    network: FraudNetwork
    context_hint: str = ""


# ---------------------------------------------------------------------------
# Step / Reset response models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


class ResetResult(BaseModel):
    observation: Dict[str, Any]
    task: str
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    task: str
    step: int
    done: bool
    total_reward: float
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None             # if None, picks randomly
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    max_steps: int
