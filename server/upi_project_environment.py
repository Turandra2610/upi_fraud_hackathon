"""
server/upi_project_environment.py
Core UPI Fraud RL Environment — generates synthetic transactions,
scores agent actions, and manages episode state.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Kanpur", "Nagpur", "Visakhapatnam", "Coimbatore",
]

UPI_APPS = ["PhonePe", "GPay", "Paytm", "BHIM", "AmazonPay", "WhatsAppPay"]

MERCHANT_CATEGORIES = [
    "retail", "food", "transport", "utility", "healthcare",
    "education", "entertainment", "transfer", "ecommerce", "fuel",
]

FRAUD_TYPES = ["phishing", "mule_account", "sim_swap", "account_takeover", "synthetic_identity"]

BANK_NAMES = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Canara", "Union", "IndusInd"]


def _vpa(seed: str = "") -> str:
    names = ["ravi", "priya", "amit", "sneha", "kumar", "deepa", "raj", "anita", "suresh", "pooja"]
    banks = ["okaxis", "okicici", "oksbi", "okhdfcbank", "ybl", "ibl"]
    name = random.choice(names) + str(random.randint(100, 9999))
    bank = random.choice(banks)
    return f"{name}@{bank}"


def _timestamp(base: datetime, offset_hours: float = 0) -> str:
    return (base + timedelta(hours=offset_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_transaction(
    sender_vpa: str,
    receiver_vpa: str,
    amount: float,
    base_time: datetime,
    offset_hours: float = 0,
    is_suspicious: bool = False,
    label: str = "legitimate",
) -> Dict[str, Any]:
    return {
        "txn_id": str(uuid.uuid4())[:12],
        "sender_vpa": sender_vpa,
        "receiver_vpa": receiver_vpa,
        "amount": round(amount, 2),
        "timestamp": _timestamp(base_time, offset_hours),
        "device_id": f"DEV-{random.randint(1000, 9999)}",
        "location": random.choice(INDIAN_CITIES) if not is_suspicious else random.choice(["Unknown", "Foreign-IP", "VPN"]),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "upi_app": random.choice(UPI_APPS),
        "is_new_payee": is_suspicious or random.random() < 0.15,
        "time_since_last_txn_minutes": round(random.uniform(0.5, 2880), 1) if not is_suspicious else round(random.uniform(0.1, 5), 1),
        "daily_txn_count": random.randint(1, 5) if not is_suspicious else random.randint(15, 40),
        "weekly_txn_count": random.randint(3, 20) if not is_suspicious else random.randint(50, 150),
        "amount_deviation_pct": round(random.uniform(-20, 20), 1) if not is_suspicious else round(random.uniform(200, 800), 1),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def generate_easy_task(seed: Optional[int] = None) -> Dict[str, Any]:
    """Single transaction — classify as fraud/legitimate/needs-review."""
    if seed is not None:
        random.seed(seed)

    base_time = datetime(2025, 1, 15, 10, 0, 0)
    scenario = random.choice(["fraud", "legitimate", "needs-review"])
    sender = _vpa("sender")
    receiver = _vpa("receiver")

    if scenario == "fraud":
        txn = _make_transaction(
            sender, receiver,
            amount=random.uniform(45000, 99000),
            base_time=base_time,
            is_suspicious=True,
            label="fraud",
        )
        hint = "Transaction shows unusual amount and new payee at odd hours."
    elif scenario == "legitimate":
        txn = _make_transaction(
            sender, receiver,
            amount=random.uniform(50, 2000),
            base_time=base_time,
            is_suspicious=False,
            label="legitimate",
        )
        hint = "Transaction appears routine — regular merchant, known payee."
    else:
        txn = _make_transaction(
            sender, receiver,
            amount=random.uniform(8000, 20000),
            base_time=base_time,
            is_suspicious=True,
            label="needs-review",
        )
        txn["location"] = random.choice(INDIAN_CITIES)  # ambiguous — city but high amount
        hint = "Amount is significantly above average; new payee but domestic location."

    return {
        "task": "single-transaction-classify",
        "transaction": txn,
        "context_hint": hint,
        "ground_truth": scenario,
    }


def generate_medium_task(seed: Optional[int] = None) -> Dict[str, Any]:
    """20 transactions from one account over 7 days — detect mule pattern."""
    if seed is not None:
        random.seed(seed)

    base_time = datetime(2025, 1, 10, 9, 0, 0)
    account_vpa = _vpa("account")
    is_compromised = random.choice([True, False])

    transactions = []
    suspicious_ids = []

    if is_compromised:
        # First 10: normal behaviour
        for i in range(10):
            receiver = _vpa()
            txn = _make_transaction(account_vpa, receiver, random.uniform(200, 2000), base_time, i * 12)
            transactions.append(txn)

        # Last 10: mule pattern — rapid small transfers to many new accounts
        for i in range(10):
            receiver = _vpa()
            txn = _make_transaction(
                account_vpa, receiver,
                amount=random.uniform(9000, 49500),  # just under limits
                base_time=base_time,
                offset_hours=60 + i * 0.25,          # rapid fire
                is_suspicious=True,
                label="fraud",
            )
            transactions.append(txn)
            suspicious_ids.append(txn["txn_id"])
        label = True
    else:
        for i in range(20):
            receiver = _vpa()
            txn = _make_transaction(account_vpa, receiver, random.uniform(100, 5000), base_time, i * 8)
            transactions.append(txn)
        label = False

    return {
        "task": "account-compromise-detect",
        "account": {
            "account_vpa": account_vpa,
            "transactions": transactions,
            "account_age_days": random.randint(90, 1500),
            "kyc_verified": True,
            "linked_bank": random.choice(BANK_NAMES),
        },
        "context_hint": "Analyze velocity, amount deviation, and payee patterns.",
        "ground_truth": {
            "is_compromised": label,
            "suspicious_txn_ids": suspicious_ids,
        },
    }


def generate_hard_task(seed: Optional[int] = None) -> Dict[str, Any]:
    """100 transactions across 15 accounts — detect fraud ring."""
    if seed is not None:
        random.seed(seed)

    base_time = datetime(2025, 1, 5, 8, 0, 0)
    num_accounts = 15
    vpas = [_vpa(f"acc{i}") for i in range(num_accounts)]

    # Ring structure: originator → 5 mules → 3 cashout accounts
    originator_idx = 0
    mule_idxs = [1, 2, 3, 4, 5]
    cashout_idxs = [6, 7, 8]
    normal_idxs = list(range(9, 15))

    originator_vpa = vpas[originator_idx]
    cashout_vpas = [vpas[i] for i in cashout_idxs]

    accounts: Dict[str, List] = {vpa: [] for vpa in vpas}
    flagged = [originator_vpa] + [vpas[i] for i in mule_idxs] + cashout_vpas

    # Originator sends to mules
    for i, mule_idx in enumerate(mule_idxs):
        for j in range(4):
            txn = _make_transaction(
                originator_vpa, vpas[mule_idx],
                amount=random.uniform(40000, 98000),
                base_time=base_time,
                offset_hours=i * 2 + j * 0.5,
                is_suspicious=True,
                label="fraud",
            )
            accounts[originator_vpa].append(txn)

    # Mules forward to cashout
    for mule_idx in mule_idxs:
        for cashout_idx in cashout_idxs[:2]:
            txn = _make_transaction(
                vpas[mule_idx], vpas[cashout_idx],
                amount=random.uniform(20000, 49000),
                base_time=base_time,
                offset_hours=random.uniform(6, 24),
                is_suspicious=True,
                label="fraud",
            )
            accounts[vpas[mule_idx]].append(txn)

    # Normal accounts — clean activity
    for idx in normal_idxs:
        for k in range(random.randint(3, 8)):
            receiver = _vpa()
            txn = _make_transaction(vpas[idx], receiver, random.uniform(100, 3000), base_time, k * 10)
            accounts[vpas[idx]].append(txn)

    # Build account history objects
    account_list = []
    for vpa, txns in accounts.items():
        account_list.append({
            "account_vpa": vpa,
            "transactions": txns,
            "account_age_days": random.randint(30, 1000),
            "kyc_verified": random.random() > 0.3,
            "linked_bank": random.choice(BANK_NAMES),
        })

    total_txns = sum(len(a["transactions"]) for a in account_list)

    return {
        "task": "fraud-ring-investigate",
        "network": {
            "accounts": account_list,
            "total_transactions": total_txns,
            "time_window_days": 7,
        },
        "context_hint": "Look for hub-spoke money flow patterns and rapid layering.",
        "ground_truth": {
            "ring_detected": True,
            "originator_account": originator_vpa,
            "cashout_endpoints": cashout_vpas,
            "flagged_accounts": flagged,
        },
    }


# ---------------------------------------------------------------------------
# Reward / grader functions
# ---------------------------------------------------------------------------

def grade_easy(action: Dict[str, Any], ground_truth: str) -> Tuple[float, str]:
    """Score easy task — single transaction classification."""
    decision = action.get("decision", "")
    confidence = float(action.get("confidence", 0.5))
    reasoning = action.get("reasoning", "")

    if decision == ground_truth:
        # Correct classification
        base = 0.6
        conf_bonus = 0.2 * confidence          # reward high confidence when correct
        reasoning_bonus = 0.2 if len(reasoning) > 20 else 0.0
        reward = min(base + conf_bonus + reasoning_bonus, 1.0)
        info = f"Correct: {decision}"
    elif ground_truth == "needs-review" and decision in ("fraud", "legitimate"):
        # Partial credit — wrong but adjacent
        reward = 0.3 * (1 - confidence)        # less penalty if uncertain
        info = f"Partial: expected needs-review, got {decision}"
    elif ground_truth == "fraud" and decision == "needs-review":
        reward = 0.25
        info = "Partial: flagged for review but missed full fraud call"
    else:
        # Completely wrong (e.g. fraud marked as legitimate)
        reward = max(0.0, 0.1 * (1 - confidence))
        info = f"Wrong: expected {ground_truth}, got {decision}"

    return round(reward, 4), info


def grade_medium(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Score medium task — account compromise detection."""
    is_compromised_pred = action.get("is_compromised", False)
    confidence = float(action.get("confidence", 0.5))
    suspicious_ids_pred = set(action.get("suspicious_txn_ids", []))
    suspicious_ids_true = set(ground_truth.get("suspicious_txn_ids", []))
    correct_label = ground_truth.get("is_compromised", False)

    label_correct = is_compromised_pred == correct_label

    if not label_correct:
        reward = max(0.0, 0.1 * (1 - confidence))
        return round(reward, 4), f"Wrong label: expected {correct_label}"

    base = 0.5

    # Partial credit for txn identification (only matters when compromised)
    if correct_label and suspicious_ids_true:
        if suspicious_ids_pred:
            precision = len(suspicious_ids_pred & suspicious_ids_true) / len(suspicious_ids_pred)
            recall = len(suspicious_ids_pred & suspicious_ids_true) / len(suspicious_ids_true)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
        else:
            f1 = 0.0
        txn_score = 0.4 * f1
    else:
        txn_score = 0.4  # not compromised + correct = full credit

    conf_bonus = 0.1 * confidence if label_correct else 0.0
    reward = min(base + txn_score + conf_bonus, 1.0)
    return round(reward, 4), f"Label correct={label_correct}, txn_f1={txn_score:.2f}"


def grade_hard(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Score hard task — fraud ring investigation."""
    ring_detected = action.get("ring_detected", False)
    confidence = float(action.get("confidence", 0.5))
    originator_pred = action.get("originator_account", "")
    cashout_pred = action.get("cashout_endpoint", "")
    flagged_pred = set(action.get("flagged_accounts", []))

    true_originator = ground_truth.get("originator_account", "")
    true_cashouts = set(ground_truth.get("cashout_endpoints", []))
    true_flagged = set(ground_truth.get("flagged_accounts", []))

    if not ring_detected:
        return 0.05 if not ground_truth.get("ring_detected") else 0.0, "Ring not detected"

    score = 0.0

    # Ring detected: 0.2 base
    score += 0.2

    # Originator correct: +0.25
    if originator_pred == true_originator:
        score += 0.25

    # Cashout endpoint in true cashouts: +0.2
    if cashout_pred in true_cashouts:
        score += 0.2

    # Flagged account overlap (partial credit)
    if true_flagged and flagged_pred:
        precision = len(flagged_pred & true_flagged) / len(flagged_pred)
        recall = len(flagged_pred & true_flagged) / len(true_flagged)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        score += 0.25 * f1

    # Confidence calibration bonus
    score += 0.1 * confidence

    reward = round(min(score, 1.0), 4)
    return reward, f"Ring={ring_detected}, originator={'✓' if originator_pred==true_originator else '✗'}"


# ---------------------------------------------------------------------------
# Environment state manager
# ---------------------------------------------------------------------------

class UPIFraudEnvironment:
    """Stateful per-session environment manager."""

    TASKS = ["single-transaction-classify", "account-compromise-detect", "fraud-ring-investigate"]

    def __init__(self):
        self.task: Optional[str] = None
        self.step_count: int = 0
        self.max_steps: int = 5
        self.done: bool = False
        self.total_reward: float = 0.0
        self.current_data: Optional[Dict[str, Any]] = None
        self.rewards_history: List[float] = []

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        self.task = task if task in self.TASKS else random.choice(self.TASKS)
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.rewards_history = []

        if self.task == "single-transaction-classify":
            self.max_steps = 5
            self.current_data = generate_easy_task(seed)
        elif self.task == "account-compromise-detect":
            self.max_steps = 10
            self.current_data = generate_medium_task(seed)
        else:
            self.max_steps = 20
            self.current_data = generate_hard_task(seed)

        obs = {k: v for k, v in self.current_data.items() if k != "ground_truth"}
        return {"observation": obs, "task": self.task, "info": {"max_steps": self.max_steps}}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.done:
            return {
                "observation": {},
                "reward": 0.0,
                "done": True,
                "info": {"error": "Episode already done"},
                "last_action_error": "Episode already done",
            }

        self.step_count += 1
        error = None

        try:
            ground_truth = self.current_data["ground_truth"]

            if self.task == "single-transaction-classify":
                reward, info_msg = grade_easy(action, ground_truth)
            elif self.task == "account-compromise-detect":
                reward, info_msg = grade_medium(action, ground_truth)
            else:
                reward, info_msg = grade_hard(action, ground_truth)

        except Exception as e:
            reward = 0.0
            info_msg = f"Grading error: {e}"
            error = str(e)

        self.total_reward += reward
        self.rewards_history.append(reward)

        # Episode ends after first meaningful action or max_steps
        self.done = (self.step_count >= self.max_steps) or (reward > 0.3)

        obs = {k: v for k, v in self.current_data.items() if k != "ground_truth"}

        return {
            "observation": obs,
            "reward": reward,
            "done": self.done,
            "info": {"step": self.step_count, "msg": info_msg},
            "last_action_error": error,
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "step": self.step_count,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "rewards_history": self.rewards_history,
            "info": {"max_steps": self.max_steps},
        }
