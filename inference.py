"""
inference.py — UPI Fraud Pattern Investigator
Baseline inference script using OpenAI client against all 3 tasks.
Emits [START], [STEP], [END] logs in exact required format.

Environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your HuggingFace / API key
"""

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "upi-fraud-investigator"
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASKS = [
    "single-transaction-classify",
    "account-compromise-detect",
    "fraud-ring-investigate",
]

# ---------------------------------------------------------------------------
# Logging helpers — EXACT required format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string — no newlines allowed on a single line
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "single-transaction-classify": textwrap.dedent("""
        You are a UPI fraud detection expert for an Indian bank.
        Analyze a single UPI transaction and classify it.
        
        You MUST respond with ONLY valid JSON in this exact format:
        {
          "decision": "fraud" | "legitimate" | "needs-review",
          "confidence": <float 0.0-1.0>,
          "fraud_type": "<phishing|mule_account|sim_swap|account_takeover|null>",
          "reasoning": "<one sentence explanation>"
        }
        
        Fraud signals: high amount deviation, new payee, unknown location, rapid transactions.
        No markdown, no extra text — pure JSON only.
    """).strip(),

    "account-compromise-detect": textwrap.dedent("""
        You are a UPI fraud analyst specializing in mule account detection.
        Analyze an account's 7-day transaction history for compromise patterns.
        
        You MUST respond with ONLY valid JSON in this exact format:
        {
          "is_compromised": true | false,
          "confidence": <float 0.0-1.0>,
          "mule_pattern_detected": true | false,
          "suspicious_txn_ids": ["txn_id1", "txn_id2"],
          "reasoning": "<one sentence explanation>"
        }
        
        Mule patterns: rapid sequential transfers, round amounts just under limits,
        many new payees in short window, high daily transaction velocity.
        No markdown, no extra text — pure JSON only.
    """).strip(),

    "fraud-ring-investigate": textwrap.dedent("""
        You are a financial crimes investigator specializing in UPI fraud rings.
        Analyze a network of 15 accounts to detect organized fraud.
        
        You MUST respond with ONLY valid JSON in this exact format:
        {
          "ring_detected": true | false,
          "confidence": <float 0.0-1.0>,
          "originator_account": "<vpa or null>",
          "cashout_endpoint": "<vpa or null>",
          "flagged_accounts": ["vpa1", "vpa2"],
          "money_flow_summary": "<one sentence>",
          "reasoning": "<one sentence>"
        }
        
        Ring patterns: hub-spoke structure, layering through mule accounts,
        concentration of inflows to few accounts, timing clusters.
        No markdown, no extra text — pure JSON only.
    """).strip(),
}


def build_user_prompt(task: str, observation: Dict[str, Any]) -> str:
    obs_json = json.dumps(observation, indent=2)
    # Truncate very long observations to stay within token limits
    if len(obs_json) > 6000:
        obs_json = obs_json[:6000] + "\n... [truncated for brevity]"

    return f"Here is the transaction data to analyze:\n\n{obs_json}\n\nProvide your analysis as JSON."


def get_agent_action(
    client: OpenAI,
    task: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Call LLM and parse JSON action. Returns fallback on failure."""
    user_prompt = build_user_prompt(task, observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw: {text[:200]}", flush=True)
        return _fallback_action(task)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return _fallback_action(task)


def _fallback_action(task: str) -> Dict[str, Any]:
    """Safe fallback actions when LLM fails."""
    if task == "single-transaction-classify":
        return {
            "decision": "needs-review",
            "confidence": 0.5,
            "fraud_type": None,
            "reasoning": "Defaulting to needs-review due to analysis error.",
        }
    elif task == "account-compromise-detect":
        return {
            "is_compromised": False,
            "confidence": 0.5,
            "mule_pattern_detected": False,
            "suspicious_txn_ids": [],
            "reasoning": "Defaulting to not compromised due to analysis error.",
        }
    else:
        return {
            "ring_detected": False,
            "confidence": 0.5,
            "originator_account": None,
            "cashout_endpoint": None,
            "flagged_accounts": [],
            "money_flow_summary": "Unable to analyze.",
            "reasoning": "Defaulting to no ring detected due to analysis error.",
        }


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str) -> float:
    """Run a full episode for one task. Returns final score in [0, 1]."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_result = env_reset(task)
        observation = reset_result.get("observation", {})
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get agent action from LLM
            action = get_agent_action(client, task, observation)
            action_str = json.dumps(action, separators=(",", ":"))

            # Step environment
            step_result = env_step(action)

            reward = float(step_result.get("reward", 0.0))
            done = step_result.get("done", False)
            error = step_result.get("last_action_error", None)
            observation = step_result.get("observation", observation)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            time.sleep(0.5)  # rate limiting buffer

        # Score = best reward achieved (clamped to [0,1])
        score = max(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 2)
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Episode error for task={task}: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task in TASKS:
        score = run_task(client, task)
        all_scores.append(score)
        time.sleep(1)  # brief pause between tasks

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"[SUMMARY] tasks={len(TASKS)} avg_score={avg_score:.2f} "
        f"scores={','.join(f'{s:.2f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
