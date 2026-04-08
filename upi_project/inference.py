import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

# Importing your specific UPI Fraud classes
from upi_fraud_env.client import UPIAction, UPIEnv 

# Configuration from Environment Variables
IMAGE_NAME = os.getenv("IMAGE_NAME", "upi_fraud_env")
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Benchmark Settings
MAX_STEPS = 5
TEMPERATURE = 0.4 # Lower temperature for more consistent reasoning
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.6 

# The "Job Description" for the AI
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a UPI Fraud Investigator. 
    You will be presented with transaction data including:
    - Transaction Amount
    - Sender Velocity (how many transactions in the last hour)
    - Device Status (New or Recognized)
    
    Your goal is to protect the user while minimizing false blocks.
    Actions you can take:
    - 'Allow': For safe, routine transactions.
    - 'Flag': For suspicious patterns that need human review.
    - 'Block': For high-probability fraud.
    
    Response format: Only return the word 'Allow', 'Flag', or 'Block'.
    """
).strip()

def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP {step}] Action: {action} | Reward: {reward:.2f} | Done: {done}", flush=True)

def build_user_prompt(obs) -> str:
    return f"Investigate this transaction: {obs}. What is your decision?"

async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN is missing. Set it in your environment variables.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Launching the environment from your Docker image
    print(f"[INIT] Starting environment from image: {IMAGE_NAME}...")
    env = await UPIEnv.from_docker_image(IMAGE_NAME)

    rewards = []
    steps_taken = 0
    
    try:
        # Initial Reset
        result = await env.reset()
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            # 1. AI decides based on the Observation
            user_prompt = build_user_prompt(obs)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            ai_decision = (completion.choices[0].message.content or "Allow").strip()

            # 2. Execute action in the environment
            # Ensure action name matches your UPIAction 'action_type' field
            result = await env.step(UPIAction(action_type=ai_decision))
            
            reward = result.reward or 0.0
            rewards.append(reward)
            obs = result.observation
            steps_taken = step
            
            log_step(step, ai_decision, reward, result.done)

            if result.done:
                break

        # Final Scoring
        total_reward = sum(rewards)
        # Normalize score between 0 and 1 (Assuming max reward per step is 10)
        final_score = min(max(total_reward / (MAX_STEPS * 10), 0.0), 1.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        
        print(f"\n[RESULT] Success: {success} | Score: {final_score:.3f} | Total Steps: {steps_taken}")

    finally:
        await env.close()
        print("[CLEANUP] Environment closed.")

if __name__ == "__main__":
    asyncio.run(main())