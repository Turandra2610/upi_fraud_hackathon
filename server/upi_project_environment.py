import uuid
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import UPIAction, UPIObservation


class UPIFraudEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._transaction = None

    def reset(self) -> UPIObservation:
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._transaction = self._generate_transaction()
        return UPIObservation(
            amount=float(self._transaction["amount"]),
            hour=int(self._transaction["hour"]),
            velocity=float(self._transaction["velocity"]),
            feedback="New transaction received. Classify as allow, flag, or block.",
            reward=0.0,
            done=False,
            task_id="easy"
        )

    def step(self, action: UPIAction) -> UPIObservation:
        self._state.step_count += 1
        amount = self._transaction["amount"]
        hour = self._transaction["hour"]
        velocity = self._transaction["velocity"]
        is_fraud = self._transaction["is_fraud"]

        reward, feedback = self._compute_reward(action.decision, is_fraud, amount)

        return UPIObservation(
            amount=float(amount),
            hour=int(hour),
            velocity=float(velocity),
            feedback=feedback,
            reward=reward,
            done=True,
            task_id="easy"
        )

    def _generate_transaction(self):
        # amount normalized 0-1 (matching your original logic)
        amount = float(np.random.rand())
        hour = int(np.random.randint(0, 24))
        velocity = float(np.random.rand())

        # Your original fraud logic: high amount = fraud, low amount = safe
        if amount > 0.8:
            is_fraud = True
        elif amount < 0.2:
            is_fraud = False
        else:
            # Middle range — random
            is_fraud = bool(np.random.rand() > 0.7)

        return {
            "amount": amount,
            "hour": hour,
            "velocity": velocity,
            "is_fraud": is_fraud
        }

    def _compute_reward(self, decision: str, is_fraud: bool, amount: float):
        # Mirrors your original reward logic exactly
        if amount > 0.8:  # Likely Fraudulent High-Value Transaction
            if decision == "block":
                return 10.0, "Correct! High-value fraud blocked successfully."
            elif decision == "flag":
                return 3.0, "Partially correct. Flagged but should be blocked."
            else:
                return -20.0, "Missed fraud! High-value fraudulent transaction allowed."

        elif amount < 0.2:  # Likely Safe Small Transaction
            if decision == "allow":
                return 5.0, "Correct! Legitimate small transaction allowed."
            elif decision == "flag":
                return 1.0, "Overly cautious. Small transaction flagged unnecessarily."
            else:
                return -10.0, "False positive! Legitimate transaction blocked."

        else:  # Normal Transaction
            if decision == "flag":
                return 1.0, "Good call. Normal transaction flagged for review."
            elif decision == "allow":
                return 1.0, "Allowed normal transaction."
            else:
                return -5.0, "Incorrectly blocked a normal transaction."

    @property
    def state(self) -> State:
        return self._state
