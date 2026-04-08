from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class UPIAction(Action):
    decision: str = Field(..., description="allow, flag, or block")


class UPIObservation(Observation):
    amount: float = Field(..., description="Transaction amount normalized 0-1")
    hour: int = Field(..., description="Hour of transaction (0-23)")
    velocity: float = Field(..., description="Transaction velocity score 0-1")
    feedback: str = Field(..., description="Grader feedback on the decision")
    reward: float = Field(..., description="Reward signal for the action taken")
    done: bool = Field(..., description="Whether the episode is complete")
    task_id: str = Field(..., description="Task difficulty: easy, medium, or hard")
