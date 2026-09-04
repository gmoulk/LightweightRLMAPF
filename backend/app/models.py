from pydantic import BaseModel
from typing import List, Optional

class MapfRequest(BaseModel):
    num_agents: int = 4
    size: int = 16
    density: float = 0.1
    max_steps: int = 128

class StepResponse(BaseModel):
    step: int
    actions: List[int]
    done: bool

class SolveRequest(BaseModel):
    num_agents: int = 4
    size: int = 16
    density: float = 0.1
    max_steps: int = 128

class SolveResponse(BaseModel):
    status: str
    num_steps: int
    success: bool
    paths: List[List[List[int]]]  # Format: [agent_0_path, agent_1_path, ...] where path is [[x0, y0], [x1, y1], ...]