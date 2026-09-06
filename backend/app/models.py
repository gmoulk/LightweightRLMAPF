from typing import List, Optional
from pydantic import BaseModel, Field

class MapfRequest(BaseModel):
    num_agents: int = 4
    size: int = 16
    density: float = 0.1
    max_steps: int = 128

class StepResponse(BaseModel):
    step: int
    actions: List[int]
    done: bool

class AgentPos(BaseModel):
    x: int
    y: int


class MAPFSolveRequest(BaseModel):
    # Random Mode Parameters
    num_agents: Optional[int] = Field(default=4, ge=1)
    size: Optional[int] = Field(default=16, ge=4, le=64)
    density: Optional[float] = Field(default=0.1, ge=0.0, le=0.5)
    
    # Shared Parameter
    max_steps: int = Field(default=128, ge=10, le=1000)

    # Custom Mode Parameters
    custom_map: Optional[List[List[int]]] = None  # 0: empty, 1: obstacle
    agents: Optional[List[AgentPos]] = None       # [{x: r, y: c}]
    goals: Optional[List[AgentPos]] = None        # [{x: r, y: c}]


class MAPFSolveResponse(BaseModel):
    success: bool
    num_steps: int
    svg_animation: Optional[str] = None