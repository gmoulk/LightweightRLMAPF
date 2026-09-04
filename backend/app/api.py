import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from app.models import MapfRequest, StepResponse, SolveRequest, SolveResponse
from app.mapf.environment import create_env, get_pbs_expert_actions, get_goal_vec

router = APIRouter()

# Shared state reference injected from main.py
state = {
    "model": None,
    "is_training": True
}

@router.post("/expert-step", response_model=StepResponse)
def get_expert_step(req: MapfRequest):
    env_config = {
        "num_agents": req.num_agents,
        "size": req.size,
        "density": req.density,
        "max_episode_steps": req.max_steps
    }
    env = create_env(env_config)
    env.reset()
    actions = get_pbs_expert_actions(env)
    return StepResponse(step=1, actions=actions, done=False)

@router.post("/solve", response_model=SolveResponse)
def solve_mapf(req: SolveRequest):
    if state["is_training"]:
        raise HTTPException(
            status_code=400, 
            detail="Model is still training. Please wait until training completes before requesting a solution."
        )

    model = state["model"]
    model.eval()

    env_config = {
        "num_agents": req.num_agents,
        "size": req.size,
        "density": req.density,
        "max_episode_steps": req.max_steps
    }
    env = create_env(env_config)
    obs, _ = env.reset()
    
    num_agents = req.num_agents
    seq_len = 10
    device = 'cpu'

    obs_buf = torch.zeros((num_agents, seq_len, 3, 11, 11), device=device)
    goal_buf = torch.zeros((num_agents, seq_len, 2), device=device)

    # Track positions over time for each agent
    paths = [ [list(pos)] for pos in env.unwrapped.get_agents_xy() ]

    solved = False
    step = 0

    for step in range(req.max_steps):
        gv = get_goal_vec(env.unwrapped, env.unwrapped.grid_config, device=device)
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)

        obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
        obs_buf[:, -1] = obs_t
        goal_buf = torch.roll(goal_buf, shifts=-1, dims=1)
        goal_buf[:, -1] = gv

        with torch.no_grad():
            logits, _, _ = model(obs_buf, goal_buf)
            actions = torch.argmax(logits, dim=-1).cpu().numpy()

        obs, rewards, terminated, truncated, _ = env.step(actions)

        # Record current agent coordinates
        curr_positions = env.unwrapped.get_agents_xy()
        for idx, pos in enumerate(curr_positions):
            paths[idx].append(list(pos))

        if all(terminated):
            solved = True
            break
        if any(truncated):
            break

    return SolveResponse(
        status="completed",
        num_steps=step + 1,
        success=solved,
        paths=paths
    )