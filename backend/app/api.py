import os
import tempfile
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from pogema import AnimationMonitor, AnimationConfig

from app.models import MapfRequest, StepResponse, MAPFSolveRequest, MAPFSolveResponse
from app.mapf.environment import create_env, get_pbs_expert_actions, get_goal_vec

router = APIRouter()

state = {
    "model": None,
    "is_training": True
}


@router.post("/expert-step", response_model=StepResponse)
def get_expert_step(req: MapfRequest):
    env = create_env(req)
    env.reset()
    actions = get_pbs_expert_actions(env)
    return StepResponse(step=1, actions=actions, done=False)


@router.post("/solve", response_model=MAPFSolveResponse)
def solve_mapf(req: MAPFSolveRequest):
    print("--- SOLVE REQUEST RECEIVED ---")
    print(f"Custom map present: {req.custom_map is not None}")
    if req.custom_map is not None:
        print(f"Num agents custom: {len(req.agents or [])}")
        print(f"Num goals custom: {len(req.goals or [])}")
    if state["is_training"]:
        raise HTTPException(
            status_code=400, 
            detail="Model is currently training in background. Please wait..."
        )

    # Validate custom map inputs if sent
    if req.custom_map is not None:
        if not req.agents or not req.goals:
            raise HTTPException(status_code=400, detail="Custom map requires agents and goals.")
        if len(req.agents) != len(req.goals):
            raise HTTPException(
                status_code=400, 
                detail=f"Agent count ({len(req.agents)}) must match Goal count ({len(req.goals)})."
            )

    model = state["model"]
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        anim_cfg = AnimationConfig()
        raw_env = create_env(req)
        env = AnimationMonitor(raw_env, animation_config=anim_cfg)

        obs, _ = env.reset()
        num_agents = env.unwrapped.get_num_agents()
        seq_len = 10
        device = 'cpu'

        # Buffers for CommTransformerNet
        obs_buf = torch.zeros((num_agents, seq_len, 3, 11, 11), device=device)
        goal_buf = torch.zeros((num_agents, seq_len, 2), device=device)

        solved = False
        step = 0

        for step in range(req.max_steps):
            gv = get_goal_vec(env.unwrapped, env.unwrapped.grid_config, device=device)
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)

            # Shift buffer windows
            obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
            obs_buf[:, -1] = obs_t
            goal_buf = torch.roll(goal_buf, shifts=-1, dims=1)
            goal_buf[:, -1] = gv

            with torch.no_grad():
                logits, _, _ = model(obs_buf, goal_buf)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

            obs, rewards, terminated, truncated, _ = env.step(actions)

            if all(terminated):
                solved = True
                break
            if any(truncated):
                break

        # Render SVG animation
        target_path = os.path.join(tmp_dir, "render.svg")
        env.save_animation(target_path)
        
        svg_content = None
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                svg_content = f.read()

    return MAPFSolveResponse(
        success=solved,
        num_steps=step + 1,
        svg_animation=svg_content
    )