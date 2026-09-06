import heapq
import torch
import numpy as np
import pogema
from pogema import GridConfig


def create_env(config_input):
    """
    Unified environment factory supporting:
    1. Custom map parameters (custom_map, agents, goals)
    2. Request objects (MAPFSolveRequest, MapfRequest)
    3. Dictionary configurations
    """
    def get_val(key, default=None):
        if isinstance(config_input, dict):
            return config_input.get(key, default)
        return getattr(config_input, key, default)

    custom_map = get_val("custom_map", None)
    agents = get_val("agents", None)
    goals = get_val("goals", None)

    # -------------------------------------------------------------
    # CUSTOM MAP MODE
    # -------------------------------------------------------------
    if custom_map is not None and agents and goals:
        # Convert custom 2D matrix (0 = empty, 1 = obstacle) to Pogema map string format
        # Pogema uses '.' for free space and '#' for obstacles
        map_lines = []
        for row in custom_map:
            line = "".join("#" if cell == 1 else "." for cell in row)
            map_lines.append(line)
        
        map_str = "\n".join(map_lines)

        starts = [(a.x, a.y) if hasattr(a, 'x') else tuple(a) for a in agents]
        targets = [(g.x, g.y) if hasattr(g, 'x') else tuple(g) for g in goals]

        grid_cfg = GridConfig(
            map=map_str,                # Formatted string: "...\n.#.\n..."
            agents_xy=starts,
            targets_xy=targets,
            num_agents=len(starts),
            obs_radius=5,
            max_episode_steps=get_val("max_steps", 128),
            integration_type="pogema",
        )

    # -------------------------------------------------------------
    # RANDOM MAP MODE
    # -------------------------------------------------------------
    else:
        num_agents = get_val("num_agents", 4)
        size = get_val("size", 16)
        density = get_val("density", 0.1)
        max_steps = get_val("max_steps", get_val("max_episode_steps", 128))

        grid_cfg = GridConfig(
            num_agents=num_agents,
            size=size,
            density=density,
            obs_radius=5,
            seed=None,
            max_episode_steps=max_steps,
            integration_type="pogema",
        )

    return pogema.pogema_v0(grid_config=grid_cfg)


class CurriculumManager:
    def __init__(self):
        self.level = 1
        self.consecutive_successes = 0
        self.required_consecutive = 2
        self.target_sr = 80.0
        
        self.config_map = {
            1: {"num_agents": 1, "size": 8,  "density": 0.0, "max_episode_steps": 256},
            2: {"num_agents": 2, "size": 12, "density": 0.0, "max_episode_steps": 128},
            3: {"num_agents": 4, "size": 16, "density": 0.0, "max_episode_steps": 128},
            4: {"num_agents": 8, "size": 24, "density": 0.0, "max_episode_steps": 256},
            5: {"num_agents": 16,"size": 32, "density": 0.0, "max_episode_steps": 256},
            6: {"num_agents": 32,"size": 64, "density": 0.0, "max_episode_steps": 256},
            7: {"num_agents": 2, "size": 8,  "density": 0.1, "max_episode_steps": 256},
            8: {"num_agents": 4, "size": 16, "density": 0.1, "max_episode_steps": 256},
            9: {"num_agents": 8, "size": 24, "density": 0.1, "max_episode_steps": 256},
            10: {"num_agents": 2,"size": 12, "density": 0.2, "max_episode_steps": 256},
            11: {"num_agents": 4,"size": 16, "density": 0.2, "max_episode_steps": 256},
            12: {"num_agents": 8,"size": 24, "density": 0.2, "max_episode_steps": 256},
            13: {"num_agents": 2,"size": 12, "density": 0.3, "max_episode_steps": 256},
        }

    def update(self, success_rate: float) -> bool:
        if success_rate >= self.target_sr:
            self.consecutive_successes += 1
            print(f"--> Target hit! Streak: {self.consecutive_successes}/{self.required_consecutive}")
        else:
            self.consecutive_successes = 0

        if self.consecutive_successes >= self.required_consecutive:
            if self.level + 1 in self.config_map:
                self.level += 1
                self.consecutive_successes = 0
                print(f"*** ADVANCING TO CURRICULUM LEVEL {self.level} ***")
                return True
        return False


def get_goal_vec(env, current_cfg, device='cpu'):
    agents = np.array(env.get_agents_xy())
    targets = np.array(env.get_targets_xy())
    
    # Safely get grid size regardless of custom or random grid initialization
    grid_size = getattr(current_cfg, 'size', None)
    if grid_size is None and hasattr(current_cfg, 'map') and current_cfg.map is not None:
        grid_size = max(current_cfg.map.shape)
    if grid_size is None:
        grid_size = 16

    vec = (targets - agents) / grid_size
    return torch.tensor(vec, dtype=torch.float32, device=device)


def get_space_time_astar(grid, start, goal, reserved, max_t=30):
    rows, cols = len(grid), len(grid[0])
    moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    start_state = (0 + abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start[0], start[1])
    pq = [start_state]

    came_from = {}
    g_score = {(start[0], start[1], 0): 0}

    while pq:
        f, t, x, y = heapq.heappop(pq)

        if (x, y) == goal or t >= max_t:
            path = []
            curr = (x, y, t)
            while curr in came_from:
                path.append((curr[0], curr[1]))
                curr = came_from[curr]
            path.append(start)
            return path[::-1]

        for i, (dx, dy) in enumerate(moves):
            nx, ny, nt = x + dx, y + dy, t + 1

            if 0 <= nx < rows and 0 <= ny < cols and not grid[nx][ny]:
                if (nx, ny, nt) in reserved: continue
                if (nx, ny, t) in reserved and (x, y, nt) in reserved: continue

                if (nx, ny, nt) not in g_score or g_score[(nx, ny, nt)] > nt:
                    g_score[(nx, ny, nt)] = nt
                    h = abs(nx-goal[0]) + abs(ny-goal[1])
                    heapq.heappush(pq, (nt + h, nt, nx, ny))
                    came_from[(nx, ny, nt)] = (x, y, t)

    return [(start[0], start[1])] * 2


def get_pbs_expert_actions(env):
    core_env = env.unwrapped
    grid = core_env.grid.get_obstacles()
    agents_xy = core_env.get_agents_xy()
    targets_xy = core_env.get_targets_xy()

    num_agents = len(agents_xy)
    reserved = set()
    expert_actions = []

    for i in range(num_agents):
        start = agents_xy[i]
        goal = targets_xy[i]

        path = get_space_time_astar(grid, start, goal, reserved)

        if len(path) > 1:
            next_step = path[1]
            dx, dy = next_step[0] - start[0], next_step[1] - start[1]
            action_map = {(0,0):0, (-1,0):1, (1,0):2, (0,-1):3, (0,1):4}
            expert_actions.append(action_map.get((dx, dy), 0))
        else:
            expert_actions.append(0)

        for t, (px, py) in enumerate(path):
            reserved.add((px, py, t))
            if (px, py) == goal:
                for future_t in range(t, 31):
                    reserved.add((px, py, future_t))
    return expert_actions