import torch
import torch.nn.functional as F
import numpy as np
from app.mapf.environment import get_goal_vec, get_pbs_expert_actions, create_env, CurriculumManager

def run_marl_evaluation(model, env, num_episodes=10, device='cpu'):
    model.eval()
    total_rewards = []
    total_lengths = []
    successes = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        active_cfg = env.unwrapped.grid_config
        num_agents = active_cfg.num_agents

        obs_buf = torch.zeros((num_agents, 10, 3, 11, 11), device=device)
        goal_buf = torch.zeros((num_agents, 10, 2), device=device)

        ep_ret = 0
        ep_len = 0

        for step in range(active_cfg.max_episode_steps):
            gv = get_goal_vec(env.unwrapped, active_cfg, device=device)
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)

            obs_buf = torch.roll(obs_buf, shifts=-1, dims=1); obs_buf[:, -1] = obs_t
            goal_buf = torch.roll(goal_buf, shifts=-1, dims=1); goal_buf[:, -1] = gv

            with torch.no_grad():
                logits, _, _ = model(obs_buf, goal_buf)
                action = torch.argmax(logits, dim=-1)

            obs, rewards, terminated, truncated, _ = env.step(action.cpu().numpy())
            ep_ret += np.mean(rewards)
            ep_len += 1

            if all(terminated):
                successes += 1
                break
            if any(truncated): 
                break

        total_rewards.append(ep_ret)
        total_lengths.append(ep_len)

    model.train()
    return np.mean(total_rewards), np.mean(total_lengths), (successes / num_episodes) * 100.0


def train_a2c(model, episodes=20000, lr=1e-4, gamma=0.99, entropy_coeff=0.05, il_coeff=0.5, seq_len=10, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curriculum = CurriculumManager()
    env = create_env(curriculum.config_map[curriculum.level])

    history = {"episodes": [], "success_rate": [], "loss": [], "level": []}

    print("--- Training Transformer via A2C ---")

    for episode in range(episodes):
        obs, _ = env.reset()
        active_cfg = env.unwrapped.grid_config
        num_agents = active_cfg.num_agents

        obs_buf = torch.zeros((num_agents, seq_len, 3, 11, 11), device=device)
        goal_buf = torch.zeros((num_agents, seq_len, 2), device=device)

        ep_obs_sequences, ep_goal_sequences = [], []
        ep_actions, ep_expert_actions = [], []
        ep_rewards, ep_masks = [], []

        prev_positions = np.array(env.unwrapped.get_agents_xy())

        for step in range(active_cfg.max_episode_steps):
            gv = get_goal_vec(env.unwrapped, active_cfg, device=device)
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)

            obs_buf = torch.roll(obs_buf, shifts=-1, dims=1); obs_buf[:, -1] = obs_t
            goal_buf = torch.roll(goal_buf, shifts=-1, dims=1); goal_buf[:, -1] = gv

            expert_acts = get_pbs_expert_actions(env)

            with torch.no_grad():
                logits, val, _ = model(obs_buf, goal_buf)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            next_obs, base_rewards, terminated, truncated, _ = env.step(action.cpu().numpy())
            curr_positions = np.array(env.unwrapped.get_agents_xy())

            shaped_rewards = np.zeros(num_agents, dtype=np.float32)

            for i in range(num_agents):
                if base_rewards[i] > 0.5:
                    shaped_rewards[i] += 2.0
                else:
                    if action[i] > 0 and np.array_equal(curr_positions[i], prev_positions[i]):
                        shaped_rewards[i] -= 0.05

            ep_obs_sequences.append(obs_buf.clone())
            ep_goal_sequences.append(goal_buf.clone())
            ep_actions.append(action)
            ep_expert_actions.append(torch.tensor(expert_acts, dtype=torch.long, device=device))
            ep_rewards.append(torch.tensor(shaped_rewards, device=device))
            ep_masks.append(torch.tensor(1.0 - np.array(terminated, dtype=np.float32), device=device))

            obs = next_obs
            prev_positions = curr_positions
            if all(terminated) or any(truncated):
                break

        rollout_len = len(ep_actions)
        if rollout_len < 2:
            continue

        b_obs = torch.cat(ep_obs_sequences, dim=0)
        b_goals = torch.cat(ep_goal_sequences, dim=0)
        b_actions = torch.cat(ep_actions, dim=0)
        b_expert_actions = torch.cat(ep_expert_actions, dim=0)

        returns = []
        G = torch.zeros(num_agents, device=device)
        for r, m in zip(reversed(ep_rewards), reversed(ep_masks)):
            G = r + gamma * G * m
            returns.insert(0, G.clone())
        b_returns = torch.cat(returns).detach()

        new_logits, new_values, _ = model(b_obs, b_goals)
        new_values = new_values.flatten()

        new_dist = torch.distributions.Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(b_actions)

        b_advantages = (b_returns - new_values).detach()
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        actor_loss = -(new_log_probs * b_advantages).mean()
        critic_loss = F.mse_loss(new_values, b_returns)
        entropy_loss = new_dist.entropy().mean()

        il_loss = F.cross_entropy(new_logits, b_expert_actions)

        total_loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy_loss + il_coeff * il_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if episode % 100 == 0:
            avg_ret, avg_len, sr = run_marl_evaluation(model, env, num_episodes=10, device=device)
            history["episodes"].append(episode)
            history["success_rate"].append(sr)
            history["loss"].append(total_loss.item())
            history["level"].append(curriculum.level)

            print(f"Ep {episode:04d} | SR: {sr:>3.1f}% | Loss: {total_loss.item():.4f} | Lvl: {curriculum.level}")

            # Check consecutive streak condition before advancing
            if curriculum.update(sr):
                env = create_env(curriculum.config_map[curriculum.level])

    return history