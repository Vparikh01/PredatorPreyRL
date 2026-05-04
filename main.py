# import matplotlib.pyplot as plt
# from environment.predator_prey_env import PredatorPreyEnv
# from rl.policy_iteration import policy_iteration, states, grid_size
# import numpy as np

# # --- Compute policy FIRST ---
# print("Running policy iteration...")
# optimal_policy, optimal_V = policy_iteration(states, grid_size)
# print("Policy computed.")

# # --- Create environment (MUST match grid_size) ---
# env = PredatorPreyEnv(grid_size=grid_size, num_predators=1, num_prey=1, max_steps=50)

# plt.ion()
# env.reset()

# # --- Run simulation ---
# for _ in range(env.max_steps):

#     # Extract current state
#     pred = [a for a in env.agents if a.type == "predator"][0]
#     prey = [a for a in env.agents if a.type == "prey"][0]

#     state = (*pred.position, *prey.position)

#     # Safety check (optional but helpful)
#     if state not in optimal_policy:
#         print("State not found in policy:", state)
#         action = np.random.randint(0, 5)
#     else:
#         action = optimal_policy[state]

#     actions = [action]

#     # Step environment
#     grid, done = env.step(predator_actions=actions)
#     env.render()

#     if done:
#         break

# print("Episode finished.")

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from environment.predator_prey_env import PredatorPreyEnv
from rl.q_learning import QLearningAgent

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GRID_SIZE   = 10
MAX_STEPS   = 50
EPISODES    = 5000
GIF_DIR     = "gifs"
GIF_PATH    = os.path.join(GIF_DIR, "episode.gif")
GIF_FPS     = 5

os.makedirs(GIF_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
env   = PredatorPreyEnv(grid_size=GRID_SIZE, num_predators=1, num_prey=1, max_steps=MAX_STEPS)
agent = QLearningAgent(grid_size=GRID_SIZE)

print("Training Q-learning...")

for ep in range(EPISODES):
    env.reset()

    for step in range(env.max_steps):
        pred_list = [a for a in env.agents if a.type == "predator"]
        prey_list = [a for a in env.agents if a.type == "prey"]

        if not prey_list:
            break

        pred  = pred_list[0]
        prey  = prey_list[0]
        state = agent.get_state(pred.position, prey.position)

        action = agent.choose_action(state)
        _, done = env.step(predator_actions=[action])

        pred_list = [a for a in env.agents if a.type == "predator"]
        prey_list = [a for a in env.agents if a.type == "prey"]

        # Prey was caught this step
        if not prey_list:
            agent.update(state, action, reward=10, next_state=state)
            break

        pred       = pred_list[0]
        prey       = prey_list[0]
        next_state = agent.get_state(pred.position, prey.position)

        # Reward shaping: encourage closing distance
        prev_dist = abs(state[0])      + abs(state[1])
        new_dist  = abs(next_state[0]) + abs(next_state[1])

        reward = -1 + (0.5 if new_dist < prev_dist else -0.5)

        agent.update(state, action, reward, next_state)

        if done:
            break

    agent.decay_epsilon()

    if ep % 500 == 0:
        print(f"  Episode {ep:>5} | epsilon = {agent.epsilon:.3f}")

print("Training finished.\n")

def capture_frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())

    return buf[..., :3].copy()


def run_and_save_gif(env, agent, path: str, fps: int = 5) -> None:

    policy = agent.get_policy()
    frames = []

    plt.ioff()
    env.reset()

    for _ in range(env.max_steps):

        pred_list = [a for a in env.agents if a.type == "predator"]
        prey_list = [a for a in env.agents if a.type == "prey"]

        if not prey_list:
            break

        pred = pred_list[0]
        prey = prey_list[0]

        state = agent.get_state(pred.position, prey.position)

        action = policy.get(state, np.random.randint(5))
        _, done = env.step(predator_actions=[action])

        env.render()

        frames.append(capture_frame(env.fig))

        if done or not [a for a in env.agents if a.type == "prey"]:
            for _ in range(3):
                frames.append(frames[-1])
            break

    plt.close("all")

    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF saved → {path} ({len(frames)} frames)")


# run_and_save_gif(env, agent, path=GIF_PATH, fps=GIF_FPS)