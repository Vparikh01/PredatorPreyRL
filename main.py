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

import matplotlib.pyplot as plt
import numpy as np

from environment.predator_prey_env import PredatorPreyEnv
from rl.q_learning import QLearningAgent

# SETUP
grid_size = 10
env = PredatorPreyEnv(grid_size=grid_size, num_predators=1, num_prey=1, max_steps=50)

agent = QLearningAgent(grid_size=grid_size)

episodes = 5000


# TRAINING LOOP
print("Training Q-learning...")

for ep in range(episodes):
    env.reset()

    for step in range(env.max_steps):

        prey_list = [a for a in env.agents if a.type == "prey"]
        pred_list = [a for a in env.agents if a.type == "predator"]

        if len(prey_list) == 0:
            break

        pred = pred_list[0]
        prey = prey_list[0]

        state = agent.get_state(pred.position, prey.position)

        action = agent.choose_action(state)
        actions = [action]

        _, done = env.step(predator_actions=actions)

        prey_list = [a for a in env.agents if a.type == "prey"]
        pred_list = [a for a in env.agents if a.type == "predator"]

        # terminal
        if len(prey_list) == 0:
            reward = 10
            next_state = state
            agent.update(state, action, reward, next_state)
            break

        pred = pred_list[0]
        prey = prey_list[0]

        next_state = agent.get_state(pred.position, prey.position)

        # REWARD SHAPING
        prev_dist = abs(state[0]) + abs(state[1])
        new_dist = abs(next_state[0]) + abs(next_state[1])

        reward = -1  # step penalty

        if new_dist < prev_dist:
            reward += 0.5
        else:
            reward -= 0.5

        agent.update(state, action, reward, next_state)

        if done:
            break

    agent.decay_epsilon()

    if ep % 500 == 0:
        print(f"Episode {ep}, epsilon={agent.epsilon:.3f}")

print("Training finished.")


# TESTING
policy = agent.get_policy()

plt.ion()
env.reset()

for _ in range(env.max_steps):

    prey_list = [a for a in env.agents if a.type == "prey"]
    pred_list = [a for a in env.agents if a.type == "predator"]

    if len(prey_list) == 0:
        break

    pred = pred_list[0]
    prey = prey_list[0]

    state = agent.get_state(pred.position, prey.position)

    action = policy.get(state, np.random.randint(5))
    actions = [action]

    _, done = env.step(predator_actions=actions)
    env.render()

    if done:
        break

print("Done.")