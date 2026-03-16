import matplotlib.pyplot as plt
from environment.predator_prey_env import PredatorPreyEnv
import numpy as np

plt.ion()
env = PredatorPreyEnv(grid_size=15, num_predators=1, num_prey=3, max_steps=50)
env.reset()
plt.figure(figsize=(6,6))
for _ in range(env.max_steps):
    # Random actions for predators for now
    actions = np.random.randint(0,5, size=env.num_predators)
    grid, done = env.step(predator_actions=actions)
    env.render()
    if done:
        break

plt.ioff()
plt.show()