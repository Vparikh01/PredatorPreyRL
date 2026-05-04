import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from environment.agent import Agent


class PredatorPreyEnv:
    def __init__(self, grid_size=10, num_predators=1, num_prey=1, max_steps=50):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.max_steps = max_steps
        self.agents = []
        self.step_count = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.ion()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agents = []
        self.step_count = 0
        occupied = set()

        # --- Create predators ---
        for _ in range(self.num_predators):
            predator = Agent("predator", self.grid_size)
            pos = predator.place_random(occupied)
            occupied.add(pos)
            self.agents.append(predator)
            self.grid[pos] = 1

        # --- Create prey ---
        for _ in range(self.num_prey):
            prey = Agent("prey", self.grid_size)
            pos = prey.place_random(occupied)
            occupied.add(pos)
            self.agents.append(prey)
            self.grid[pos] = 2

        return self.grid.copy()

    def step(self, predator_actions=None):
        self.grid.fill(0)

        # --- Move predators ---
        pred_idx = 0
        for agent in self.agents:
            if agent.type == "predator":
                action = (
                    predator_actions[pred_idx]
                    if predator_actions is not None
                    else np.random.randint(0, 5)
                )
                pred_idx += 1
                agent.move(action)

        # --- Move prey (random) ---
        for agent in self.agents:
            if agent.type == "prey":
                action = np.random.randint(0, 5)
                agent.move(action)

        # --- Capture logic ---
        pred_positions = {a.position for a in self.agents if a.type == "predator"}
        prey_agents = [a for a in self.agents if a.type == "prey"]

        captured = [prey for prey in prey_agents if prey.position in pred_positions]

        for prey in captured:
            self.agents.remove(prey)

        # --- Update grid ---
        for agent in self.agents:
            if agent.type == "predator":
                self.grid[agent.position] = 1
            else:
                self.grid[agent.position] = 2

        # --- Done condition ---
        self.step_count += 1
        num_prey_left = len([a for a in self.agents if a.type == "prey"])

        done = (self.step_count >= self.max_steps) or (num_prey_left == 0)

        return self.grid.copy(), done

    def render(self):
        self.ax.clear()

        cmap = ListedColormap(["white", "red", "blue"])
        self.ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=2)

        # grid lines
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1,
                )
                self.ax.add_patch(rect)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        plt.pause(0.1)