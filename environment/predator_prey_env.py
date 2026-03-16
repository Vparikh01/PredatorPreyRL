import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from environment.agent import Agent

class PredatorPreyEnv:
    def __init__(self, grid_size=15, num_predators=1, num_prey=3, max_steps=50):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.max_steps = max_steps
        self.agents = []
        self.step_count = 0
        self.fig, self.ax = plt.subplots(figsize=(6,6))  # Create figure once
        plt.ion()  # Turn on interactive mode

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agents = []
        self.step_count = 0
        occupied = set()

        # Create predators
        for _ in range(self.num_predators):
            predator = Agent("predator", self.grid_size)
            pos = predator.place_random(occupied)
            occupied.add(pos)
            self.agents.append(predator)
            self.grid[pos] = 1

        # Create prey
        for _ in range(self.num_prey):
            prey = Agent("prey", self.grid_size)
            pos = prey.place_random(occupied)
            occupied.add(pos)
            self.agents.append(prey)
            self.grid[pos] = 2

        return self.grid.copy()

    def step(self, predator_actions=None):
        #predator_actions: list of actions for each predator
        #prey move randomly
        self.grid.fill(0)
        occupied = set()
    
        # Move predators
        for i, agent in enumerate(self.agents):
            if agent.type == "predator":
                action = predator_actions[i] if predator_actions else np.random.randint(0,5)
                agent.move(action)
                occupied.add(agent.position)

        # Move prey
        for agent in self.agents:
            if agent.type == "prey":
                action = np.random.randint(0,5)
                agent.move(action)
                occupied.add(agent.position)

        # Update grid
        for agent in self.agents:
            if agent.type == "predator":
                self.grid[agent.position] = 1
            else:
                self.grid[agent.position] = 2

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.grid.copy(), done

    def render(self):
        self.ax.clear()  # Clear previous frame

        # Show the grid colors
        cmap = ListedColormap(["white", "red", "blue"])
        self.ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=2)

        # Draw borders around each cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
                self.ax.add_patch(rect)

        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        plt.pause(0.1)  # Pause to update display