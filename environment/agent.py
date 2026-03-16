import numpy as np

class Agent:
    def __init__(self, agent_type, grid_size):
        self.type = agent_type
        self.grid_size = grid_size
        self.position = None

    def place_random(self, occupied_positions):
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if (x, y) not in occupied_positions:
                self.position = (x, y)
                return self.position

    def move(self, action):
        # action: 0=up, 1=down, 2=left, 3=right, 4=stay
        x, y = self.position
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        self.position = (x, y)
        return self.position