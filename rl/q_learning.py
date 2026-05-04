import numpy as np


class QLearningAgent:
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.grid_size = grid_size
        self.num_actions = 5

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: (dx, dy) -> action values
        self.Q = {}

    # STATE ENCODING (IMPORTANT FIX)
    def get_state(self, pred_pos, prey_pos):
        dx = prey_pos[0] - pred_pos[0]
        dy = prey_pos[1] - pred_pos[1]

        # clip to avoid explosion of state space
        dx = max(-self.grid_size, min(self.grid_size, dx))
        dy = max(-self.grid_size, min(self.grid_size, dy))

        return (dx, dy)

    # INIT STATE IF NEW
    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)

    # ACTION SELECTION
    def choose_action(self, state):
        self._ensure_state(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        return np.argmax(self.Q[state])

    # Q UPDATE
    def update(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        best_next = np.max(self.Q[next_state])

        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )

    # POLICY EXTRACTION
    def get_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy

    # EPSILON DECAY
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)