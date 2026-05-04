import numpy as np


def get_next_states(state, action, grid_size):
    px, py, tx, ty = state

    def move(x, y, action):
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < grid_size - 1:
            y += 1
        return x, y

    new_px, new_py = move(px, py, action)

    outcomes = []

    for prey_action in range(5):
        new_tx, new_ty = move(tx, ty, prey_action)

        next_state = (new_px, new_py, new_tx, new_ty)

        if (new_px, new_py) == (new_tx, new_ty):
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        prob = 1 / 5

        outcomes.append((prob, next_state, reward, done))

    return outcomes

grid_size = 10
states = []

for px in range(grid_size):
    for py in range(grid_size):
        for tx in range(grid_size):
            for ty in range(grid_size):
                states.append((px, py, tx, ty))


V = {s: 0 for s in states}
policy = {s: np.random.randint(0, 5) for s in states}

if __name__ == "__main__":
    state = (5, 5, 7, 7)
    outcomes = get_next_states(state, action=0, grid_size=10)

    for o in outcomes:
        print(o)

        gamma = 0.9

def policy_evaluation(V, policy, states, grid_size, gamma=0.9, theta=1e-4):
    while True:
        delta = 0

        for s in states:
            v = V[s]
            a = policy[s]

            new_v = 0

            outcomes = get_next_states(s, a, grid_size)

            for prob, next_state, reward, done in outcomes:
                if done:
                    new_v += prob * reward
                else:
                    new_v += prob * (reward + gamma * V[next_state])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        if delta < theta:
            break

    return V

def policy_improvement(V, policy, states, grid_size, gamma = 0.9):
    policy_stable = True

    for s in states:
        old_action = policy[s]

        action_values = []

        for a in range(5):  # 5 possible actions
            q = 0

            outcomes = get_next_states(s, a, grid_size)

            for prob, next_state, reward, done in outcomes:
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_state])

            action_values.append(q)

        best_action = np.argmax(action_values)
        policy[s] = best_action

        if old_action != best_action:
            policy_stable = False

    return policy, policy_stable

def policy_iteration(states, grid_size):
    V = {s: 0 for s in states}
    policy = {s: np.random.randint(0, 5) for s in states}

    while True:
        V = policy_evaluation(V, policy, states, grid_size, gamma=0.9, theta=1e-4)

        policy, stable = policy_improvement(V, policy, states, grid_size)

        if stable:
            break

    return policy, V


if __name__ == "__main__":
    optimal_policy, optimal_V = policy_iteration(states, grid_size)

    test_state = (5, 5, 7, 7)

    print("Optimal action:", optimal_policy[test_state])
    print("State value:", optimal_V[test_state])