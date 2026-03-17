"""
Training script for Deep Q Network agent
"""

import numpy as np
import matplotlib.pyplot as plt

from model import DQN, ReplayBuffer


class GridWorld:

    def __init__(self):

        self.size = 5
        self.goal = (4, 4)

    def reset(self):

        self.agent = [0, 0]
        return np.array(self.agent) / self.size

    def step(self, action):

        if action == 0:
            self.agent[0] = max(0, self.agent[0] - 1)
        elif action == 1:
            self.agent[0] = min(self.size - 1, self.agent[0] + 1)
        elif action == 2:
            self.agent[1] = max(0, self.agent[1] - 1)
        elif action == 3:
            self.agent[1] = min(self.size - 1, self.agent[1] + 1)

        reward = -1
        done = False

        if tuple(self.agent) == self.goal:
            reward = 10
            done = True

        return np.array(self.agent) / self.size, reward, done


env = GridWorld()

state_dim = 2
action_dim = 4

model = DQN(state_dim, action_dim)
buffer = ReplayBuffer()

episodes = 200

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

reward_history = []


for episode in range(episodes):

    state = env.reset()
    total_reward = 0

    for step in range(50):

        if np.random.rand() < epsilon:
            action = np.random.randint(action_dim)
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values)

        next_state, reward, done = env.step(action)

        buffer.add((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if buffer.size() > 32:

            batch = buffer.sample(32)

            states = []
            targets = []

            for s, a, r, ns, d in batch:

                q = model.predict(s)

                if d:
                    q[a] = r
                else:
                    q_next = model.predict(ns)
                    q[a] = r + model.gamma * np.max(q_next)

                states.append(s)
                targets.append(q)

            model.fit(states, targets)

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    reward_history.append(total_reward)

    print(f"Episode {episode} Reward {total_reward}")


plt.plot(reward_history)

plt.title("DQN Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig("training_rewards.png")

plt.show()


with open("metrics.txt", "w") as f:

    f.write("DQN Training Metrics\n")
    f.write(f"Episodes: {episodes}\n")
    f.write(f"Average Reward: {np.mean(reward_history)}\n")