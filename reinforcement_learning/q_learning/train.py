"""
Training script for Q-Learning agent.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import QLearning


class GridWorld:
    """
    Simple GridWorld environment.
    """

    def __init__(self):

        self.size = 5
        self.goal = (4, 4)

    def reset(self):

        self.agent = [0, 0]
        return self.get_state()

    def get_state(self):

        return self.agent[0] * self.size + self.agent[1]

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

        return self.get_state(), reward, done


env = GridWorld()

state_size = env.size * env.size
action_size = 4

agent = QLearning(state_size, action_size)

episodes = 300

reward_history = []


for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(50):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.fit(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward

        if done:
            break

    agent.epsilon = max(0.05, agent.epsilon * 0.995)

    reward_history.append(total_reward)

    print(f"Episode {episode} Reward {total_reward}")


plt.plot(reward_history)

plt.title("Q-Learning Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig("training_rewards.png")

plt.show()


with open("metrics.txt", "w") as f:

    f.write("Q-Learning Training Metrics\n")
    f.write(f"Episodes: {episodes}\n")
    f.write(f"Average Reward: {np.mean(reward_history)}\n")