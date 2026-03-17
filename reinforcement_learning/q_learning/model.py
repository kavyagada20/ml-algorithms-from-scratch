"""
Q-Learning implementation from scratch.

Author: Kavya Gada
Repository: ml-algorithms-from-scratch
"""

import numpy as np


class QLearning:
    """
    Q-Learning agent for discrete environments.
    """

    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=1.0):

        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((state_size, action_size))

        self.loss_history = []

    def predict(self, state):
        """
        Return Q-values for a given state.
        """
        return self.q_table[state]

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.argmax(self.q_table[state])

    def fit(self, state, action, reward, next_state, done):

        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        loss = (target - current_q) ** 2
        self.loss_history.append(loss)

        self.q_table[state, action] += self.lr * (target - current_q)