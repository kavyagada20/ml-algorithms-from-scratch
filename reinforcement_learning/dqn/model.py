"""
Deep Q Network (DQN) implemented from scratch.

Author: Kavya Gada
Repository: ml-algorithms-from-scratch
"""

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.capacity = capacity
        self.buffer = []

    def add(self, experience):

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, batch_size):

        idx = np.random.choice(len(self.buffer), batch_size)

        return [self.buffer[i] for i in idx]

    def size(self):

        return len(self.buffer)


class DQN:

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr

        self.W1 = np.random.randn(state_dim, 24) * 0.1
        self.b1 = np.zeros(24)

        self.W2 = np.random.randn(24, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

        self.loss_history = []

    def forward(self, state):

        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)

        q_values = np.dot(a1, self.W2) + self.b2

        return q_values, a1

    def predict(self, state):

        q_values, _ = self.forward(state)

        return q_values

    def fit(self, states, targets):

        losses = []

        for state, target in zip(states, targets):

            q_values, hidden = self.forward(state)

            loss = np.mean((q_values - target) ** 2)
            losses.append(loss)

            grad = 2 * (q_values - target)

            dW2 = np.outer(hidden, grad)
            db2 = grad

            dhidden = np.dot(self.W2, grad)
            dhidden[hidden <= 0] = 0

            dW1 = np.outer(state, dhidden)
            db1 = dhidden

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        self.loss_history.append(np.mean(losses))