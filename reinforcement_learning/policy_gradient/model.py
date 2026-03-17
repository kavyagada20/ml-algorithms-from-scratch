"""
Policy Gradient (REINFORCE) implemented from scratch.

Author: Kavya Gada
Repository: ml-algorithms-from-scratch
"""

import numpy as np


class PolicyGradient:
    """
    Policy Gradient Agent using simple neural network.
    """

    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        # simple neural network
        self.W1 = np.random.randn(state_dim, 16) * 0.1
        self.b1 = np.zeros(16)

        self.W2 = np.random.randn(16, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

        self.loss_history = []

    def softmax(self, x):

        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward(self, state):

        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.tanh(z1)

        logits = np.dot(a1, self.W2) + self.b2

        probs = self.softmax(logits)

        return probs, a1

    def predict(self, state):

        probs, _ = self.forward(state)

        return probs

    def fit(self, states, actions, rewards):

        G = 0
        returns = []

        # compute discounted rewards
        for r in reversed(rewards):

            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        losses = []

        for state, action, G in zip(states, actions, returns):

            probs, hidden = self.forward(state)

            dlog = probs.copy()
            dlog[action] -= 1

            loss = -np.log(probs[action]) * G
            losses.append(loss)

            dlog *= G

            dW2 = np.outer(hidden, dlog)
            db2 = dlog

            dh = np.dot(self.W2, dlog) * (1 - hidden ** 2)

            dW1 = np.outer(state, dh)
            db1 = dh

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        self.loss_history.append(np.mean(losses))