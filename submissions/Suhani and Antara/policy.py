import numpy as np

class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon

        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

        self.t = 0

    def select_arm(self):

        if self.t < self.n_arms:
            return self.t

        progress = (self.t / self.horizon) ** 1.5
        samples = np.random.beta(self.alpha, self.beta)
        means = (self.alpha - 1) / (self.alpha + self.beta - 2 + 1e-9)
        score = (1-progress)*samples + progress*means
        return int(np.argmax(score))

    def update(self, arm, reward):

        self.t += 1

        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
