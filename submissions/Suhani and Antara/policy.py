import numpy as np
class Policy:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon
        self.alpha = np.full(n_arms, 0.5)
        self.beta = np.full(n_arms, 0.5)
        self.t = 0
        self.min_rounds = 5   # show every ad at least 5x before trusting comparisons

    def select_arm(self):
        if self.t < self.n_arms * self.min_rounds:
            return self.t % self.n_arms
        progress = (self.t / self.horizon) ** 0.2
        samples = np.random.beta(self.alpha, self.beta)
        means = self.alpha / (self.alpha + self.beta)
        score = (1 - progress) * samples + progress * means
        return int(np.argmax(score))

    def update(self, arm, reward):
        self.t += 1
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
