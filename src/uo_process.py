import numpy as np


class UOProcess:
    def __init__(self, x0=1.0, mu=0.0, sigma=0.2, theta=1.0):
        self.theta = theta
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.x = 0
        self.W = np.zeros((2, 2))
        self.t = 0

    def sample(self, dt=1e-2):
        ex = np.exp(-self.theta * (self.t + dt))
        self.W += np.sqrt(np.exp(2 * self.theta * (self.t + dt)) - np.exp(2 * self.theta * self.t)) * \
                  np.random.randn(2, 2) / np.sqrt(2 * self.theta)
        # self.x = self.x0 * ex + self.mu * (1 - ex) + self.sigma * ex * self.W
        self.x = self.mu * (1 - ex) + self.sigma * ex * self.W
        self.t += dt
        return self.x

    def reset(self, sigma):
        self.W = 0
        self.t = 0
        self.x = self.mu
        self.sigma = sigma
