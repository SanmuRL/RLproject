import numpy as np
import random
import copy

class OUnoise(object):
    def __init__(self, dim, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(dim)
        self.seed = random.seed(seed)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.states = copy.copy(self.mu)

    def noise(self):
        x = self.states
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.states = x + dx
        return self.states
