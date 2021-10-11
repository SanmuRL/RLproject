import gym
from gym.spaces import Box
import numpy as np

class RepeatAction(gym.Wrapper):
    def __init__(self, env, times):
        super().__init__(env)
        self.times = times

    def step(self, action):
        sum_reward = 0.0
        for i in range(self.times):
            obs, reward, done, info = self.env.step(action)
            sum_reward += reward
            if done:
                break
        return obs, sum_reward, done, info