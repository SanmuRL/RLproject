"""
将图片进行下采样以及重复动作设置
"""

import gym
from gym.spaces import Box
from skimage import transform
import torch
import numpy as np

class ResizeShape(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self.size = size
        if isinstance(self.size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.observation_space = Box(low=0, high=255, shape=self.size, dtype=np.uint8)

    def observation(self, observation):
        obs_after = transform.resize(observation, self.size)
        obs_after *= 255
        obs_after = obs_after.astype(np.uint8)
        return obs_after

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