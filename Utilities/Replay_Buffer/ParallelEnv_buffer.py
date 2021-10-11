import torch
import numpy as np
import random
from torch.utils.data import BatchSampler, SubsetRandomSampler
from collections import namedtuple, deque

class Parallel_buffer(object):
    def __init__(self, num_steps, observation_space, num_processes):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs = torch.ones(num_steps+1, num_processes, *observation_space).to(self.device)
        self.action_probs = torch.ones(num_steps, num_processes, 1).to(self.device)
        self.action = torch.ones(num_steps, num_processes, 1).to(self.device)
        self.rewards = torch.ones(num_steps, num_processes, 1).to(self.device)
        self.sum_rewards = torch.ones(num_steps + 1, num_processes, 1).to(self.device)
        self.values = torch.ones(num_steps+1, num_processes, 1).to(self.device)
        self.dones = torch.ones(num_steps+1, num_processes, 1).to(self.device)
        self.returns = torch.ones(num_steps+1, num_processes, 1).to(self.device)
        self.step = 0

    def add_experience(self, state, action, action_prob, reward, done, value):
        self.obs[self.step + 1].copy_(state)
        self.action[self.step].copy_(action)
        self.action_probs[self.step].copy_(action_prob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step + 1].copy_(done)
        self.values[self.step].copy_(value)

        self.step = (self.step + 1) % self.num_steps

    def next_episode(self):
        self.obs[0].copy_(self.obs[-1])
        self.dones[0].copy_(self.dones[-1])
        self.returns[0].copy_(self.returns[-1])

    def calculate_returns(self, last_values, gamma):
        self.returns[-1].copy_(last_values)
        self.sum_rewards[-1].copy_(last_values)
        sum_episodes = 1e-2
        sum_rewards = 0
        for i in reversed(range(self.action.shape[0])):
            for j in range(self.num_processes):
                if self.dones[i+1][j].item() == 0:
                    sum_rewards += self.sum_rewards[i+1][j].item()
                    sum_episodes += 1
            self.returns[i] = self.returns[i+1] * self.dones[i+1] * gamma + self.rewards[i]
            self.sum_rewards[i] = self.sum_rewards[i+1] * self.dones[i+1] + self.rewards[i]
        return sum_rewards / sum_episodes


    def sample(self, advantages, learning_cycle):
        data_nums = self.num_steps * self.num_processes
        batch_size = data_nums // learning_cycle
        data_index = BatchSampler(SubsetRandomSampler(range(data_nums)), batch_size=batch_size, drop_last=False)
        for index in data_index:
            obs = self.obs[:-1].view(-1, *self.obs.shape[2:])[index]
            action_prob = self.action_probs.view(-1, 1)[index]
            action = self.action.view(-1, 1)[index]
            value = self.values.view(-1, 1)[index]
            advantage = advantages.view(-1, 1)[index]
            returns = self.returns.view(-1, 1)[index]
            yield obs, action_prob, action, value, advantage, returns
