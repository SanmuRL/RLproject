"""ε-greedy探索策略"""
import random
import torch
import numpy as np

class Epsilon_greedy(object):
    def __init__(self, config):
        self.config = config
        if "exploration_cycle_length" in self.config.keys():
            self.exploration_cycle_length = self.config["exploration_cycle_length"]
        else:
            self.exploration_cycle_length = None

    def choose_action(self, action_info):
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        if turn_off_exploration:
            epsilon = 0
        else:
            epsilon = self.cal_epsilon(action_info)
        if random.random() > epsilon:
            action = torch.argmax(action_values).item()
        else:
            action = np.random.randint(0, action_values.shape[1])
        return action, epsilon

    def cal_epsilon(self, action_info, epsilon=1):
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config["epsilon_decay_denominator"]
        if self.exploration_cycle_length is None:
            epsilon = epsilon / (1 + (episode_number / epsilon_decay_denominator))
        else:
            max_epsilon = 0
            min_epsilon = 0
            # increase = (max_epsilon - min_epsilon) / (self.exploration_cycle_length / 2)
            # circle = [i for i in range(self.exploration_cycle_length / 2)] + [i for i in range(
            #     self.exploration_cycle_length / 2, 0, -1)]
            increase = (max_epsilon - min_epsilon) / self.exploration_cycle_length
            epsilon = max(max_epsilon - episode_number * increase, min_epsilon)
        return epsilon
