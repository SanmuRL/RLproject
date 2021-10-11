"""DDQN算法实现"""

import torch
import random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utilities.action_strategies.Epsilon_Greedy import Epsilon_greedy
from Utilities.Replay_Buffer.basic_buffer import basic_buffer
from Model.Mario_model_DQN import Q_Network
from Model.Mario_model2 import MarioNet
import datetime

class DDQN(object):
    def __init__(self, config):
        self.config = config
        self.gamma = self.config["gamma"]
        self.step = self.config["step"]
        self.capacity = self.config["capacity"]
        self.batch_size = self.config["batch_size"]
        self.img_stack = self.config["img_stack"]
        self.action_dim = self.config["action_dim"]
        self.save_path_local = self.config["save_path_local"]
        self.save_path_target = self.config["save_path_target"]
        self.useExit = self.config["useExit"]
        #self.mse_loss = torch.nn.MSELoss()
        self.mse_loss = torch.nn.SmoothL1Loss()
        self.replay_buffer = basic_buffer(self.batch_size, self.capacity)
        self.exploration_strategy = Epsilon_greedy(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_local_network = Q_Network(self.img_stack, self.action_dim).to(self.device)
        self.q_target_network = Q_Network(self.img_stack, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_local_network.parameters(),
                                    lr=self.step)
        if self.useExit:
            self.q_local_network.load_state_dict(torch.load(self.save_path_local + '2021-09-30T21-23-35.pth')['net'])
            self.q_target_network.load_state_dict(torch.load(self.save_path_target + '2021-09-30T21-23-35.pth')['net'])
            self.optimizer.load_state_dict(torch.load(self.save_path_local + '2021-09-30T21-23-35.pth')['optimizer'])


    def update_local_network(self):
        states, actions, rewards, next_states, dones = self.sample_experiences()
        with torch.no_grad():
            target_Q_value = self.cal_target_value(rewards, next_states, dones)
        local_Q_value = self.cal_local_value(states, actions)
        self.q_local_network.train(mode=True)
        self.optimizer.zero_grad()
        loss = self.mse_loss(local_Q_value, target_Q_value)
        loss.backward()
        self.optimizer.step()
        return local_Q_value.mean().item(), loss.item()

    def cal_local_value(self, states, actions):
        local_Q_value = self.q_local_network(states).gather(1, actions.long())
        #print(local_Q_value)
        return local_Q_value

    def cal_target_value(self, rewards, next_states, dones):
        target_Q_value_next = self.q_local_network(next_states).detach().max(1)[0].unsqueeze(1)   #和DQN的唯一区别
        target_Q_value = (rewards + self.gamma * target_Q_value_next * (1-dones.float())).float()
        #print(target_Q_value)
        return target_Q_value

    def choose_action(self, state, episode_number):
        state = np.array([screen for screen in list(state)])
        state = torch.from_numpy(state).float().to(self.device)
        self.q_local_network.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)
            action_values = self.q_local_network(state)
        self.q_local_network.train()
        action = self.exploration_strategy.choose_action({"action_values": action_values,
                                                         "turn_off_exploration": self.config["turn_off_exploration"],
                                                          "episode_number": episode_number})
        return action

    def locally_save_policy(self):
        state_local = {'net': self.q_local_network.state_dict(), 'optimizer': self.optimizer.state_dict()}
        state_target = {'net': self.q_target_network.state_dict()}
        path_local_cur = self.save_path_local + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        path_target_cur = self.save_path_target + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        torch.save(state_local, path_local_cur)
        torch.save(state_target, path_target_cur)

    def sample_experiences(self):
        states, next_states, actions, rewards, dones = self.replay_buffer.sample()
        return states, actions, rewards, next_states, dones

    def soft_target_update(self, tau):
        for target_param, local_param in zip(self.q_target_network.parameters(), self.q_local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def hard_target_update(self):
        for target_param, local_param in zip(self.q_target_network.parameters(), self.q_local_network.parameters()):
            target_param.data.copy_(local_param.data)