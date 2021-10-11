"""
BipedalWalker-v3
observation_space (-inf, inf) dim: 24
action_space (-1, 1) dim: 4
"""
from Model.BipedalWalker_mode_DDPG import Critic, Actor
import torch
import datetime
import random
import torch.optim as optim
import numpy as np
import gym
from Utilities.Replay_Buffer.basic_buffer import basic_buffer
from Utilities.action_strategies.Epsilon_Greedy import Epsilon_greedy
from Utilities.action_strategies.OUnoise import OUnoise

class DDPG(object):
    def __init__(self, config):
        self.config = config
        self.gamma = self.config["gamma"]
        self.step = self.config["step"]
        self.capacity = self.config["capacity"]
        self.batch_size = self.config["batch_size"]
        self.observation_dim = self.config["observation_dim"]
        self.action_dim = self.config["action_dim"]
        self.save_Q_local = self.config["save_Q_local"]
        self.save_Q_target = self.config["save_Q_target"]
        self.save_Pi_local = self.config["save_Pi_local"]
        self.save_Pi_target = self.config["save_Pi_target"]
        self.useExit = self.config["useExit"]
        self.epsilon = self.config["epsilon"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.OUnoise = OUnoise(self.action_dim, 1)
        self.mse_loss = torch.nn.SmoothL1Loss()
        self.replay_buffer = basic_buffer(self.batch_size, self.capacity)
        self.exploration_strategy = Epsilon_greedy(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_local_network = Critic(self.observation_dim, self.action_dim).to(self.device)
        self.q_target_network = Critic(self.observation_dim, self.action_dim).to(self.device)
        self.Pi_local_network = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.Pi_target_network = Actor(self.observation_dim, self.action_dim).to(self.device)
        self.optim_critic = optim.Adam(self.q_local_network.parameters(), lr=self.step)
        self.optim_actor = optim.Adam(self.Pi_local_network.parameters(), lr=self.step)

    def choose_action(self, observation, needNoise=True):
        observation = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action = self.Pi_local_network(observation).squeeze(0)
        if needNoise:
            noise = self.OUnoise.noise()
            action = action.cpu().numpy()
            action = action + self.epsilon * noise
            return np.clip(action, -1, 1), self.epsilon
        else:
            return action.cpu().numpy(), self.epsilon

    def update(self):
        states, next_states, actions, rewards, dones = self.replay_buffer.sample()
        with torch.no_grad():
            target_Q = self.calculate_target_Q(next_states, rewards, dones)
        local_Q = self.q_local_network(states, actions)
        loss_Q = self.mse_loss(local_Q, target_Q)
        self.optim_critic.zero_grad()
        loss_Q.backward()
        self.optim_critic.step()

        action_pred = self.Pi_local_network(states)
        loss_actor = -self.q_local_network(states, action_pred).mean()
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        self.soft_target_update(0.01)
        return local_Q.mean().item(), loss_Q.item(), loss_actor.item()


    def calculate_target_Q(self, next_states, rewards, dones):
        next_actions = self.Pi_target_network(next_states)
        next_Q = self.q_target_network(next_states, next_actions)
        target_Q = rewards + self.gamma * (1 - dones.float()) * next_Q
        return target_Q

    def updateNoise(self):
        self.epsilon -= self.epsilon_decay
        self.OUnoise.reset()

    def soft_target_update(self, tau):
        for target_param, local_param in zip(self.q_target_network.parameters(), self.q_local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.Pi_target_network.parameters(), self.Pi_local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def locally_save_policy(self):
        stateQ_local = {'net': self.q_local_network.state_dict(), 'optimizer': self.optim_critic.state_dict()}
        stateQ_target = {'net': self.q_target_network.state_dict()}
        path_localQ_cur = self.save_Q_local + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        path_targetQ_cur = self.save_Q_local + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        torch.save(stateQ_local, path_localQ_cur)
        torch.save(stateQ_target, path_targetQ_cur)
        stateActor_local = {'net': self.Pi_local_network.state_dict(), 'optimizer': self.optim_actor.state_dict()}
        stateActor_target = {'net': self.Pi_target_network.state_dict()}
        path_localActor_cur = self.save_Pi_local + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        path_targetActor_cur = self.save_Pi_target + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        torch.save(stateActor_local, path_localActor_cur)
        torch.save(stateActor_target, path_targetActor_cur)



