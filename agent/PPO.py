"""PPO算法实现"""

import torch
import random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utilities.action_strategies.Epsilon_Greedy import Epsilon_greedy
from Utilities.Replay_Buffer.basic_buffer import basic_buffer
from Model.Mario_model_PPO import Policy
from Model.Mario_model2 import MarioNet
import datetime
from torch.distributions import Categorical

class PPO(object):
    def __init__(self, config):
        self.config = config
        self.gamma = self.config["gamma"]
        self.step = self.config["step"]
        self.clip = self.config["clip"]
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.batch_nums = self.config["batch_nums"]
        self.img_stack = self.config["img_stack"]
        self.action_dim = self.config["action_dim"]
        self.save_path_local = self.config["save_path_local"]
        self.useExit = self.config["useExit"]
        self.grads = {}
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(self.img_stack, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=self.step)
        if self.useExit:
            self.policy.load_state_dict(torch.load(self.save_path_local + '2021-10-06T01-57-37.pth')['net'])
            self.optimizer.load_state_dict(torch.load(self.save_path_local + '2021-10-06T01-57-37.pth')['optimizer'])

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def update_policy(self, Parallel_buffer):
        advantages = Parallel_buffer.returns[:-1] - Parallel_buffer.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        sum_loss = 0
        L1 = 0
        L2 = 0

        for _ in range(self.num_epochs):

            dataLoader = Parallel_buffer.sample(advantages, self.batch_nums)
            for sampler in dataLoader:
                obs, action_prob, action, value, advantage, returns = sampler
                local_value, dis = self.policy(obs)
                # new_distributions = [Categorical(D) for D in dis]
                # new_action_prob = torch.cat([distribution.log_prob(a) for distribution, a
                #                          in zip(new_distributions, action)]).unsqueeze(1)
                new_action_prob = dis.gather(dim=1, index=action.long())
                ratio = new_action_prob / action_prob
                #h1 = local_value.register_hook(self.save_grad('local_value'))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage
                loss1 = torch.min(surr1, surr2).mean()
                loss2 = self.SmoothL1Loss(returns, local_value)
                loss = -loss1 + 0.5 * loss2
                #print('loss: %f, loss1: %f, loss2: %f' % (loss.item(), loss1.item(), loss2.item()))
                sum_loss += loss.item()
                L1 += loss1.item()
                L2 += loss2.item()
                self.optimizer.zero_grad()
                loss.backward()
                #print(self.policy.cnn[0].weight.grad)
                #print(self.grads['local_value'])
                #h1.remove()
                self.optimizer.step()

        return sum_loss, L1, L2

    def choose_action(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        value, dis = self.policy(states)
        distributions = [Categorical(d.cpu()) for d in dis]
        actions = torch.tensor([distribution.sample().item() for distribution in distributions]).unsqueeze(1)
        # actions_log = torch.tensor([distribution.log_prob(action).numpy() for distribution, action
        #                             in zip(distributions, actions)])
        actions_prob = dis.gather(dim=1, index=actions.long().to(self.device))
        return value, actions, actions_prob

    def locally_save_policy(self):
        state_local = {'net': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict()}
        path_local_cur = self.save_path_local + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.pth'
        torch.save(state_local, path_local_cur)


