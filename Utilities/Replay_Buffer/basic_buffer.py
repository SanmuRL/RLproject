import torch
import numpy as np
import random
from collections import namedtuple, deque

class basic_buffer(object):
    def __init__(self, batch_size, capacity, device=None):
        self.batch_size = batch_size
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("experience", field_names=["state", "action", "reward", "next_state", "done"])
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """
        添加经验数据
        对于每个done，不应是[true]这种list，便于之后神经网络计算
        存入memory的是namedsuple类型的元组, 每个元组中的done都是True而不是[True]
        """
        if type(dones) == list:
            assert type(dones[0]) != list, "done should not be a list"
            experiences = [self.experience(np.array([screen for screen in list(state)]), action, reward,
                                           np.array([screen for screen in list(next_state)]), done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]

            self.memory.extend(experiences)
        else:
            # experiences = self.experience(state, action, reward, next_state, done)
            # self.memory.append(experiences)
            state = torch.tensor(states).float().cuda()
            next_state = torch.tensor(next_states).float().cuda()
            action = torch.tensor(actions).float().cuda()
            reward = torch.tensor([rewards]).float().cuda()
            done = torch.tensor([dones]).cuda()
            self.memory.append((state, next_state, action, reward, done))

    def sample(self):
        """随机取出的是namedtuple的集合，并将其进行分离"""
        # batch = random.sample(self.memory, self.batch_size)
        # states = torch.from_numpy(np.stack([e.state for e in batch if e is not None], axis=0)).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.stack([e.next_state for e in batch if e is not None], axis=0)).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([int(e.done) for e in batch if e is not None])).float().to(self.device)
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done
        #return states, actions, rewards, next_states, dones

    def enough_experience(self):
        return len(self.memory) >= self.capacity

    def __len__(self):
        return len(self.memory)
