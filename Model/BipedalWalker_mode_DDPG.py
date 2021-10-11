import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.Linear1 = nn.Linear(observation_dim, 48)
        self.Linear2 = nn.Linear(48, 24)
        self.Linear3 = nn.Linear(24 + self.action_dim, 1)

    def forward(self, observation, action):
        x = self.Linear1(observation)
        x = F.relu(x)
        x = self.Linear2(x)
        x = F.relu(x)
        x = torch.cat((x, action), dim=1)
        x = self.Linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.observation_dim = observation_dim
        self.Linear1 = nn.Linear(self.observation_dim, 48)
        self.Linear2 = nn.Linear(48, 48)
        self.Linear3 = nn.Linear(48, action_dim)

    def forward(self, observation):
        x = self.Linear1(observation)
        x = F.relu(x)
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return F.tanh(x)
