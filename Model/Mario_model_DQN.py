import torch
import torch.nn as nn

#4 * 84 * 84
class Q_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q_Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=4, stride=2), # 8 * 41 * 41
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2), # 16 * 20 * 20
            nn.ReLU(),
            nn.Conv2d(16, 48, kernel_size=2, stride=2),  # 48 * 10 * 10
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=2, stride=2), # 96 * 5 * 5
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2), # 192 * 2 * 2
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=2, stride=2),  # 192 * 1 * 1
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return x