import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc = nn.Linear(3072, 512)

        self.mu_layer = nn.Linear(512, 2)
        self.log_std_layer = nn.Linear(512, 2)

    def forward(self, state, fusion=False):
        # x shape: (Batch, 4, 128, 64)
        
        x = state / 255.0

        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layer
        x = F.relu(self.fc(x))
        

        if fusion:
            # Get Distribution Parameters
            mu = self.mu_layer(x)
            log_std = self.log_std_layer(x)
            # Clamp log_std to prevent numerical instability (Important for SAC)
            log_std = torch.clamp(log_std, min=-20, max=2)
        
            return mu, log_std

        else:

            return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic takes State + Action as input
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q_value(x)