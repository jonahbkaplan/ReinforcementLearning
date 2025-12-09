import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module): 
    """
    Actor network which ouputs means and standard deviations for action probability distributions
    """
    def __init__(self, state_size):
        """
        Params:
            state_size (int) -> number of frames we are using per iteration (default 4)
        """
        super(Actor, self).__init__()

        # Downscale image through convolution
        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        #Linear condense
        self.fc = nn.Linear(3072, 512)

        self.mu_layer = nn.Linear(512, 2) # Means for acceleration and steering
        self.log_std_layer = nn.Linear(512, 2) # Log variance for acceleration and steering

    def forward(self, state):
        """
        Params:
            state  (array of size [4, 128, 64]) -> the set of grayscale images to be fed into the CNN
        """
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
        
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mu, log_std

class Critic(nn.Module):
    """
    Actor network which ouputs value function from given state and action sampled from actor
    """
    def __init__(self, action_dim):
        """
        Params:
            action_dim (array of size [2]) -> acceleration and sterring values
        """
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Note: We output 512 features from the image
        self.fc = nn.Linear(3072, 512)

        # Input: 512 (Image features) + 2 (Action vector) = 514
        self.l1 = nn.Linear(512 + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1) # Output: One Q-value

    def forward(self, state, action):
        """
        Params:
            state      (array of size [512]) -> the set of grayscale images to be fed into the CNN
            action     (array of size [2]) -> acceleration and sterring values
        """
        x = state / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # 2. Fuse with Action
        # Concatenate along dimension 1
        x = torch.cat([x, action], dim=1) 
        
        # 3. Calculate Q-Value
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q_value = self.q_out(x)
        
        return q_value