import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module): 
    """
    Actor network which ouputs means and standard deviations for action probability distributions
    """
    def __init__(self, state_size = 105):
        """
        Params:
            state_size (int) -> Dimension of flattened kinematic obs (e.g. 5 vehicles * 5 features = 25)
        """
        super(Actor, self).__init__()


        '''
        # Downscale image through convolution
        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        #Linear condense
        self.fc = nn.Linear(3072, 512)

        self.mu_layer = nn.Linear(512, 2)      # Means for acceleration and steering
        self.log_std_layer = nn.Linear(512, 2) # Log variance for acceleration and steering
        '''

        # MLP Layers 
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Output Heads
        self.mu_layer = nn.Linear(256, 2)       # Means for acceleration and steering
        self.log_std_layer = nn.Linear(256, 2)  # Log Variance variance for acceleration and steering

        # Near zero weight init to avoid running into wall early
        torch.nn.init.xavier_uniform_(self.mu_layer.weight)
        self.mu_layer.weight.data.mul_(0.001) 
        self.mu_layer.bias.data.fill_(0.0)

    def forward(self, state):
        """
        Params:
            state  (array of size [4, 128, 64]) -> the set of grayscale images to be fed into the CNN
        """
        
        '''
        # 1. Normalise 
        x = state / 255.0

        # 2. Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 3. Flatten
        x = x.view(x.size(0), -1)
        
        # 4. Dense Layer
        x = F.relu(self.fc(x))
        '''

        # Flatten the (V, F) matrix into a single vector
        x = state.view(state.size(0), -1)
        
        #  Dense Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        
        # Obtain mean and log variance
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mu, log_std

class Critic(nn.Module):
    """
    Actor network which ouputs value function from given state and action sampled from actor
    """
 
    def __init__(self, state_size=105, action_dim=2):
        """
        Params:
            state_size (int) -> Dimension of flattened kinematic obs
            action_dim (int) -> Dimension of action space
        """
        super(Critic, self).__init__()

        '''
        # Same as actor network
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Same linear condense as the actor
        self.fc = nn.Linear(3072, 512)
        '''

        # Input: 512 (Image features) + 2 (Action vector)
        self.l1 = nn.Linear(state_size + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1) # Output: One Q-value

    def forward(self, state, action):
        """
        Params:
            state  (tensor) -> [Batch, 5, 5]
            action (tensor) -> [Batch, 2]
        """
        '''
        x = state / 255.0

        # Equivalent to passing through actor but with new weights
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        '''

        state = state.view(state.size(0), -1)

        # 2. Fuse with Action
        x = torch.cat([state, action], dim=1) 
        
        # 3. Calculate Q-Value
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q_value = self.q_out(x)
        
        return q_value