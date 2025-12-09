import os, sys
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(parent)

from Networks import Critic
from Networks import Actor
from algorithms.Agent import Agent
import random
import torch
from torch.nn import functional as F
import torch.optim as optim
import numpy as np


class SAC(Agent):
    # Soft Actor-Critic
    # Marcel  
    def __init__(self, env, discount = 0.9, alpha = 1.0, state_size = 4, lr=0.001):
        """
        SAC Implemention.

        Params:
            env        (Object) -> the environment to act in
            discount   (float) -> Discount factor (Gamma)
            alpha      (float) -> Controls log probabilites
            state_size (int) -> How many images passed into actor model
            lr         (int) -> learning rate
        """
        super().__init__(env)

        self.env = env
        self.replayBuffer = []
        self.policyParams = 0
        self.state_size = state_size

        #Define actor model
        self.actor = Actor(self.state_size) # We are going for 4 images per iteration
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Define critics
        self.critic_1 = Critic(action_dim = 2)
        self.critic_2 = Critic(2)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        #Define target critics
        self.target_critic_1 = Critic(2)
        self.target_critic_2 = Critic(2)     
        
        # Initialise target critics with same parameters as critics
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
          
        # Hyperparameters
        self.discount = discount
        self.alpha = alpha

    def predict(self, obs):
        pass

    def learn(self, batch_size=256) :
        """
        Main loop to update actor, critic and target networks. All operation occur at a batch size level
        """

        # Check if we have enough data
        if len(self.replayBuffer) < batch_size:
            return

        # Choose a batch random experience from the replay buffer
        batch = random.sample(self.replayBuffer, batch_size)         
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)


        # Updating the critic network
        state = torch.FloatTensor(np.array(state_batch))
        new_state = torch.FloatTensor(np.array(next_state_batch))
        action = torch.FloatTensor(np.array(action_batch))
        reward = torch.FloatTensor(np.array(reward_batch)).unsqueeze(1)
        done = torch.FloatTensor(np.array(done_batch)).unsqueeze(1)

        with torch.no_grad():
            action_new, log_prob = self.choose_action(new_state)
            Q1_action_value_new = self.target_critic_1(new_state, action_new)
            Q2_action_value_new = self.target_critic_2(new_state, action_new)
            min_action_value_new = torch.min(Q1_action_value_new, Q2_action_value_new)
  
            TD_entropy = reward + self.discount * (1 - done) * (min_action_value_new - self.alpha * log_prob)

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        loss_q1 = F.mse_loss(current_q1, TD_entropy)
        loss_q2 = F.mse_loss(current_q2, TD_entropy)
        critic_loss = loss_q1 + loss_q2

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Updating the actor network

        new_action, log_prob = self.choose_action(state)

        # Get Q-value of the NEW action
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # Calculate Loss (Flip sign to Maximize)
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Updating the target network parameters
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


    
    def collect_data(self):
        """
        Collect experiences to store in the replay buffer
        """

        state, _ = self.env.reset()      # Reset environment
        done = truncated = False         # Initialise terminal status
        while not (done or truncated):   # Check if terminal

            action = self.choose_action_numpy(state)                           # Choose action
            new_state, reward, done, truncated, _ = self.env.step(action)      # Step in environment
            self.replayBuffer.append((state, action, reward, new_state, done)) # Add step to buffer                                     
            state = new_state                                                  # Update current state
            self.env.render()                                              
        
        self.env.reset()

    
    def choose_action(self, state):
        """
        Randomly sample an action as an equation from a state along with its log_prob
        """

        mu, log_std = self.actor(state)                                      # Actor model outputs 
        std = torch.exp(log_std)                                             # Get normal variance
        dist = torch.distributions.Normal(mu, std)                           # 2D normal distrubution
        action_tensor = dist.rsample()                                       # Random sample from distribution
        action_tensor_scaled = torch.tanh(action_tensor)                     # Rescale between -1 and 1 to match environment logic

        # Jacobian correction
        log_prob = dist.log_prob(action_tensor)
        log_prob -= torch.log(1 - action_tensor_scaled.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_tensor_scaled, log_prob
    
    def soft_update(self, local_model, target_model, rho=0.005):
        """
        Updates the target network parameters with rho defining the influence of each model on the update
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                rho * local_param.data + (1.0 - rho) * target_param.data
            )

    def choose_action_numpy(self, state):
        """
        Input: Numpy Array of shape (4, 128, 64)
        Output: Numpy Array of shape (2,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_tensor, _ = self.choose_action(state_tensor)
            
        return action_tensor.cpu().numpy()[0]