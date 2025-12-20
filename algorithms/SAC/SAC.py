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
from collections import deque


class SAC(Agent):
    # Soft Actor-Critic
    # Marcel  
    def __init__(self, env, discount = 0.95, alpha = 0.2, state_size = 105, lr=0.0003):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.replayBuffer = deque(maxlen=50000) # Cap size at 50000
        self.state_size = state_size

        #Define actor model and optimiser
        self.actor = Actor(self.state_size).to(self.device) # 4 images per iteration
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Define critics and optimisers
        self.critic_1 = Critic(action_dim = 2).to(self.device)
        self.critic_2 = Critic(action_dim = 2).to(self.device)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        #Define target critics 
        self.target_critic_1 = Critic(action_dim = 2).to(self.device)
        self.target_critic_2 = Critic(action_dim = 2).to(self.device)    
        
        # Initialise target critics with same parameters as critics
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
          
        # Hyperparameters
        self.discount = discount

        # Adaptive alpha
        self.target_entropy = -float(env.action_space.shape[0]) 
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item() # Initial value

    def predict(self, state):
        """
        Randomly sample an action as an equation from a state along with its log_prob
        """

        mu, log_std = self.actor(state)                                      # Mean and log variance of action distributions from actor 
        std = torch.exp(log_std)                                             # Get normal variance
        dist = torch.distributions.Normal(mu, std)                           # Define normal distrubution
        action_tensor = dist.rsample()                                       # Random sample from distribution using reparameterisation trick
        action_tensor_scaled = torch.tanh(action_tensor)                     # Rescale between -1 and 1 to match environment logic

        # Jacobian correction
        log_prob = dist.log_prob(action_tensor)
        log_prob -= torch.log(1 - action_tensor_scaled.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_tensor_scaled, log_prob

    def learn(self, batch_size=256) :
        """
        Main loop to update actor, critic and target networks. All operation occur at a batch size level
        """

        # Check if we have enough data
        if len(self.replayBuffer) < batch_size:
            return None, None, None

        # Choose a batch random experience from the replay buffer
        batch = random.sample(self.replayBuffer, batch_size)         
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert to tensors for operations
        state = torch.FloatTensor(np.array(state_batch)).to(self.device)
        new_state = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        action = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward = torch.FloatTensor(np.array(reward_batch)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.array(done_batch)).unsqueeze(1).to(self.device)

        # 1. Updating the critic network
        with torch.no_grad():
            action_new, log_prob = self.predict(new_state)

            # Get smallest Q-value of the new action from both target critics
            Q1_action_value_new = self.target_critic_1(new_state, action_new)
            Q2_action_value_new = self.target_critic_2(new_state, action_new)
            min_action_value_new = torch.min(Q1_action_value_new, Q2_action_value_new)

            TD_entropy = reward + self.discount * (1 - done) * (min_action_value_new - self.alpha * log_prob)

        # Get Q_values of actions from critics
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Calculate critic loss
        loss_q1 = F.mse_loss(current_q1, TD_entropy)
        loss_q2 = F.mse_loss(current_q2, TD_entropy)
        critic_loss = loss_q1 + loss_q2

        # Gradient descent
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 2. Updating the actor network

        # Freeze critics for speedup and memory
        for p in self.critic_1.parameters(): p.requires_grad = False
        for p in self.critic_2.parameters(): p.requires_grad = False

        # New action
        new_action, log_prob = self.predict(state)

        # Get smallest Q-value of the new action from both critics
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # Calculate Loss
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        # Gradient descent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critics
        for p in self.critic_1.parameters(): p.requires_grad = True
        for p in self.critic_2.parameters(): p.requires_grad = True

        # 3. Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 4. Updating the target network parameters
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return critic_loss.item(), actor_loss.item(), self.alpha


    
    def collect_data(self, current_state):
        """
        Collect experiences to store in the replay buffer
        """

        action = self.sample_action_numpy(current_state)                                 # Choose action
        next_state, reward, done, truncated, _ = self.env.step(action)                   # Step in environment
        real_done = done and not truncated                                               # Set up done condition
        self.replayBuffer.append((current_state, action, reward, next_state, real_done)) # Add step to buffer                                                                                          

        return next_state, reward, done, truncated                                          
        

    
    def soft_update(self, local_model, target_model, rho=0.005):
        """
        Updates the target network parameters with rho defining the influence of each model on the update
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                rho * local_param.data + (1.0 - rho) * target_param.data
            )

    def choose_action_deterministic(self, state):
        """
        Input: Numpy Array of shape (4, 128, 64)
        Output: Numpy Array of shape (2,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, _ = self.actor(state_tensor)  # Action is defined as the mean as that is the most likely outcome
            action_tensor = torch.tanh(mu)    # -1 to 1 mapping to match environment
            
        return action_tensor.cpu().numpy()[0] # Convert action to numpy for environment stepping
    
    def sample_action_numpy(self, state):
        """
        Input: Numpy Array (4, 128, 64)
        Output: Numpy Array (2,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tensor, _ = self.predict(state_tensor)
            
        return action_tensor.cpu().numpy()[0] # Convert action to numpy for environment stepping
    

    def train_loop(self, total_timesteps=100000, batch_size=256):
        """
        MAIN LOOP: Interleaves Data Collection and Training
        """

        # Initialisation
        rewards_history = []
        cl_list = []
        ac_list = []
        alpha_decay = []
        state, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(total_timesteps):

            if (step % 1000) == 0:
                print(f"Step-milestone: {step}")
            
            # Add step to buffer
            state, reward, done, truncated = self.collect_data(state)

            episode_reward += reward

            # Perform model updates if enough data in buffer
            if len(self.replayBuffer) > batch_size:
                cl, ac, alpha = self.learn(batch_size)

                if cl is not None:
                    cl_list.append(cl)
                    ac_list.append(ac)
                    alpha_decay.append(alpha)

            # Reset environment is terminal state is reached
            if done or truncated:
                print(f"Step {step}: Episode Reward: {episode_reward}")
                rewards_history.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0

        # Save model parameters
        self.save_checkpoint("models/sac_final_kinematics.pth")
        self.save_actor_only("models/sac-actor_kinematics.pth")

        return rewards_history, cl_list, ac_list, alpha_decay
    

    # FUNCTIONS TO SAVE MODEL PARAMETERS AND LOAD THEM BACK

    def save_checkpoint(self, filename="sac_checkpoint.pth"):
            print(f"Saving checkpoint to {filename}...")
            checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_1_state_dict': self.critic_1.state_dict(),
                'critic_2_state_dict': self.critic_2.state_dict(),
                'target_critic_1_state_dict': self.target_critic_1.state_dict(),
                'target_critic_2_state_dict': self.target_critic_2.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
                'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'alpha_optimizer': self.alpha_optimizer.state_dict(),
            }
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename="sac_checkpoint.pth"):
        print(f"Loading checkpoint from {filename}...")
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.alpha = self.log_alpha.exp().item()

    def save_actor_only(self, filename="sac_actor.pth"):
            """Saves just the actor for lightweight inference/video recording"""
            torch.save(self.actor.state_dict(), filename)
        
    def load_actor_only(self, filename="sac_actor.pth"):
        self.actor.load_state_dict(torch.load(filename, map_location=self.device))