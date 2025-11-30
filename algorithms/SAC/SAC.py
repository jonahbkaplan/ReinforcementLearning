from algorithms.Agent import Agent
from algorithms.SAC.Networks import Actor
from algorithms.SAC.Networks import Critic
import random
import torch


class SAC(Agent):
    # Soft Actor-Critic
    # Marcel  
    def __init__(self, env, episodes = 100, discount = 0.9, alpha = 1.0, state_size = 4):
        """
        SAC Implemention.

        Params:
            env        (Object) -> the environment to act in
            episodes   (int) -> of episodes to run the training algorithm
            discount   (float) -> Discount factor (Gamma)
            alpha      (float) -> Controls log probabilites
            state_size (int) -> How many images passed into actor model
        """
        super().__init__(env)

        self.env = env
        self.replayBuffer = []
        self.policyParams = 0
        self.state_size = state_size

        #Models
        self.actor = Actor(self.state_size) # We are going for 4 images per iteration

        self.critic_1 = Critic()
        self.critic_2 = Critic()
        
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()        

        # Hyperparameters
        self.discount = discount
        self.alpha = alpha


    def predict(self, obs):
        pass

    def learn(self) :
        batch = random.choice(self.replayBuffer)         # Choose a random experience from the replay buffer
        (state, action, reward, new_state, done) = batch # Unpack batch

        action_new = self.choose_action(new_state)
        Q1_action_value_new = self.target_critic_1(new_state, action_new)
        Q2_action_value_new = self.target_critic_2(new_state, action_new)
        min_action_value_new = min(Q1_action_value_new, Q2_action_value_new)
        
        TD_entropy = reward + self.discount * (1 - done) * (min_action_value_new - self.alpha * 1)

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

    
    def collect_data(self):

        obs, _ = self.env.reset()
        self.isDone = False 
        while not self.isDone: # Check is terminal

            action = self.choose_action(obs)

            new_obs, reward, done, _, _ = self.env.step(action)
            self.replayBuffer.append((obs, action, reward, new_obs, done))
            self.isDone = done
            obs = new_obs
        
        self.env.reset()

    
    def choose_action(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # Convert to tensor for efficiency
        mu, log_std = self.actor(state_tensor) # Actor model outputs 
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std) # 2D normal distrubution
        action_tensor = dist.sample()
        action_tensor = torch.tanh(action_tensor) # Rescale between -1 and 1 to match environment logic
        action_numpy = action_tensor.cpu().numpy()[0] # Back to numpy

        return action_numpy