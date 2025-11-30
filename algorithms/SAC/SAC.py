from algorithms.Agent import Agent
from algorithms.SAC.Networks import Actor
from algorithms.SAC.Networks import Critic
import random
import torch


class SAC(Agent):
    # Soft Actor-Critic
    # Marcel  
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.replayBuffer = []
        self.policyParams = 0

        self.actor = Actor(4)

        self.critic_1 = Critic()
        self.critic_2 = Critic()
        
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()        

        self.discount = 0.9
        self.alpha = 1


    def predict(self, obs):
        pass

    def learn(self) :
        batch = random.choice(self.replayBuffer)
        (state, action, reward, new_state, done) = batch

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
        while not self.isDone:

            action = self.choose_action(obs)

            new_obs, reward, done, _, _ = self.env.step(action)
            self.replayBuffer.append((obs, action, reward, new_obs, done))
            self.isDone = done
            obs = new_obs
        
        self.env.reset()

    
    def choose_action(self, obs):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mu, log_std = self.actor(state_tensor)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action_tensor = dist.sample()
        action_tensor = torch.tanh(action_tensor)
        action_numpy = action_tensor.cpu().numpy()[0]

        return action_numpy