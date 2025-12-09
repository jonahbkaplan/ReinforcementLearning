from algorithms.Agent import Agent

import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

config = {
    "action": {
        "type": "DiscreteMetaAction" 
    }
}

env = gym.make("intersection-v0", render_mode='rgb_array')
#print(env.action_space)

OBS_SHAPE = (15,7)
OBS_DIM = OBS_SHAPE[0] * OBS_SHAPE[1]
N_ACTIONS = 3


class PPO(Agent):
  class PPOActorCritic(nn.Module):
    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(OBS_DIM, 128)
      self.fc2 = nn.Linear(128, 128)
      self.policy_head = nn.Linear(128, N_ACTIONS)
      self.value_head = nn.Linear(128, 1)

    def forward(self, obs):
      if obs.ndim == 2:
        obs = obs.flatten().unsqueeze(0)
      else: 
        obs = obs.view(obs.size(0), -1)
      
      x = F.relu(self.fc1(obs))
      x = F.relu(self.fc2(x))
      logits = self.policy_head(x)
      value = self.value_head(x)
      return logits, value
        

  def __init__(self, env):
    super().__init__()
    self.env = env

    self.gamma = 0.99
    self.Lambda = 0.95
    self.clip = 0.2
    self.LR = 3e-4
    self.batch_size = 64
    self.Epochs = 4
    self.steps_per_update = 256 #512
    self.training_length = 100

  
    self.model = PPO.PPOActorCritic()
    self.optimizer = optim.Adam(self.model.parameters(), lr = self.LR)


  def compute_gae(self, rewards, values, dones, last_value = 0): 
    advantages = []
    gae = 0
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
      gae = delta + self.gamma * self.Lambda * (1 - dones[t]) * gae 
      advantages.insert(0, gae)
    

    if np.isnan(advantages).any():
      print("non number in advantages!")

    return advantages

  def ppo_update(self, obs, actions, prev_log_probabilities, returns, advantages):
    for _ in range(self.Epochs):
      logits, values = self.model(obs)
      dist = torch.distributions.Categorical(logits=logits)
      new_log_probs = dist.log_prob(actions)
      ratio = torch.exp(new_log_probs - prev_log_probabilities)

      obj_1 = ratio * advantages 

      obj_2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
      policy_loss = -torch.min(obj_1, obj_2).mean()

      value_loss = F.mse_loss(values.squeeze(), returns)

      loss = policy_loss + 0.5 * value_loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if torch.isnan(logits).any():
        print("NaN in logits!")
        exit()

  def collect_trajs(self):
    list_obs, actions, rewards, dones, log_probabilities, values = [], [], [], [], [], []
    observations, info = self.env.reset()

    for _ in range(self.steps_per_update):
      tensor_obs = torch.tensor(observations, dtype = torch.float32).unsqueeze(0)
      logits, value = self.model(tensor_obs)
      dist = torch.distributions.Categorical(logits=logits)
      action = dist.sample()
      log_probability = dist.log_prob(action)

      next_observations, reward, done, truncated, info = self.env.step(action.item())

      list_obs.append(observations)
      actions.append(action)
      rewards.append(reward)
      dones.append(done or truncated)
      log_probabilities.append(log_probability)
      values.append(value.squeeze().item())

      observations = next_observations
      if done or truncated:
        observations, info = self.env.reset()


      if np.isnan(observations).any():
        print("NaN in observation!")

    return list_obs, actions, rewards, dones, log_probabilities, values

  def predict(self, pred_obs): 
    with torch.no_grad():
      tensor_observations = torch.tensor(pred_obs, dtype = torch.float32).unsqueeze(0)
      logits, value = self.model(tensor_observations)
      dist = torch.distributions.Categorical(logits = logits)
      action = dist.sample()
      return action.item()


  def learn(self):
    for i in range(self.training_length):
      list_obs, actions, rewards, dones, log_probabilities, values = self.collect_trajs()
      advantages = self.compute_gae(rewards, values, dones)
      returns = [a+v for a, v in zip(advantages, values)]


      observation_tensor = torch.tensor(np.array(list_obs), dtype=torch.float32)
      actions_tensor = torch.stack(actions).squeeze(-1).long()
      returns_tensor = torch.tensor(returns, dtype=torch.float32)
      advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32)
      prev_log_prob_tensor = torch.stack(log_probabilities).squeeze(-1).detach()

      #observation_tensor = observation_tensor.view(len(list_obs), -1)


      advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

      self.ppo_update(observation_tensor, actions_tensor, prev_log_prob_tensor, returns_tensor, advantages_tensor)

      print(f"iteration {i}: mean reward = {np.mean(rewards):.3f}")
