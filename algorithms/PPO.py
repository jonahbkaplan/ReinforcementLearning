import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from algorithms.Agent import Agent


config = {
    "action": {
        "type": "DiscreteMetaAction" 
    },
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 40, 
    "initial_spacing": 2,
    "collision_reward": -1,  
    "reward_speed_range": [40, 50],
    
}

config2 = {
  "observation": {"type": "Kinematics"},
  "action": {"type": "DiscreteMetaAction"},
  "lanes_count": 4,
  "vehicles_count": 50,
  "duration": 40,
  "initial_spacing": 1,

  "collision_reward": -10,
  "reward_speed_range": [20, 30],

  "simulation_frequency": 15,
  "policy_frequency": 5,

  "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}


config1 = {
    "action": {
        "type": "DiscreteMetaAction" 
    },
}

train_env = gym.make("highway-fast-v0", render_mode='rgb_array', config=config2)
eval_env = gym.make("highway-fast-v0", render_mode='rgb_array', config=config2)
watch_env = gym.make("highway-fast-v0", render_mode='human', config=config2)


class PPO(Agent):
  # Neural network for the actor, to produce the probability distribution of actions for a state
  class Actor(nn.Module):
    def __init__(self, observations_dim, n_actions):
      super().__init__()
      self.raw_action_scores = nn.Sequential(
        nn.Linear(observations_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        # Ouptut is the size of the number of dimensions, each value is then used to work out how likely an action is
        nn.Linear(128, n_actions),
      )
    def forward(self, observations):
      if observations.ndim == 1:
        observations = observations.unsqueeze(0)
      elif observations.ndim > 2:
        observations = observations.view(observations.size(0), -1)
      return self.raw_action_scores(observations)
  class Critic(nn.Module):
    # Critic neural network, gives the total expected return of the given state.
    def __init__(self, observations_dim):
      super().__init__()
      self.critic_value = nn.Sequential(
        nn.Linear(observations_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        # Output is only one value.
        nn.Linear(128, 1),
      )
    def forward(self, observations):
      if observations.ndim == 1:
        observations = observations.unsqueeze(0)
      elif observations.ndim > 2:
        observations = observations.view(observations.size(0), -1)
      return self.critic_value(observations).squeeze(-1)

  def __init__(self, env, eval_env=None, watch_env= None):
    super().__init__(env)
    self.env = env
    self.eval_env = eval_env if eval_env is not None else env
    self.watch_env = watch_env if watch_env is not None else env

    observation_space = env.observation_space
    self.observations_dim = int(np.prod(observation_space.shape))
    action_space = env.action_space
    self.n_actions = action_space.n

    print(observation_space)
    '''
    if isinstance(observation_space, gym.spaces.Box):
      self.observations_dim = int(np.prod(observation_space.shape))
    else: 
      raise NotImplementedError(
        f"Unsupported observation input, type: {type(observation_space)}"
      )

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
      self.n_actions = action_space.n
    else: 
      raise NotImplementedError(
        f"Unsupported action input, type: {type(action_space)}"
      )'''


    self.gamma = 0.99
    self.Lambda = 0.95
    self.clip = 0.2
    self.LR = 0.00005
    self.batch_size = 64
    self.Epochs = 2
    self.steps_per_update = 2048 #512
    self.training_length = 1000

    #Defining the actor and critic neural networks
    self.actor = PPO.Actor(self.observations_dim, self.n_actions)
    self.critic = PPO.Critic(self.observations_dim)

    #Defining what optimisers are being used, along with learning rate.
    self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.LR)
    self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.LR)

    self.observations_mean = np.zeros(self.observations_dim, dtype=np.float32)
    self.observations_var = np.ones(self.observations_dim, dtype=np.float32)
    self.observations_count = 0.0001

    

  def update_observations_stats(self, observations):
    self.observations_count += 1.0
    delta = observations - self.observations_mean
    self.observations_mean += delta / self.observations_count
    delta2 = observations - self.observations_mean
    self.observations_var += delta2 ** 2

  def flatten_observations(self, observations, update=True):
    observations = np.asarray(observations, dtype=np.float32).reshape(-1)
    if update:
        self.update_observations_stats(observations)
    std = np.sqrt(self.observations_var / self.observations_count)
    observations = (observations - self.observations_mean) / (std + 0.0000001)
    # Return the standardized observations, clipping them to remove any extreme outliers.
    return np.clip(observations, -10.0, 10.0)

  def compute_advantages(self, rewards, critic_values, dones, last_value = 0): 
    advantages = []
    next_critic_values = critic_values[1:] + [last_value]
    for t in range(len(rewards)):
      adv = rewards[t] + self.gamma * (1.0 - float(dones[t])) * next_critic_values[t] - critic_values[t]
      advantages.append(adv)
    return advantages

  def updateNNs(self, observations, actions, prev_log_probabilities, returns, advantages):
    N = observations.shape[0]
    #print(N)
    for _ in range(self.Epochs):
      # Create random order of indices for inputs
      indices = torch.randperm(N)
      for start in range(0, N, self.batch_size):
        end = start + self.batch_size
        # Split the random order of indices up according to batch size
        batch_idx = indices[start:end]

        # Extracting the values from the inputs based on their index
        batch_observations = observations[batch_idx]
        batch_actions = actions[batch_idx]
        batch_old_log_probabilities = prev_log_probabilities[batch_idx]
        batch_returns = returns[batch_idx]
        batch_advantages = advantages[batch_idx]

        # Prevent gradient descent occuring through advantages and probabilities
        batch_old_log_probabilities = batch_old_log_probabilities.detach()
        batch_advantages = batch_advantages.detach()

        # Collect action scores from the actor NN
        raw_action_scores = self.actor(batch_observations)
        # Use a distribution model to convert them to probabilities
        action_probabilities = torch.distributions.Categorical(logits=raw_action_scores)
        # Log these probabilities
        new_log_probs = action_probabilities.log_prob(batch_actions)

        # Ratio of new action probabilities to old action probabilities
        ratio = torch.exp(new_log_probs - batch_old_log_probabilities)


        # Critic NN gives a value for how good the current state is
        critic_values = self.critic(batch_observations)

        # Multiply this ratio from the log probabilities by the advantages 
        ratio_x_adv = ratio * batch_advantages 
        # Do the same, but this time clip them between an interval, to prevent one update changing the policy too much
        clipped_ratio_x_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages
        # Use the more modest of the two changes to find the loss for the actor NN
        policy_loss = -(torch.min(ratio_x_adv, clipped_ratio_x_adv).mean())
        # Use mse to workout the critic loss
        critic_value_loss = F.mse_loss(critic_values.squeeze(), batch_returns)
        # Calculate the entropy loss for the action probabilities, to ensure exploration is still occuring.
        entropy_loss = action_probabilities.entropy().mean()
        
        # clear the previously stored gradients
        self.actor_opt.zero_grad()
         # Carry out gradient descent on the actor NN
        (policy_loss - 0.001 * entropy_loss).backward()
        # Limit gradients to prevent gradients exploding
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        # clear the previously stored gradients
        self.critic_opt.zero_grad()
        # Carry out gradient descent on the critic NN
        (0.5 * critic_value_loss).backward()
        # Limit gradients to prevent gradients exploding
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

  # the function that explores the environment, and collects the results.
  def collect_trajs(self):
    # Create lists for the variables in this update to be stored in.
    list_observations, actions, rewards, dones, log_probabilities, critic_values = [], [], [], [], [], []
    observations, info = self.env.reset()

    # One loop is a step. 
    for _ in range(self.steps_per_update):
      # Convert observations to tensor:
      flat_observations = self.flatten_observations(observations)
      tensor_observations = torch.tensor(flat_observations, dtype = torch.float32).unsqueeze(0)

      # Use actor NN, then find the log probabilities of each action
      raw_action_scores = self.actor(tensor_observations)
      action_probabilities = torch.distributions.Categorical(logits=raw_action_scores)
      action = action_probabilities.sample()
      log_probability = action_probabilities.log_prob(action)

      # Find the critic value of the current state
      critic_value = self.critic(tensor_observations)

      # The agent takes the next step in the environment and returns data
      next_observations, reward, done, truncated, info = self.env.step(action.item())


      # Add all data to the correct list
      list_observations.append(flat_observations)
      actions.append(action)
      rewards.append(reward)
      dones.append(done or truncated)
      log_probabilities.append(log_probability)
      critic_values.append(critic_value.squeeze().item())

      # Update the current observations
      observations = next_observations
      # Check if the environment has finished (crash or ended)
      if done or truncated:
        observations, info = self.env.reset()
      '''if np.isnan(observations).any():
        print("Non number in observations!")'''

    return list_observations, actions, rewards, dones, log_probabilities, critic_values, observations, (done or truncated)


  # This function simply predicts, it takes in the observations and generates an action based on the actor NN.
  def predict(self, pred_observations): 
   #with torch.no_grad():
      # Flattens actions and converts them to tensors
      flat_observations = self.flatten_observations(pred_observations, update = False)
      tensor_observations = torch.tensor(flat_observations, dtype = torch.float32).unsqueeze(0)
      # Use observations to generate an action from the probabilities.
      raw_action_scores = self.actor(tensor_observations)
      action_probabilities = torch.distributions.Categorical(logits = raw_action_scores)
      action = action_probabilities.sample()
      return action.item()

  # This is another predict function, only this one is greedy, pciking the action with the highest probability.
  def predict_greedily(self, pred_observations):
      #with torch.no_grad():
        # Flattens actions and converts them to tensors
        flat_observations = self.flatten_observations(pred_observations, update= False)
        tensor_observations = torch.tensor(flat_observations, dtype = torch.float32).unsqueeze(0)
         # Use observations to generate the most likely action from the probabilities.
        raw_action_scores = self.actor(tensor_observations)
        critic_value = self.critic(tensor_observations)
        action = torch.argmax(raw_action_scores, dim =1)
        return action.item()


  def learn(self):
    rewards_history = []
    eval_history = []
    self.eval_random_policy()

    # This is the training loop, where the agent learn for the given training length.
    for i in range(self.training_length):
      # Collect all necessary data from the trajectories function
      list_observations, actions, rewards, dones, log_probabilities, critic_values, last_observation, last_done = self.collect_trajs()
      # Checks if the step has ended, in which case there are no more rewards to be gained
      if last_done: 
        last_value = 0.0
      # If step has not ended, the value of the next state is computed using the critic NN.
      else:
        last_flat = self.flatten_observations(last_observation, update=False)
        #with torch.no_grad():
        last_v = self.critic(torch.tensor(last_flat, dtype=torch.float32).unsqueeze(0))
        last_value = last_v.item()

      # Compute the advantages of actions that were taken
      advantages = self.compute_advantages(rewards, critic_values, dones, last_value)
      # Calculate total amount of reward to expect from this state and beyond
      returns = [a+v for a, v in zip(advantages, critic_values)]

      # Convert all data to tensor for inputing into function to update NNs.
      observation_tensor = torch.tensor(np.array(list_observations), dtype=torch.float32)
      actions_tensor = torch.stack(actions).squeeze(-1).long()
      returns_tensor = torch.tensor(returns, dtype=torch.float32)
      advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32)
      prev_log_prob_tensor = torch.stack(log_probabilities).squeeze(-1).detach()

      # Standardise advantages to stabilise updates
      advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 0.00000008)

      # Use all these tensors to update the two neural networks
      self.updateNNs(observation_tensor, actions_tensor, prev_log_prob_tensor, returns_tensor, advantages_tensor)
      print(f"iteration {i}: mean reward = {np.mean(rewards):.3f}")

      # Run an evaluation function to assess performance every 10 iterations
      if (i + 1) % 10 == 0:
        eval_returns = self.evaluate()
        eval_history.append(eval_returns)

      
      # Add rewards to reward history to plot at the end.
      rewards_history.append(np.mean(rewards))
    plt.plot(rewards_history)
    plt.show()

  # Assesses the performance of a random policy for comparison to PPO
  def eval_random_policy(self, n_episodes=30):
    returns = []
    # Loops multiple times to ensure accurate value
    for _ in range(n_episodes):
        observations, info = self.env.reset()
        # these track if the vehicle crashes or reaches the time limit
        done, truncated = False, False
        # Before the episode has started, the return is 0
        ep_return = 0.0
        # Loop until the environment ends:
        while not (done or truncated):
            # Select a random action:
            a = self.env.action_space.sample()
            # get the results of that 
            #result = self.env.step(a)
            # get the results of that action
            observations, rewards, terminated, truncated, info = self.env.step(a)
            # is the episode finished:
            done = terminated or truncated
            # Add to rewards total
            ep_return += rewards
        returns.append(ep_return)
    print("Random policy avg return:", np.mean(returns))

  # Real evaluation loop
  def evaluate(self, no_episodes=50):
    # this function calculates the mean return, episode length and crash percentage
    returns = []
    episode_lengths = []
    crashes = 0

    for seed in range(1000, 1000+no_episodes):
      observations, info = self.eval_env.reset(seed=seed)
      done, truncated, crashed = False, False, False
      ep_return = 0.0 
      steps = 0
      while not (done or truncated):
        a = self.predict_greedily(observations)
        result = self.eval_env.step(a)
        if len(result) == 5:
          observations, r, terminated, truncated, info = result
          done = terminated or truncated
        else: 
          observations, r, done, info = result
          truncated = False
        steps += 1
        ep_return += r
        if info.get("crashed", False):
                crashed = True

        if terminated or truncated:
          break
      returns.append(ep_return)
      episode_lengths.append(steps)
      if crashed:
        crashes += 1
    avg_return = float(np.mean(returns))
    std = np.std(returns)
    avg_length = np.mean(episode_lengths)
    crash_rate = crashes / no_episodes

    print(
        f"Eval over {no_episodes} episodes | "
        f"Return: {avg_return:.3f} | "
        f"Std:  {std:.3f} | "
        f"Length: {avg_length:.1f} steps | "
        f"Crash rate: {crash_rate*100:.1f}%"
    )

    return {
        "return": avg_return,
        "length": avg_length,
        "crash_rate": crash_rate
    }
    

  def watch_agent(self, n_episodes=5):
    for ep in range(n_episodes):
        observations, info = self.watch_env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        while not (done or truncated):
            action = self.predict_greedily(observations)
            next_observations, reward, terminated, truncated, info = self.watch_env.step(action)
            done = terminated or truncated

            ep_return += reward
            self.watch_env.render()
            observations = next_observations
        print(f"Episode {ep+1} return: {ep_return:.3f}")


agent = PPO(train_env, eval_env=eval_env, watch_env=watch_env)
agent.learn()
agent.watch_agent(n_episodes = 100)



