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
import os

config1 = {
  "observation": {"type": "Kinematics"},
  "action": {"type": "DiscreteMetaAction"},
  "lanes_count": 4,
}

train_env = gym.make("highway-fast-v0", render_mode='rgb_array', config=config1)
eval_env = gym.make("highway-fast-v0", render_mode='human', config=config1)
watch_env = gym.make("highway-v0", render_mode='human', config=config1)

print(train_env.unwrapped.config)

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
    
    # Hyperparameter values for the agent, chosen after much deliberation.
    self.gamma = 0.9
    self.Lambda = 0.95
    self.clip = 0.2
    self.LR = 0.0003
    self.batch_size = 128
    self.Epochs = 4
    self.steps_per_update = 1024 #512
    self.training_length = 1500

    #Defining the actor and critic neural networks
    self.actor = PPO.Actor(self.observations_dim, self.n_actions)
    self.critic = PPO.Critic(self.observations_dim)

    #Defining what optimisers are being used, along with learning rate.
    self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.LR)
    self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.LR)


    self.observations_mean = np.zeros(self.observations_dim, dtype=np.float32)
    self.observations_var = np.ones(self.observations_dim, dtype=np.float32)
    self.observations_count = 0.0000001

  # Welfords algorithm for updating mean and variance:
  def update_observations_stats(self, new_observations):
    # records how many observation there have been
    self.observations_count += 1.0
    # Difference between new observation and current average
    delta = new_observations - self.observations_mean
    # Update this average
    self.observations_mean += delta / self.observations_count
    # Difference between oberservations and new average
    delta2 = new_observations - self.observations_mean
    # update the current variance
    self.observations_var += delta * delta2

  # Flatten the inputs for the neural networks to use:
  def flatten_observations(self, observations, update=True):
    # Convert to a single vector 
    observations = np.asarray(observations, dtype=np.float32).reshape(-1)
    # update the mean and variance if applicable:
    if update:
        self.update_observations_stats(observations)
    # find the standard deviation:
    var = self.observations_var / max(self.observations_count - 1.0, 1.0)
    std = np.sqrt(var)
    # Normalise for mean = 0 and variance = 1
    observations = (observations - self.observations_mean) / (std + 0.0000001)
    # Return the standardized observations, clipping them to remove any extreme outliers.
    return np.clip(observations, -10.0, 10.0)

  # Calculates the Generalised estimated advantage:
  def calculate_gae(self, rewards, values, list_ended, last_value):
    # Create place to store gae for each state
    List_gae = []
    gae = 0.0
    # Adds the state reached after action has been taken 
    values = values + [last_value]
    # Work back from end of time step, treating the initial time step as 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(list_ended[t])
        # Compute TD error: 
        td_error = rewards[t] + self.gamma * values[t+1] * mask - values[t]
        # Weight the td_error according to gamma and lambda.
        gae = td_error + self.gamma * self.Lambda * mask * gae
        List_gae.append(gae)
    List_gae.reverse()
    # The advantages are then summed with the values of each state for the critic.
    returns = [adv + v for adv, v in zip(List_gae, values[:-1])]
    return List_gae, returns

  def updateNNs(self, observations, actions, prev_log_probabilities, returns, advantages):
    N = observations.shape[0]
    # Reuses data for some sampling efficiency
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
        (policy_loss - 0.01 * entropy_loss).backward()
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
    list_observations, actions, rewards, list_ended, log_probabilities, critic_values = [], [], [], [], [], []
    episode_returns = []
    episode_discounted_returns = []

    current_ep_return = 0.0
    current_ep_discounted = 0.0
    discount = 1.0
    observations, info = self.env.reset()
    # One loop is a step. 
    for _ in range(self.steps_per_update):
      # Convert observations to tensor:
      flat_observations = self.flatten_observations(observations)
      tensor_observations = torch.tensor(flat_observations, dtype = torch.float32).unsqueeze(0)
      # Use actor NN, then find the log probabilities of each action
      with torch.no_grad():
        raw_action_scores = self.actor(tensor_observations)
        action_probabilities = torch.distributions.Categorical(logits=raw_action_scores)
        action = action_probabilities.sample()
        log_probability = action_probabilities.log_prob(action)
        # Find the critic value of the current state
        critic_value = self.critic(tensor_observations)
        # The agent takes the next step in the environment and returns data
        next_observations, reward, terminated, truncated, info = self.env.step(action.item())
      
      current_ep_return += reward
      current_ep_discounted += discount * reward
      discount *= self.gamma
      # Add all data to the correct list
      list_observations.append(flat_observations)
      actions.append(action.item())
      rewards.append(reward)
      list_ended.append(terminated or truncated)
      log_probabilities.append(log_probability.item())
      critic_values.append(critic_value.squeeze().item())
      # Update the current observations
      observations = next_observations
      # Check if the environment has finished (crashed or ended) and reset if so
      if terminated or truncated:
        episode_returns.append(current_ep_return)
        episode_discounted_returns.append(current_ep_discounted)

        current_ep_return = 0.0
        current_ep_discounted = 0.0
        discount = 1.0

        observations, info = self.env.reset()
    return list_observations, actions, rewards, list_ended, log_probabilities,critic_values,observations, (terminated or truncated), episode_returns,episode_discounted_returns

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

  def train(self):
    rewards_history, returns_history, length_history, crash_rate_history= [], [], [], []
    total_rewards_history = []
    discounted_rewards_history = []
    self.evaluate_random()
    # This is the training loop, where the agent learns for the given training length.
    for i in range(self.training_length):
      # Collect all necessary data from the trajectories function
      list_observations, actions, rewards, list_ended, log_probabilities, critic_values, last_observation, episode_ended, episode_returns, episode_discounted_returns = self.collect_trajs()
      # Calculate the average for discounted and total returns from the trajectories
      mean_ep_return = np.mean(episode_returns) if episode_returns else 0.0
      mean_disc_ep_return = np.mean(episode_discounted_returns) if episode_discounted_returns else 0.0
       # Checks if the step has ended, in which case there are no more rewards to be gained
      if episode_ended: 
        last_value = 0.0
      # If step has not ended, the value of the next state is computed using the critic NN.
      else:
        last_flat = self.flatten_observations(last_observation, update=False)
        #with torch.no_grad():
        last_v = self.critic(torch.tensor(last_flat, dtype=torch.float32).unsqueeze(0))
        last_value = last_v.item()
      # Compute the advantages of actions that were taken, along with the total amount of reward to expect from this state and beyond
      advantages, returns = self.calculate_gae(rewards, critic_values, list_ended, last_value)
      # Convert all data to tensor for inputing into function to update NNs.
      observation_tensor = torch.tensor(np.array(list_observations), dtype=torch.float32)
      actions_tensor = torch.tensor(actions, dtype=torch.long)
      returns_tensor = torch.tensor(returns, dtype=torch.float32)
      advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32)
      prev_log_prob_tensor = torch.tensor(log_probabilities, dtype=torch.float32)
      # Standardise advantages to stabilise updates
      advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 0.00000008)


      total_rewards_history.append(total_rewards_history)
      discounted_rewards_history.append(mean_disc_ep_return)
      print(
          f"iter {i}: "
          f"mean_episode_return={mean_ep_return:.3f}, "
          f"mean_discounted_episode_return={mean_disc_ep_return:.3f}"
      )
      # Use all these tensors to update the two neural networks
      self.updateNNs(observation_tensor, actions_tensor, prev_log_prob_tensor, returns_tensor, advantages_tensor)
      print(f"iteration {i}: mean reward = {np.mean(rewards):.3f}")
      # Add rewards to reward history to plot at the end.
      rewards_history.append(np.mean(rewards))
      # Run an evaluation function to assess performance every certain number of iterations
      if (i + 1) % 10 == 0:
        avg_return, avg_length, crash_rate = self.evaluate()
        returns_history.append(avg_return)
        length_history.append(avg_length)
        crash_rate_history.append(crash_rate)

      if (i + 1) % 5 == 0:
        self.save("ppo_latest.pt")
        print("model saved!!")
    # Plot and print values once the training loop has ended:
    print(rewards_history)
    print(returns_history)
    print(length_history)
    print(crash_rate_history)
    print(total_rewards_history)
    print(discounted_rewards_history)
    plt.plot(rewards_history)
    plt.show()
    plt.plot(returns_history)
    plt.show()
    plt.plot(length_history)
    plt.show()
    plt.plot(crash_rate_history)
    plt.show()

  # Assesses the performance of a random policy for comparison to PPO
  def evaluate_random(self, n_episodes=30):
    returns = []
    # Loops multiple times for accuracy
    for _ in range(n_episodes):
        # Start new episode
        observations, info = self.env.reset()
        # these track if the vehicle crashes or reaches the time limit
        terminated, truncated = False, False
        episode_return = 0.0
        # Loop until the environment ends:
        while not (terminated or truncated):
            # Action selected randomly:
            random_action = self.env.action_space.sample()
            # get the results of that action
            observations, rewards, terminated, truncated, info = self.env.step(random_action)
            # Add to rewards total
            episode_return += rewards
        returns.append(episode_return)
    print("Random policy avg return:", np.mean(returns))

  # Real evaluation loop
  def evaluate(self, no_episodes=50):
    # this function calculates the mean return, episode length and crash percentage
    # This is done to help monitor performance of agent during training.
    returns = []
    episode_lengths = []
    crashes = 0
    # seeding is used for evaluation loops, this ensures the agent is being tested on the same environments each time.
    for _ in range(no_episodes):
      # Start a new episode in the environment.
      observations, info = self.eval_env.reset()
      # At first the environment hasnt crashed or ended.
      terminated, truncated, crashed = False, False, False
      episode_return = 0.0 
      steps = 0
      # While loop runs until environment ends
      while not (terminated or truncated):
        # Predict greedily, meaning action with the largest prob in the distribution is chosen
        greedy_action = self.predict_greedily(observations)
        # Get the outcomes of that action
        observations, rewards, terminated, truncated, info = self.eval_env.step(greedy_action)
        steps += 1
        # Add rewards from this action to the total
        episode_return += rewards
        # Check if car has crashed
        if info.get("crashed", False):
                crashed = True
      # Add the total rewards and length of the episode to a list
      returns.append(episode_return)
      episode_lengths.append(steps)
      if crashed:
        crashes += 1
    # Find averages and standard deviation:
    avg_return = np.mean(returns)
    std = np.std(returns)
    avg_length = np.mean(episode_lengths)
    crash_rate = (crashes / no_episodes)
    print(f"Eval over {no_episodes} episodes: Return = {avg_return:.3f}, Std = {std:.3f}, Length = {avg_length:.1f} steps, Crash rate = {crash_rate*100:.1f}%")
    return avg_return, avg_length, crash_rate
       
  # Function that tests and displays the performance of the agent after training
  def test_and_watch(self, n_episodes=50):
    # Once again several episodes to account for outliers
    for episode in range(n_episodes):
        # New environment
        observations, info = self.watch_env.reset()
        terminated, truncated = False, False
        episode_return = 0.0
        # While environment is running:
        while not (terminated or truncated):
            # Predict greedily again
            action = self.predict_greedily(observations)
            # Take the action
            observations, reward, terminated, truncated, info = self.watch_env.step(action)
            episode_return += reward
            # Display the performance for humans to see
            self.watch_env.render()
        print(f"Episode {episode+1} return: {episode_return:.3f}")

# Initialise agent, train it and test it.
agent = PPO(train_env, eval_env=eval_env, watch_env=watch_env)
agent.train()
agent.test_and_watch(n_episodes = 100)
