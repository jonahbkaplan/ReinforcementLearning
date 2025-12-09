from algorithms.Agent import Agent
from algorithms.Network import StateValueNN as ValueNet
from algorithms.Network import DiscretePolicyNN as DiscretePNN
from algorithms.Network import ContinuousPolicyNN as ContinuousPNN
from gymnasium import spaces
import numpy as np
import torch


class Reinforce(Agent):
    def __init__(self, env, episodes=100, discount=0.9, step_size_theta=2e-9, step_size_w=2e-6, flags=None):
        """
        REINFORCE with optional baseline function.

        :param Object env: the environment to act in
        :param Integer episodes: Number of episodes to run the training algorithm
        :param Real discount: Discount factor (Gamma)
        :param Real step_size_theta: Step size for parameter updates (Alpha_Theta)
        :param Real step_size_w: Step size for baseline function updates (Alpha_w)
        :param Array flags: index 0 indicates true or false to include baseline function
        """
        super().__init__(env)                                                                                           # Store the environment object
        self.__episodes = episodes                                                                                      # Store the number of training episodes
        self.__discount = discount                                                                                      # Store the discount factor
        self.__flags = flags                                                                                            # Store the state of the flags
        if flags is None:                                                                                               # If no flag argument used...
            self.__flags = [0]                                                                                          # Initialise a flag array with default flags
        elif flags[0]:                                                                                                  # If the first flag is true...
            self.__value_network = ValueNet(np.prod(env.observation_space.shape))                                       # Store the value estimation neural network
            self.__value_optimiser = torch.optim.SGD(self.__value_network.parameters(), lr=step_size_w)                 # Initialise baseline function optimiser
        self.__discrete_actions = (type(env.action_space) == spaces.Discrete)                                           # Determine the type of action space
        if self.__discrete_actions:                                                                                     # If the action space is discrete...
            self.__policy_network = DiscretePNN(np.prod(env.observation_space.shape), env.action_space.n)               # Create and store a policy NN for discrete action space
        else:                                                                                                           # If the action space is continuous...
            self.__policy_network = ContinuousPNN(np.prod(env.observation_space.shape), env.action_space.shape[0])      # Create and store a policy NN for continuous action space
        self.__policy_optimiser = torch.optim.SGD(self.__policy_network.parameters(), lr=step_size_theta)               # Initialise policy estimation optimiser

    def __generate_trajectory(self):
        states, actions, rewards, done, truncated = [], [], [], False, False                                            # Initialise empty trajectories and exit flags
        state, info = self.env.reset()                                                                                  # Get current state
        while not (done or truncated):                                                                                  # Until a terminal state is reached
            action = self.predict(state)                                                                                # Get next action based on agent's policy
            new_state, reward, done, truncated, info = self.env.step(action)                                            # Take action and observe environment
            states.append(torch.tensor(state.flatten(), dtype=torch.float32))                                           # Add state to trajectory
            actions.append(action)                                                                                      # Add action to trajectory
            rewards.append(reward)                                                                                      # Add reward to trajectory
            state = new_state                                                                                           # Update state
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards)                                        # Return trajectory

    def predict(self, obs):
        tensor = torch.tensor(obs.flatten(), dtype=torch.float32)                                                       # Convert the state into a float tensor
        if self.__discrete_actions:                                                                                     # If discrete action space...
            action = self.__policy_network(tensor)                                                                      # Compute the action from the policy net
            action = torch.distributions.Categorical(action).sample().item()                                            # Select index of highest value action
        else:                                                                                                           # If continuous action space...
            means, log_stds = self.__policy_network(tensor)                                                             # Compute the mean and log standard deviation of each type of action
            stds = log_stds.exp()                                                                                       # Convert log stds to regular stds
            dists = torch.distributions.Normal(means, stds)                                                             # Convert distribution parameters to distributions
            dists = torch.distributions.Independent(dists, 1)                                      # Treat action dimensions jointly
            action = dists.sample()                                                                                     # Take samples of each action dimension from the distributions
        return action                                                                                                   # Return the policy's chosen action

    def learn(self, verbose=False):
        undiscounted_rewards, average_rewards = [], []                                                                  # Initialise storage for evaluation metrics
        for episode in range(self.__episodes):                                                                          # Train using self.__episodes number of trajectories
            states_tensor, actions_tensor, rewards_tensor = self.__generate_trajectory()                                # Generate a trajectory following the agent's current policy
            timesteps = rewards_tensor.size(dim=0)                                                                       # Store the length of this episode's trajectory
            discounts_tensor = torch.tensor([self.__discount**t for t in range(timesteps)], dtype=torch.float32)        # Generate discounts tensor
            discounted_rewards = [discounts_tensor[:timesteps-t] * rewards_tensor[t:] for t in range(timesteps)]        # Apply discounts to returns
            r_t_g_tensor = torch.tensor([torch.sum(discounted_rewards[t]) for t in range(timesteps)], dtype=torch.float32) # Compute rewards to go
            if self.__flags[0]:                                                                                         # If including baseline...
                values = self.__value_network(states_tensor)                                                            # Collect V(s_t, w) for all t
                r_t_g_tensor = r_t_g_tensor - values.detach()                                                           # delta = G - v(s_t, w)
                w_loss = torch.sum(values * r_t_g_tensor)                                                               # Apply the weights: (alpha_w * (discount ** t) * delta) and sum the losses for all timesteps
                self.__value_optimiser.zero_grad()                                                                      # Clear the gradients
                w_loss.backward()                                                                                       # Compute gradients
                self.__value_optimiser.step()                                                                           # Gradient ascent: w += (alpha_w * (discount ** t) * delta * Dv(s_t,w))
            action_probabilities = self.__policy_network(states_tensor)                                                 # Compute the action probabilities for each state
            if self.__discrete_actions:                                                                                 # If the environment uses a discrete action space...
                distributions = torch.distributions.Categorical(action_probabilities)                                   # Create categorical distributions from action probabilities
            else:                                                                                                       # If continuous action space...
                distributions = None # TODO continuous action space
            log_probabilities = - distributions.log_prob(actions_tensor)                                                # Compute the log of the probability of the actions taken (minus for gradient ascent)
            theta_loss = torch.sum(log_probabilities * r_t_g_tensor)                                                    # Apply the weights: (alpha_theta * (discount ** t) * delta) and sum the losses for all timesteps
            self.__policy_optimiser.zero_grad()                                                                         # Clear the gradients
            theta_loss.backward()                                                                                       # Compute gradients
            self.__policy_optimiser.step()                                                                              # Gradient ascent: theta += (alpha_theta * (discount ** t) * delta * Dlog(policy(A_t | S_t, Theta)))
            undiscounted_rewards.append(torch.sum(rewards_tensor))                                                      # Append episode metrics
            average_rewards.append(torch.sum(rewards_tensor)/timesteps)                                                 # Append episode metrics
            if verbose:                                                                                                 # If verbose...
                print(f"Episode: {episode + 1}, Reward: {undiscounted_rewards[episode]:.2f}")                           # Update user of progress
        return undiscounted_rewards, average_rewards                                                                    # Return metrics for evaluation