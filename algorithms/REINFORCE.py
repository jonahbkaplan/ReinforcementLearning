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
        tau, done, truncated = [], False, False                                                                         # Initialise empty trajectory and exit flags
        obs, info = self.env.reset()                                                                                    # Get current state
        while not (done or truncated):                                                                                  # Until a terminal state is reached
            action = self.predict(obs)                                                                                  # Get next action based on agent's policy
            new_obs, reward, done, truncated, info = self.env.step(action)                                              # Take action and observe environment
            tau += [[action, obs, reward]]                                                                              # Add observation to trajectory
            obs = new_obs                                                                                               # Update state
        return tau                                                                                                      # Return trajectory

    def __estimate_value(self, state):
        tensor = torch.tensor(state.flatten(), dtype=torch.float32)                                                     # Convert the state into a float tensor
        return self.__value_network(tensor).detach()                                                                    # Return the estimated state value (detached from the NN)

    def __update_parameters(self, s, a, d, t):
        # Adapted from https://epichka.com/blog/2023/pg-math-explained/
        action_probabilities = self.__policy_network(torch.tensor(s.flatten(), dtype=torch.float32))                    # Compute the action probabilities
        sampler = torch.distributions.Categorical(action_probabilities)                                                 # Convert the probabilities to a distribution
        log_probabilities = -sampler.log_prob(a)                                                                        # Compute the log of the probability of the action taken
        utility = log_probabilities * (self.__discount ** t) * d                                                        # Multiply by discounted return-to-go
        self.__policy_optimiser.zero_grad()                                                                             # Clear the gradients
        utility.backward()                                                                                              # Compute gradients
        self.__policy_optimiser.step()                                                                                  # Gradient Ascent
        #todo value net update
        #todo continuous action space update
        #todo tensor everything to do in parallel

    def predict(self, obs):
        tensor = torch.tensor(obs.flatten(), dtype=torch.float32)                                                       # Convert the state into a float tensor
        if self.__discrete_actions:                                                                                     # If discrete action space...
            action = self.__policy_network(tensor)                                                                      # Compute the action from the policy net
            action = torch.distributions.Categorical(action).sample()                                                   # Select index of highest value action
        else:                                                                                                           # If continuous action space...
            means, stds = self.__policy_network(tensor)                                                                 # Compute the mean and standard deviation of each type of action
            action = []                                                                                                 # Initialise storage for action values
            for action_type in range(len(means)):                                                                       # Loop over each type of action...
                action_sample = torch.distributions.Normal(means, stds).sample()                                        # Take a sample from the estimated distribution
                #todo scale to env space
                action += [action_sample]                                                                               # Append sampled actions to action list
        return action                                                                                                   # Return the policy's chosen action

    def learn(self):
        for episode in range(self.__episodes):                                                                          # Train using self.__episodes number of trajectories
            trajectory = self.__generate_trajectory()                                                                   # Generate a trajectory following the agent's current policy
            for step_t in range(0, len(trajectory)):                                                                    # Loop t from t=0 to t=T where T is the number of timesteps in the trajectory
                reward_to_go = 0                                                                                        # Initialise 'G' to 0
                for step_k in range(step_t + 1, len(trajectory) + 1):                                                   # Loop through each SAR of the sub-trajectory
                    reward_to_go += (self.__discount ** (step_k-step_t-1)) * trajectory[step_k-1][2]                    # add to the running sum, G
                delta = reward_to_go                                                                                    # Initialise new variable in case of baseline function
                if self.__flags[0]:                                                                                     # "If including baseline..."
                    delta -= self.__estimate_value(trajectory[step_t][1])                                               # delta = G - v(s_t, w)
                    self.__update_parameters(trajectory[step_t][1], trajectory[step_t][0], delta, step_t)               # w = w + (alpha_w * (discount ** t) * delta * Dv(s_t,w))
                self.__update_parameters(trajectory[step_t][1], trajectory[step_t][0], delta, step_t)                   # theta = theta + (alpha_theta * (discount ** t) * delta * Dlog(policy(A_t | S_t, Theta)))