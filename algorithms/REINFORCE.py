from algorithms.Agent import Agent
from algorithms.Network import StateValueNN as ValueNet
from algorithms.Network import DiscretePolicyNN as DiscretePNN
from algorithms.Network import ContinuousPolicyNN as ContinuousPNN
from gymnasium import spaces
import numpy as np
import torch


class Reinforce(Agent):
    def __init__(self, env, episodes=100, discount=0.9, step_size_theta=2e-13, step_size_w=2e-13, flags=None):
        """
        REINFORCE with optional baseline function.

        :param Object env: the environment to act in
        :param Integer episodes: Number of episodes to run the training algorithm
        :param Real discount: Discount factor (Gamma)
        :param Real step_size_theta: Step size for parameter updates (Alpha_Theta)
        :param Real step_size_w: Step size for baseline function updates (Alpha_w)
        :param Array flags: index 0 indicates true or false to include baseline function
        """
        super().__init__(env)                                                                                       # Store the environment object
        self.__episodes = episodes                                                                                  # Store the number of training episodes
        self.__discount = discount                                                                                  # Store the discount factor
        self.__step_size_theta = step_size_theta                                                                    # Store the parameter stepsize
        self.__flags = flags                                                                                        # Store the state of the flags
        if flags is None:                                                                                           # If no flag argument used...
            self.__flags = [0]                                                                                      # Initialise a flag array with default flags
        elif flags[0]:                                                                                              # If the first flag is true...
            self.__step_size_w = step_size_w                                                                        # Store the baseline function stepsize
            self.__value_network = ValueNet(np.prod(env.observation_space.shape))                                   # Store the value estimation neural network
        self.__discrete_actions = (type(env.action_space) == spaces.Discrete)                                       # Determine the type of action space
        if self.__discrete_actions:                                                                                 # If the action space is discrete...
            self.__policy_network = DiscretePNN(np.prod(env.observation_space.shape), env.action_space.n)           # Create and store a policy NN for discrete action space
        else:                                                                                                       # If the action space is continuous...
            self.__policy_network = ContinuousPNN(np.prod(env.observation_space.shape), env.action_space.shape[0])  # Create and store a policy NN for continuous action space

    def __generate_trajectory(self):
        tau, done, truncated = [], False, False                             # Initialise empty trajectory and exit flags
        obs, info = self.env.reset()                                        # Get current state
        while not (done or truncated):                                      # Until a terminal state is reached
            action = self.predict(obs)                                      # Get next action based on agent's policy
            new_obs, reward, done, truncated, info = self.env.step(action)  # Take action and observe environment
            tau += [[action, obs, reward]]                                  # Add observation to trajectory
            obs = new_obs                                                   # Update state
        return tau                                                          # Return trajectory

    def __estimate_value(self, state):
        tensor = torch.tensor(state.flatten(), dtype=torch.float32) # Convert the state into a float tensor
        return self.__value_network(tensor).detach()                # Return the estimated state value (detached from the NN)

    def __update_parameters(self, d, t=None):
        network = self.__value_network if t is None else self.__policy_network                  # Determine which network to update
        network.zero_grad()                                                                     # Zero the gradients before the backwards pass
        network.backward()                                                                      # TODO fix
        with torch.no_grad():                                                                   # Don't track parameter updates (prevents auto grad errors)
            for param in network.parameters():                                                  # Loop over and update the network parameters
                if t is None:                                                                   # If updating the value function network...
                    update = self.__step_size_w * d * param.grad                                # w = w + (alpha_w * delta * Dv(s_t,w))
                else:                                                                           # If updating the policy estimation network #TODO fix param.grad below to do log of prob of action
                    #print(f"update = {self.__step_size_theta} * ({self.__discount} ** {t}) * {d} * {param.grad}")
                    update = self.__step_size_theta * (self.__discount ** t) * d * param.grad   # theta = theta + (alpha_theta * (discount ** t) * delta * Dlog(policy(A_t | S_t, Theta)))
                param += update                                                                 # Update the network parameters (gradient ascent)

    def predict(self, obs):
        tensor = torch.tensor(obs.flatten(), dtype=torch.float32)   # Convert the state into a float tensor
        if self.__discrete_actions:                                 # If discrete action space...
            action = self.__policy_network(tensor)                  # Compute the action from the policy net
            action = torch.argmax(action)                           # Select index of highest value action (greedy) # TODO sample
        else:                                                       # If continuous action space...
            means, stds = self.__policy_network(tensor)             # Compute the mean and standard deviation of each type of action
            action = means                                          # Select means of each type of action (greedy) # TODO sample
        return action                                               # Return the policy's chosen action

    def learn(self):
        for episode in range(self.__episodes):                                                          # Train using self.__episodes number of trajectories
            trajectory = self.__generate_trajectory()                                                   # Generate a trajectory following the agent's current policy
            for step_t in range(0, len(trajectory)):                                                    # Loop t from t=0 to t=T where T is the number of timesteps in the trajectory
                reward_to_go = 0                                                                        # Initialise 'G' to 0
                for step_k in range(step_t + 1, len(trajectory) + 1):                                   # Loop through each SAR of the sub-trajectory
                    reward_to_go += (self.__discount ** (step_k-step_t-1)) * trajectory[step_k-1][2]    # add to the running sum, G
                delta = reward_to_go                                                                    # Initialise new variable in case of baseline function
                if self.__flags[0]:                                                                     # "If including baseline..."
                    delta -= self.__estimate_value(trajectory[step_t][1])                               # delta = G - v(s_t, w)
                    self.__update_parameters(delta)                                                     # w = w + (alpha_w * delta * Dv(s_t,w))
                self.__update_parameters(delta, step_t)                                                 # theta = theta + (alpha_theta * (discount ** t) * delta * Dlog(policy(A_t | S_t, Theta)))