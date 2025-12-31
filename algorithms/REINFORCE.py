from algorithms.Agent import Agent
from algorithms.Network import StateValueNN, DiscretePolicyNN, ContinuousPolicyNN
from gymnasium import spaces
import numpy as np
import torch


class Reinforce(Agent):
    def __init__(self, env, epochs=100, episodes_per_epoch=10, discount=0.9, step_size_theta=2**-9, step_size_w=2**-6):
        """
        REINFORCE with optional baseline function.

        :param Object env: the environment to act in
        :param Integer epochs: Number of epochs to run the training algorithm
        :param Integer episodes_per_epoch: Number of episodes used per epoch
        :param Real discount: Discount factor (Gamma)
        :param Real step_size_theta: Step size for parameter updates (Alpha_Theta)
        :param Real step_size_w: Step size for baseline function updates (Alpha_w)
        """
        super().__init__(env)
        self.__epochs = epochs
        self.__episodes = episodes_per_epoch
        self.__discount = discount
        self.__discrete_actions = (type(env.action_space) == spaces.Discrete)
        input_size = np.prod(env.observation_space.shape)
        output_size = env.action_space.n if self.__discrete_actions else env.action_space.shape[0]
        self.__policy_network = DiscretePolicyNN(input_size, output_size) if self.__discrete_actions else ContinuousPolicyNN(input_size, output_size)
        self.__policy_optimiser = torch.optim.Adam(self.__policy_network.parameters(), lr=step_size_theta)
        self.__value_network = StateValueNN(np.prod(env.observation_space.shape))
        self.__value_optimiser = torch.optim.Adam(self.__value_network.parameters(), lr=step_size_w)
        self.__raw_action = None
        # Determine and store the bounds of each action dimension
        # self.__lower_bounds, self.__upper_bounds = torch.tensor(env.action_space.low), torch.tensor(env.action_space.high)


    def __generate_trajectory(self):
        states, actions, log_probs, rewards, done, truncated = [], [], [], [], False, False
        state, info = self.env.reset()
        while not (done or truncated):
            # Get next action based on agent's policy, with chance of exploration
            action, log_prob = self.predict(state, greedy=False)
            new_state, reward, done, truncated, info = self.env.step(action)
            # Add state, action, and reward to trajectories and store raw action for loss calculation
            states.append(torch.tensor(state.flatten(), dtype=torch.float32))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = new_state
        return torch.stack(states), torch.tensor(actions), torch.stack(log_probs), torch.tensor(rewards)


    def predict(self, obs, greedy=True):
        state_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
        if self.__discrete_actions:
            action = self.__policy_network(state_tensor)
            if greedy:
                return torch.argmax(action).item(), 0
            else:
                dist = torch.distributions.Categorical(action)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob
        else:
            means, log_stds = self.__policy_network(state_tensor)
            if greedy:
                return torch.tanh(means).detach(), [0 for _ in range(len(means))]
            log_stds = torch.clamp(log_stds, min=-20, max=2) # Clamp for stability
            stds = log_stds.exp() # Convert log stds to regular stds
            # If continuous action space, sample distributions and scale to [-1,1]
            dists = torch.distributions.Normal(means, stds)
            dists = torch.distributions.Independent(dists, 1) # Treat action dimensions jointly
            actions = dists.sample()
            # Jacobian correction required due to tanh squashing (non-linear transformation)
            log_probability = (dists.log_prob(actions) - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6))
            return torch.tanh(actions).detach(), log_probability


    def learn(self, verbose=False):
        undiscounted_rewards, average_rewards = [], []
        for epoch in range(self.__epochs):
            epoch_states, epoch_log_probs, epoch_rtgs, epoch_rewards = [], [], [], []
            for _ in range(self.__episodes):
                states_tensor, actions_tensor, log_prob_tensor, rewards_tensor = self.__generate_trajectory()
                discounts = self.__discount ** torch.arange(rewards_tensor.size(dim=0))
                # Cumulative sum for rewards to go
                rtgs = torch.flip(torch.cumsum(torch.flip(rewards_tensor * discounts, dims=[0]), dim=0),dims=[0]) / discounts
                epoch_states.append(states_tensor)
                epoch_log_probs.append(log_prob_tensor)
                epoch_rtgs.append(rtgs)
                epoch_rewards.append(torch.sum(rewards_tensor))
            # Concatenate epoch episodes
            states_tensor = torch.cat(epoch_states)
            log_prob_tensor = torch.cat(epoch_log_probs)
            rtgs_tensor = torch.cat(epoch_rtgs)
            values = self.__value_network(states_tensor)
            advantages = rtgs_tensor - values.detach()
            # Apply loss for gradient ascent and update parameters
            w_loss = -(values * advantages).mean()
            self.__value_optimiser.zero_grad()
            w_loss.backward()
            self.__value_optimiser.step()
            theta_loss = -(log_prob_tensor * advantages).mean()
            self.__policy_optimiser.zero_grad()
            theta_loss.backward()
            self.__policy_optimiser.step()
            average_rewards.append(np.mean(epoch_rewards))
            undiscounted_rewards.append(epoch_rewards)
            if verbose:
                print(f"Epoch: {epoch + 1}, Average Reward: {average_rewards[-1]:.2f}")
        return undiscounted_rewards, average_rewards # Return metrics for evaluation
