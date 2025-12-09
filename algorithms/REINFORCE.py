from algorithms.Agent import Agent
from algorithms.Network import StateValueNN as ValueNet
from algorithms.Network import DiscretePolicyNN as DiscretePNN
from algorithms.Network import ContinuousPolicyNN as ContinuousPNN
from gymnasium import spaces
import numpy as np
import torch


class Reinforce(Agent):
    def __init__(self, env, episodes=100, discount=0.9, step_size_theta=2**-9, step_size_w=2**-6, flags=None):
        """
        REINFORCE with optional baseline function.

        :param Object env: the environment to act in
        :param Integer episodes: Number of episodes to run the training algorithm
        :param Real discount: Discount factor (Gamma)
        :param Real step_size_theta: Step size for parameter updates (Alpha_Theta)
        :param Real step_size_w: Step size for baseline function updates (Alpha_w)
        :param Array flags: index 0 indicates true or false to include baseline function
        """
        # Store parameters required for training and predicting
        super().__init__(env)
        self.__episodes = episodes
        self.__discount = discount
        self.__flags = flags

        # If no flags given (currently only for baseline function)
        if flags is None:
            # Initialise array with default flags
            self.__flags = [0]
        elif flags[0]:
            # If using baseline function, initialise value approximator and optimiser
            self.__value_network = ValueNet(np.prod(env.observation_space.shape))
            self.__value_optimiser = torch.optim.SGD(self.__value_network.parameters(), lr=step_size_w)

        # Determine the type of action space
        self.__discrete_actions = (type(env.action_space) == spaces.Discrete)
        if self.__discrete_actions:
            # Create and store a policy NN for discrete action space
            self.__policy_network = DiscretePNN(np.prod(env.observation_space.shape), env.action_space.n)
        else:
            # Create and store a policy NN for continuous action space
            self.__policy_network = ContinuousPNN(np.prod(env.observation_space.shape), env.action_space.shape[0])
            # Initialise storage for raw action
            self.__raw_action = None
            # Determine and store the bounds of each action dimension
            self.__lower_bounds, self.__upper_bounds = torch.tensor(env.action_space.low), torch.tensor(env.action_space.high)
        # Initialise policy estimation optimiser
        self.__policy_optimiser = torch.optim.SGD(self.__policy_network.parameters(), lr=step_size_theta)

    def __generate_trajectory(self):
        # Initialise empty trajectories and exit flags
        states, actions, raw_actions, rewards, done, truncated = [], [], [], [], False, False
        # Get current state
        state, info = self.env.reset()

        # Until a terminal state is reached
        while not (done or truncated):
            # Get next action based on agent's policy, with chance of exploration
            action = self.predict(state, greedy=False)

            # If continuous action space, get raw action and append to storage
            if not self.__discrete_actions:
                raw_actions.append(self.__raw_action)

            # Take action and observe environment
            new_state, reward, done, truncated, info = self.env.step(action)

            # Add state, action, and reward to trajectories
            states.append(torch.tensor(state.flatten(), dtype=torch.float32))
            actions.append(action)
            rewards.append(reward)

            # Update state
            state = new_state
        # Return trajectory
        return torch.stack(states), torch.tensor(actions), torch.tensor(raw_actions), torch.tensor(rewards)

    def predict(self, obs, greedy=True):
        # Convert the state into a float tensor
        tensor = torch.tensor(obs.flatten(), dtype=torch.float32)

        # If discrete action space...
        if self.__discrete_actions:
            # Compute action probabilities
            action = self.__policy_network(tensor)
            # Return best action if greedy
            if greedy:
                return torch.argmax(action).item()
            else:
                # Otherwise return a sample from the distribution
                return torch.distributions.Categorical(action).sample().item()

        # If continuous action space...
        else:
            # Compute the mean and log standard deviation of each type of action
            means, log_stds = self.__policy_network(tensor)
            # Convert log stds to regular stds
            stds = log_stds.exp()
            # Return best actions if greedy
            if greedy:
                actions = means
            else:
                # Otherwise return samples from the distribution
                dists = torch.distributions.Normal(means, stds)
                # Treat action dimensions jointly
                dists = torch.distributions.Independent(dists, 1)
                # Sample action
                actions = dists.rsample()
            # Store raw action for gradient ascent
            self.__raw_action = actions
            # Scale actions to [-1,1]
            actions = torch.tanh(actions)
            # Stretch actions to action space (from [-1,1] to [lower,upper])
            actions = self.__lower_bounds + (actions+1)*0.5*(self.__upper_bounds-self.__lower_bounds)
            # Take samples of each action dimension from the distributions
            return actions.detach()

    def learn(self, verbose=False):
        # Metrics
        undiscounted_rewards, average_rewards = [], []

        # Training
        for episode in range(self.__episodes):
            # Generate a sample trajectory using the current policy
            states_tensor, actions_tensor, raw_actions_tensor, rewards_tensor = self.__generate_trajectory()
            # Store the length of this episode's trajectory
            timesteps = rewards_tensor.size(dim=0)

            # Step through the trajectory
            for t in range(timesteps):
                # Compute reward-to-go
                discounts = self.__discount ** torch.arange(timesteps - t)
                r_t_g = torch.sum(discounts * rewards_tensor[t:])

                # If using baseline, update r_t_g to delta and update w parameters
                if self.__flags[0]:
                    # Get the state's value estimate
                    value = self.__value_network(states_tensor[t])
                    # Update reward-to-go: delta = G - v(s_t, w)
                    r_t_g = r_t_g - value.detach()

                    # Apply loss for gradient ascent and update parameters
                    w_loss = - value * r_t_g
                    self.__value_optimiser.zero_grad()
                    w_loss.backward()
                    self.__value_optimiser.step()

                # Updates theta parameters
                # Get log probability of action take
                if self.__discrete_actions:
                    action_probabilities = self.__policy_network(states_tensor[t])
                    dist = torch.distributions.Categorical(action_probabilities)
                    log_probability = dist.log_prob(actions_tensor[t])
                else:
                    means, log_stds = self.__policy_network(states_tensor[t])
                    stds = log_stds.exp()
                    dist = torch.distributions.Normal(means, stds)
                    dist = torch.distributions.Independent(dist, 1)

                    # Jacobian correction required due to tanh squashing (non-linear transformation)
                    log_probability = (dist.log_prob(raw_actions_tensor[t]) - torch.log(1 - torch.tanh(raw_actions_tensor[t]).pow(2) + 1e-6)).sum(-1)

                # Apply loss for gradient ascent
                theta_loss = -log_probability * r_t_g

                # Complete the parameter update
                self.__policy_optimiser.zero_grad()
                theta_loss.backward() # TODO fix
                self.__policy_optimiser.step()

            # Append new metrics
            undiscounted_rewards.append(torch.sum(rewards_tensor))
            average_rewards.append(torch.sum(rewards_tensor) / timesteps)

            # Report to user
            if verbose:
                print(f"Episode: {episode + 1}, Reward: {undiscounted_rewards[episode]:.2f}")

        # Return metrics for evaluation
        return undiscounted_rewards, average_rewards


if __name__ == '__main__':
    import gymnasium as gym
    import highway_env
    from matplotlib import pyplot as plt

    # Controls:
    CONTINUOUS_ACTIONS = True
    TESTING = True
    EPISODES = 1
    BASELINE = True
    ###########

    if CONTINUOUS_ACTIONS and TESTING:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
    elif CONTINUOUS_ACTIONS and not TESTING:
        env = gym.make('highway-fast-v0', config={"action": {"type": "ContinuousAction"}}, render_mode='rgb_array')
    elif not CONTINUOUS_ACTIONS and TESTING:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    elif not CONTINUOUS_ACTIONS and not TESTING:
        env = gym.make('highway-fast-v0', render_mode='rgb_array')
    else:
        print("invalid controls, aborting...")
        quit()

    model = Reinforce(env, episodes=EPISODES, flags=[BASELINE])
    rs, ars = model.learn(verbose=True)

    plt.plot(rs)
    plt.title(f"Undiscounted Total Rewards vs Episodes ({"using" if BASELINE else "without"} baseline)")
    plt.xlabel("Episode")
    plt.ylabel("Undiscounted Total Reward")
    plt.show()