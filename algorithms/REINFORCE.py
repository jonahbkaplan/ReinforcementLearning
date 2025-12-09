from algorithms.Agent import Agent
from algorithms.Network import StateValueNN as ValueNet
from algorithms.Network import DiscretePolicyNN as DiscretePNN
from algorithms.Network import ContinuousPolicyNN as ContinuousPNN
from gymnasium import spaces
import numpy as np
import torch


class Reinforce(Agent):
    def __init__(self, env, episodes=100, discount=0.9, step_size_theta=2**-13, step_size_w=2e-5, flags=None):
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
        # Metrics
        undiscounted_rewards, average_rewards = [], []

        # Training
        for episode in range(self.__episodes):
            # Generate a sample trajectory using the current policy
            states_tensor, actions_tensor, rewards_tensor = self.__generate_trajectory()
            # Store the length of this episode's trajectory
            timesteps = rewards_tensor.size(dim=0)

            # Step through the trajectory
            for t in range(timesteps):
                ## ACCURACY AT COST OF SPEED:
                # Compute reward-to-go
                discounts = self.__discount ** torch.arange(timesteps - t)
                r_t_g = torch.sum(discounts * rewards_tensor[t:])

                # If using baseline, update r_t_g to delta and update w parameters
                if self.__flags[0]:
                    pass # TODO once without baseline stabilised

                # Updates theta parameters
                action_probabilities = self.__policy_network(states_tensor[t])
                # Get log probability of action take
                if self.__discrete_actions:
                    dist = torch.distributions.Categorical(action_probabilities)
                else:
                    pass # TODO once discrete action stabilised
                log_probability = - dist.log_prob(actions_tensor[t])

                # Apply loss for gradient ascent
                theta_loss = log_probability * r_t_g

                # Complete the parameter update
                self.__policy_optimiser.zero_grad()
                theta_loss.backward()
                self.__policy_optimiser.step()

            # Append new metrics
            undiscounted_rewards.append(torch.sum(rewards_tensor))
            average_rewards.append(torch.sum(rewards_tensor) / timesteps)

            # Report to user
            if verbose:
                print(f"Episode: {episode + 1}, Reward: {undiscounted_rewards[episode]:.2f}")

        # Return metrics for evaluation
        return undiscounted_rewards, average_rewards

            #discounts_tensor = torch.tensor([self.__discount**t for t in range(timesteps)], dtype=torch.float32)        # Generate discounts tensor
            #discounted_rewards = [discounts_tensor[:timesteps-t] * rewards_tensor[t:] for t in range(timesteps)]        # Apply discounts to returns
            #r_t_g_tensor = torch.tensor([torch.sum(discounted_rewards[t]) for t in range(timesteps)], dtype=torch.float32) # Compute rewards to go
            #if self.__flags[0]:                                                                                         # If including baseline...
                #values = self.__value_network(states_tensor)                                                            # Collect V(s_t, w) for all t
                #r_t_g_tensor = r_t_g_tensor - values.detach()                                                           # delta = G - v(s_t, w)
                #w_loss = torch.sum(values * r_t_g_tensor)                                                               # Apply the weights: (alpha_w * (discount ** t) * delta) and sum the losses for all timesteps
                #self.__value_optimiser.zero_grad()                                                                      # Clear the gradients
                #w_loss.backward()                                                                                       # Compute gradients
                #self.__value_optimiser.step()                                                                           # Gradient ascent: w += (alpha_w * (discount ** t) * delta * Dv(s_t,w))
            #action_probabilities = self.__policy_network(states_tensor)                                                 # Compute the action probabilities for each state

if __name__ == '__main__':
    import gymnasium as gym
    import highway_env
    from matplotlib import pyplot as plt

    # Controls:
    CONTINUOUS_ACTIONS = False
    TESTING = True
    EPISODES = 1000
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

    model = Reinforce(env, episodes=EPISODES, flags=[0])
    rs, ars = model.learn(verbose=True)

    #done = truncated = False
    #obs, info = env.reset()
    #while not (done or truncated):
    #    action = model.predict(obs)
    #    obs, reward, done, truncated, info = env.step(action)
    #    env.render()

    #plt.imshow(env.render())
    #plt.show()
    #env.close()

    plt.plot(rs)
    plt.title("Undiscounted Total Rewards vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Undiscounted Total Reward")

    #axes[1].plot(ars)
    #axes[1].set_title("Average Reward vs Episodes")
    #axes[1].set_xlabel("Episode")
    #axes[1].set_ylabel("Average Reward")

    plt.show()