from algorithms.Agent import Agent


class Reinforce(Agent):
    def __init__(self, env, episodes=10, discount=0.9, step_size_theta=2e-13, step_size_w=2e-13, flags=None):
        """
        REINFORCE with optional baseline function.

        :param Object env: the environment to act in
        :param Integer episodes: Number of episodes to run the training algorithm
        :param Real discount: Discount factor (Gamma)
        :param Real step_size_theta: Step size (Theta)
        :param Array flags: index 0 indicates true or false to include baseline
        """
        super().__init__(env)                       # Store the environment object
        self.__episodes = episodes                  # Store the number of training episodes
        self.__discount = discount                  # Store the discount factor
        self.__step_size_theta = step_size_theta    # Store the parameter stepsize
        if flags is None:                           # If no flag argument used...
            self.__flags = [0]                      # Initialise a flag array with default flags
        elif flags[0]:                              # If the first flag is true...
            self.__step_size_w = step_size_w        # Store the baseline function stepsize
            pass                                    #TODO state value parameterisation v(s,w)
        self.__theta = []                           #TODO Network parameters
        self.policy = 1                             #TODO

    def __generate_trajectory(self):
        tau, done, truncated = [], False, False                             # Initialise empty trajectory and exit flags
        obs, info = self.env.reset()                                        # Get current state
        while not (done or truncated):                                      # Until a terminal state is reached
            action = self.predict(obs)                                      # Get next action based on agent's policy
            new_obs, reward, done, truncated, info = self.env.step(action)  # Take action and observe environment
            tau += [[action, obs, reward]]                                  # Add observation to trajectory
            obs = new_obs                                                   # Update state
        return tau                                                          # Return trajectory

    def predict(self, obs):
        return self.policy  #TODO get action from obs (state)

    def learn(self):
        for episode in range(self.__episodes):                                                          # Train using self.__episodes number of trajectories
            trajectory = self.__generate_trajectory()                                                   # Generate a trajectory following the agent's current policy
            for step_t in range(0, len(trajectory)):                                                    # Loop t from t=0 to t=T where T is the number of timesteps in the trajectory
                reward_to_go = 0                                                                        # Initialise 'G' to 0
                for step_k in range(step_t + 1, len(trajectory) + 1):                                   # Loop through each SAR of the sub-trajectory
                    reward_to_go += (self.__discount ** (step_k-step_t-1)) * trajectory[step_k-1][2]    # add to the running sum, G
                delta = reward_to_go                                                                    # Initialise new variable in case of baseline function
                if self.__flags[0]:                                                                     # "If including baseline..."
                    pass                                                                                #TODO delta -= v(s_t, w)
                    pass                                                                                #TODO w = w + (step_size_w * delta * Dv(s_t,w)
                pass                                                                                    #TODO self.__theta = self.__theta + (self.__step_size_theta * (self.__discount ** step_t) * delta * D log policy(A_t | S_t, Theta))