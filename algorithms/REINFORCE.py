from algorithms.Agent import Agent


class Reinforce(Agent):
    # REINFORCE with a value-function estimator baseline
    # Ruaidhri

    def __init__(self, env, episodes=10, discount=0.9, step_size=2e-13, flags=None):
        """
        REINFORCE with optional baseline function

        :param env: the environment to act in
        :param episodes: Number of episodes to run the training algorithm
        :param discount: Discount factor (Gamma)
        :param step_size: Step size (Alpha)
        :param flags: index 0 indicates true or false to include baseline
        """
        super().__init__(env)

        self.__episodes = episodes
        self.__discount = discount
        self.__step_size = step_size

        if flags is None:
            self.__flags = [0]
        elif flags[0]:
            pass #TODO state value parameterisation v(s,w)

        self.__theta = [] # TODO Network parameters
        self.policy = 1 # TODO

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
        return self.policy #TODO get action from obs (state)

    def learn(self):
        for episode in range(self.__episodes):
            trajectory = self.__generate_trajectory()
            for step_t in range(0, len(trajectory)):
                reward_to_go = 0
                for step_k in range(step_t + 1, len(trajectory) + 1):
                    reward_to_go += (self.__discount ** (step_k-step_t-1)) * trajectory[step_k-1][2]
                if self.__flags[0]: # "If including baseline..."
                    #TODO delta = reward_to_go - v(s_t, w)
                    #TODO w = w + (step_size_w * delta * Dv(s_t,w)
                    #TODO self.__theta = self.__theta + (step_size_theta * (discount ** step_t) * delta * D log policy(A_t | S_t, Theta)
                    pass
                else: # "If not including baseline..."
                    self.__theta = self.__theta + (self.__step_size * (self.__discount ** step_t) * reward_to_go * 1) #TODO 1 = D log policy(A_t | S_t, Theta)