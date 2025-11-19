from algorithms.Agent import Agent
from stable_baselines3 import DQN

class StableBaselineDQN(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
        self.model.learn(int(2e4))
        self.model.save("highway_dqn/model")


    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)

    def learn(self) :
        pass