from algorithms.Agent import Agent


class TestAgent(Agent):
    def predict(self, obs):
        return self.env.unwrapped.action_type.actions_indexes["IDLE"]

    def learn(self) :
        pass