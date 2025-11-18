from algorithms.Agent import Agent


class TestAgent(Agent):
    def policy(self, state, greedy=False):
        return self.env.unwrapped.action_type.actions_indexes["IDLE"]

    def learn(self, state_action_pairs, rewards, next_states):
        pass