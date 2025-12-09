import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

class StateValueNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.__layer_in = nn.Linear(input_size, 100)
        self.__layer_out = nn.Linear(100, 1)

    def forward(self, state_tensor):
        x = nnf.relu(self.__layer_in(state_tensor))
        return self.__layer_out(x)


class DiscretePolicyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.__layer_in = nn.Linear(input_size, 100)
        self.__layer_out = nn.Linear(100, output_size)

    def forward(self, state_tensor):
        x = nnf.relu(self.__layer_in(state_tensor))
        action_values = self.__layer_out(x)
        action_probabilities = nnf.softmax(action_values, dim=-1)
        return action_probabilities


class ContinuousPolicyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.__layer_in = nn.Linear(input_size, 100)
        self.__layer_mean = nn.Linear(100, output_size)
        self.__layer_std = nn.Linear(100, output_size)

    def forward(self, state_tensor):
        x = nnf.relu(self.__layer_in(state_tensor))
        output_means = self.__layer_mean(x)
        output_stds = self.__layer_std(x)
        return output_means, output_stds