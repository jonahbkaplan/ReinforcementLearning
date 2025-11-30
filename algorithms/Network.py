import torch
import torch.nn as nn
import torch.nn.functional as nnf

class StateValueNN(nn.Module):
    # TODO experiment to find best value estimation structure
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.__layer_in = nn.Linear(input_size, hidden_size)
        self.__layers = []
        self.__layer_out = nn.Linear(hidden_size, 1)

    def forward(self, state_tensor):
        x = nnf.relu(self.__layer_in(state_tensor))
        for layer in self.__layers:
            x = nnf.relu(layer(x))
        return self.__layer_out(x).squeeze(-1) # Squeeze converts output to scalar (-1 prevents it collapsing the batch size)


class PolicyNN(nn.Module):
    # TODO create proper architecture for policy estimation
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.__layer_in = nn.Linear(input_size, hidden_size)
        self.__layers = []
        self.__layer_out = nn.Linear(hidden_size, output_size)