from enum import Enum
import torch
from torch import nn
import numpy as np


class LayerType(Enum):
    LSTM = 'LSTM'
    Dense = 'Dense'


class SequentialNetworkWithTypes(nn.Module):

    def __init__(self, input_dim, layers, hidden_activation=nn.ReLU(), output_activation=None, device='cpu'):
        super().__init__()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.device = device

        self.layer_types = []
        self.layers = nn.ModuleList()
        self.lstm_size = 0

        for layer_type, layer_parameters in layers:

            if layer_type == LayerType.Dense:
                layer = nn.Linear(input_dim, layer_parameters, device=device)
                input_dim = layer_parameters

            elif layer_type == LayerType.LSTM:
                layer = nn.LSTMCell(input_dim, layer_parameters, device=device)
                self.lstm_size = layer_parameters
                input_dim = layer_parameters

            else:
                raise Exception(f'Layer \"{layer_type}\" does not exist')

            self.layers.append(layer)
            self.layer_types.append(layer_type)

        self.to(device)

    def forward(self, tensor, memory=None):
        hid = self.convert_input(tensor)
        for i, layer_type in enumerate(self.layer_types):
            is_output_layer = i == len(self.layer_types) - 1
            layer = self.layers[i]

            if layer_type == LayerType.Dense:
                hid = layer(hid)
                if not is_output_layer:
                    hid = self.hidden_activation(hid)

            elif layer_type == LayerType.LSTM:
                memory = layer(hid, memory)
                hid = memory[0]

        if self.output_activation:
            hid = self.output_activation(hid)

        if not memory:
            return hid

        return hid, memory

    def convert_input(self, tensor):
        if type(tensor) is torch.Tensor:
            return tensor
        if type(tensor).__module__ != np.__name__:
            tensor = np.array(tensor)
        return torch.tensor(tensor, dtype=torch.float, device=self.device)

    def get_initial_state(self, batch_size=1):
        return torch.zeros((batch_size, self.lstm_size), device=self.device),\
               torch.zeros((batch_size, self.lstm_size), device=self.device)
