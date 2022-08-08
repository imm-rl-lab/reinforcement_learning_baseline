import numpy as np
import torch
from torch import nn


class SequentialNetwork(nn.Module):
    def __init__(self, layers, hidden_activation=nn.ReLU(), output_activation=None, initial_weight_clip=3e-3, output_n=1):
        super().__init__()
        self.network = self.get_network(layers, hidden_activation, output_activation, initial_weight_clip)
        self.layers = layers
        self.output_n = output_n
        return None

    def get_network(self, layers, hidden_activation, output_activation, initial_weight_clip):
        network = []
        
        hidden_layers = layers[:-1]
        for layer, next_layer in zip(hidden_layers, hidden_layers[1:]):
            sub_network = nn.Linear(layer, next_layer)
            weight_clip = 1. / np.sqrt(sub_network.weight.data.size()[0])
            sub_network.weight.data.uniform_(- weight_clip, + weight_clip)
            network.append(sub_network)
            network.append(hidden_activation)
            
        last_sub_network = nn.Linear(layers[-2], layers[-1])
        last_sub_network.weight.data.uniform_(- initial_weight_clip, + initial_weight_clip)
        network.append(last_sub_network)

        if output_activation:
            network.append(output_activation)
        
        return nn.Sequential(*network)

    def forward(self, tensor):
        if type(tensor) is not torch.Tensor:
            tensor = torch.FloatTensor(tensor)
        if self.output_n==1:
            return self.network(tensor)
        else:
            output = self.network(tensor)
            split_size = int(self.layers[-1]/self.output_n)
            return torch.split(output, split_size, dim=1)
