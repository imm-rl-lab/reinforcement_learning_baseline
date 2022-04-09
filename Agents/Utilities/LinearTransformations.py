import torch


def transform_interval(value, value_min, value_max):
    'linear transformation making from the interval [-1, 1] the interval [value_min, value_max]'
    if type(value) is torch.Tensor:
        value_min = torch.FloatTensor(value_min)
        value_max = torch.FloatTensor(value_max)

    return value * (value_max - value_min) / 2 + (value_max + value_min) / 2
