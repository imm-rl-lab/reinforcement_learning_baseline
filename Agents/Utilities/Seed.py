import numpy, random, torch


def seed(value):
    numpy.random.seed(value)
    random.seed(value)
    torch.manual_seed(value)
    return None
    