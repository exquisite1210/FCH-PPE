import torch.nn as nn

def get_activation(nonlinearity, param):
    if nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'leaky_relu':
        return nn.LeakyReLU(param, inplace=True)
    elif nonlinearity == 'elu':
        return nn.ELU(param, inplace=True)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))