import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from type_def import *
from layers.get_activation import get_activation

class Conditioner(nn.Module):
    '''
    An autoregressive conditioner.
    params = c(x_{1:i-1},y)
    '''
    def __init__(self, in_dim: int, cond_dim: int, out_param_dim: int,
                 h_dim: list=[], input_order: str='same', mode: str='sequential',
                 nonlinearity: str='elu', act_param: float=1.0):
        '''
        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        cond_dim : int
            Dimension of the conditional variable y.
        out_param_dim : int
            Dimension of the parameters.
        h_dim : list, optional
            List with number of hidden units for each hidden layer.
            The default is [].
        input_order : str, optional
            Strategy for assigning degrees to input units, which can be 'random',
            'same' or 'inverse'.
            The default is 'same'.
        mode : str, optional
            Strategy for assigning degrees to hidden units, which can be 'random'
            or 'sequential'.
            The default is 'sequential'.
        nonlinearity : str, optional
            Nonlinearity used in neural networks.
            The default is 'elu'.
        act_param : float, optional
            Parameter for some nonlinearity.
            The default is 1.0.
        '''
        super(Conditioner, self).__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.out_param_dim = out_param_dim
        self.out_dim = in_dim * out_param_dim
        self.h_dim = h_dim
        self.input_order = input_order
        self.mode = mode
        
        # Assign degrees to each unit
        degrees = self._assign_degrees()
        # Create masks
        masks = self._create_masks(degrees)
        
        # Create models
        activation = get_activation(nonlinearity, act_param)
        self.mlp = CondMaskedMLP(in_dim, cond_dim, self.out_dim, h_dim, masks, activation)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        degrees = self._assign_degrees()
        masks = self._create_masks(degrees)
        self.mlp.reset_parameters(masks)
            
    def forward(self, input: Tensor, cond_var: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.mlp(input, cond_var)
        output = output.view(*output.size()[:-1], self.out_param_dim, self.in_dim)
        with torch.no_grad():
            dim = output.dim()
            perm = list(range(dim-2)) + [dim-1, dim-2]
        
        return output.permute(perm)
    
    def _assign_degrees(self) -> list:
        '''
        Assign a degree for each hidden and input unit. A unit with degree d can only receive input from units with
        degree less than d.
        '''
        degrees = []
        if self.input_order == 'random':
            degrees_0 = torch.randperm(self.in_dim) + 1
        elif self.input_order == 'same':
            degrees_0 = torch.arange(1, self.in_dim+1)
        elif self.input_order == 'inverse':
            degrees_0 = torch.arange(self.in_dim, 0, -1)
        else:
            raise ValueError("Unsupported input_order {}".format(self.input_order))
        degrees.append(degrees_0)
        
        if self.mode == 'random':
            for N in self.h_dim:
                min_prev_degree = torch.min(degrees[-1])
                degrees_l = torch.randint(min_prev_degree, max(self.in_dim,2), N)
                degrees.append(degrees_l)
        elif self.mode == 'sequential':
            for N in self.h_dim:
                degrees_l = torch.arange(N) % max(self.in_dim-1, 1) + 1
                degrees.append(degrees_l)
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        
        return degrees
    
    def _create_masks(self, degrees: list) -> Tuple[list, Tensor]:
        '''
        Create binary masks that make the connectivity autoregressive.
        '''
        masks = []
        for d0, d1 in zip(degrees[:-1], degrees[1:]):
            masks.append(d0.unsqueeze(0) <= d1.unsqueeze(1))
            
        output_mask = degrees[-1].unsqueeze(0) < degrees[0].unsqueeze(1)
        masks.append(output_mask.repeat(self.out_param_dim, 1))
        
        return masks
        
class CondMaskedMLP(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, out_dim: int, h_dim: list,
                 masks: Tensor, activation):
        '''
        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        cond_dim : int
            Dimension of the conditional variable y.
        out_dim : int
            Dimension of the output.
        h_dim : list
            List with number of hidden units for each hidden layer.
        masks : Tensor
            Masks for weights.
        activation
            Nonlinearity function used in neural networks.
        '''
        super(CondMaskedMLP, self).__init__()
        self.activation = activation
        
        self.fcs = nn.ModuleList()
        self.masks = nn.ParameterList()
        
        if h_dim:
            in_dims = [in_dim] + h_dim[:-1]
            out_dims = h_dim
            next_dim = h_dim[-1]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                self.masks.append(nn.Parameter(masks[i], requires_grad=False))
            self.fc_y = nn.Linear(cond_dim, h_dim[0])
        else:
            next_dim = in_dims
            self.fc_y = nn.Linear(cond_dim, out_dim)
        self.fcs.append(nn.Linear(next_dim, out_dim))
        self.masks.append(nn.Parameter(masks[-1], requires_grad=False))
                
        self.reset_parameters()
        
    def reset_parameters(self, masks: Tensor=None):
        for l in self.fcs:
            l.reset_parameters()
        
        if masks:
            for i, m in enumerate(masks):
                self.masks[i].data = m
        nn.init.kaiming_uniform_(self.fc_y.weight, a=math.sqrt(5))
        nn.init.constant_(self.fc_y.bias, 0.0)
    
    def forward(self, input: Tensor, cond_var: Tensor):
        if len(self.fcs) == 1:
            return masked_linear(self.fcs[0], self.masks[0], input) + self.fc_y(cond_var)
        
        input = self.activation(masked_linear(self.fcs[0], self.masks[0], input)
                                + self.fc_y(cond_var))
        for i in range(1, len(self.fcs)-1):
            input = self.activation(masked_linear(self.fcs[i], self.masks[i], input))
        input = masked_linear(self.fcs[-1], self.masks[-1], input)
            
        return input

def masked_linear(fc, mask: Tensor, input: Tensor) -> Tensor:
    '''
    A Linear layer with mask.
    '''
    return F.linear(input, fc.weight * mask, fc.bias)