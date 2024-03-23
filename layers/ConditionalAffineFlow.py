import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from type_def import *
from layers.get_activation import get_activation

class ConditionalAffineFlow(nn.Module):
    '''
    Affine Flow for modeling conditional probability p(x|y).
    '''
    def __init__(self, in_features: int, cond_features: int, nonlinearity_a: str='softpuls',
                 identity_init: bool=False):
        '''
        Parameters
        ----------
        in_features : int
            Dimension of the input.
        cond_features : int
            Dimension of the conditional variable y.
        nonlinearity_a : str, optional
            Nonlinearity for output a, which guarantees a are all positive.
            The default is 'exp'.
        identity_init : bool, optional
            Whether to initialize the flow as an identity flow.
            The default is False.
        '''
        super(ConditionalAffineFlow, self).__init__()
        self.in_features = in_features
        self.identity_init = identity_init
        
        # Create affine parameters
        self.affine_a = ResLinear(cond_features, in_features)
        self.affine_b = ResLinear(cond_features, in_features)
        
        if nonlinearity_a == 'exp':
            self.nonlinearity_a = 'torch.exp'
        elif nonlinearity_a == 'softpuls':
            self.nonlinearity_a = 'F.softplus'
        else:
            raise ValueError("Unsupported nonlinearity_a {}".format(nonlinearity_a))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.affine_a.reset_parameters()
        self.affine_b.reset_parameters()
        if self.identity_init:
            # let b = 0
            self.affine_b.l2.weight.data.uniform_(-0.001, 0.001)
            self.affine_b.l3.weight.data.uniform_(-0.001, 0.001)
            self.affine_b.l2.bias.data.fill_(0.0)
            self.affine_b.l3.bias.data.fill_(0.0)
            # let a = 1
            self.affine_a.l2.weight.data.uniform_(-0.001, 0.001)
            self.affine_a.l3.weight.data.uniform_(-0.001, 0.001)
            if self.nonlinearity_a == 'torch.exp':
                inv = 0.0
            elif self.nonlinearity_a == 'F.softplus':
                inv = math.log(math.exp(1) - 1) * 0.5
            self.affine_a.l2.bias.data.fill_(inv)
            self.affine_a.l3.bias.data.fill_(inv)
            
    def forward(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Transform x into z which follows a standard norm distribution.

        Parameters
        ----------
        x : Tensor [B x * x D]
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x * x D]
        '''
        self.a = eval(self.nonlinearity_a)(self.affine_a(cond_var))
        b = self.affine_b(cond_var)
        z = x * self.a + b
        
        return z
        
    def log_prob(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Compute the log probabilities at x.

        Parameters
        ----------
        x : Tensor [B x * x D]
            Point at which pdf is to be evaluated.
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.

        Returns
        -------
        Tensor [B x *]
            Log probabilities at x.
        '''
        z = self.forward(x, cond_var)
        
        log_prob_z = -0.5 * math.log(2 * math.pi) -0.5 * z**2
        log_prob_x = log_prob_z.sum(-1) + self.log_det()

        return log_prob_x
    
    def log_det(self) -> Tensor:
        '''
        Compute the log determinant of the Jacobian.
        '''
        return torch.log(self.a).sum(-1)
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError
        
class ResLinear(nn.Module):
    '''
    Linear layer with residual connection.
    z = linear2(act(linear1(x))) + linear3(x)
    '''
    def __init__(self, in_features: int, out_features: int,
                 nonlinearity: str='relu', act_param: float=0.1, 
                 has_res_layer: bool=True):
        super(ResLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_res_layer = has_res_layer
        
        self.l1 = nn.Linear(in_features, out_features)
        self.l2 = nn.Linear(out_features, out_features)
        if self.has_res_layer:
            self.l3 = nn.Linear(in_features, out_features)
        self.activation = get_activation(nonlinearity, act_param)
        
        self.reset_parameters()

    def reset_parameters(self, mask: Tensor=None):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        if self.has_res_layer:
            self.l3.reset_parameters()
        
    def forward(self, input: Tensor):
        z = self.l2(self.activation(self.l1(input)))
        if self.has_res_layer:
            z = z + self.l3(input)
        else:
            z = z + input
        
        return z