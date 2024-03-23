import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from type_def import *
from layers.Conditioner import Conditioner

class ConditionalNAF_DSF(nn.Module):
    '''
    Neural Autoregressive Flow-DSF [1] for modeling conditional probability p(x|y).
    
    [1] Huang, C.; Krueger, D.; Lacoste, A.; and Courville, A. C. 2018. Neural autoregressive flows.
    In Proceedings of the 35th International Conference on Machine Learning, 2083–2092. Stockholm, Sweden.
    '''
    def __init__(self, in_features: int, cond_features: int, hidden_features: list=[],
                 input_order: str='same', mode: str='sequential',
                 nonlinearity: str='elu', act_param: float=1.0,
                 ds_dim: int=16, num_ds_layer: int=1, identity_init: bool=False,
                 eps: float=1e-6):
        '''
        Parameters
        ----------
        in_features : int
            Dimension of the input.
        cond_features : int
            Dimension of the conditional variable y.
        hidden_features : list, optional
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
        ds_dim : int, optional
            Number of hidden units for the sigmoidal neural network.
            The default is 16.
        num_ds_layer : int, optional
            Number of the sigmoidal neural network.
            The default is 1.
        identity_init : bool, optional
            Whether to initialize the flow as an identity flow.
            The default is False.
        eps : float, optional
            A small constant for numerical stabilization.
            The default is 1e-6.
        '''
        super(ConditionalNAF_DSF, self).__init__()
        self.in_features = in_features
        self.ds_dim = ds_dim
        self.num_ds_layer = num_ds_layer
        self.identity_init = identity_init
        self.eps = eps
        
        # Create conditioner
        out_dim1 = 3 * (hidden_features[-1] // in_features) * num_ds_layer
        out_dim2 = 3 * ds_dim * num_ds_layer
        self.conditioner = Conditioner(in_features, cond_features, out_dim1,
                                       hidden_features, input_order, mode,
                                       nonlinearity, act_param)
        self.out_to_dsparams = nn.Linear(out_dim1, out_dim2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conditioner.reset_parameters()
        if self.identity_init:
           self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
           self.out_to_dsparams.bias.data.fill_(0.0)
           # bias for a
           inv = math.log(math.exp(1) - 1)
           nparams = 3 * self.ds_dim
           for i in range(self.num_ds_layer):
               start = i * nparams
               self.out_to_dsparams.bias.data[start:start+self.ds_dim].fill_(inv)
        else:
            self.out_to_dsparams.reset_parameters()
            
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
        params = self.conditioner(x, cond_var) # [B x * x D x out_dim1], note that the params is output with no nonlinearity
        params = self.out_to_dsparams(params) # [B x * x D x out_dim2], (a, b, w)
        
        start = 0
        self.logdet = 0.0
        for i in range(self.num_ds_layer):
            a = F.softplus(params[..., start:start+self.ds_dim]) # [B x * x D x ds_dim]
            start += self.ds_dim
            b = params[..., start:start+self.ds_dim] # [B x * x D x ds_dim]
            start += self.ds_dim
            w_ = params[..., start:start+self.ds_dim] # [B x * x D x ds_dim]
            w = torch.softmax(w_, dim=-1)
            start += self.ds_dim
            
            pre_sigm = a * x.unsqueeze(-1) + b # [B x * x D x ds_dim]
            x_pre = torch.sum(w * torch.sigmoid(pre_sigm), dim=-1) # [B x * x D]
            x_pre_clipped = x_pre * (1-self.eps) + self.eps * 0.5
            x = torch.log(x_pre_clipped / (1 - x_pre_clipped)) # [B x * x D]
            
            logdet = F.log_softmax(w_, dim=-1) + F.logsigmoid(pre_sigm) + \
                      F.logsigmoid(-pre_sigm) + torch.log(a)
            logdet = torch.logsumexp(logdet, dim=-1) + math.log(1-self.eps) - \
                      torch.log(x_pre_clipped) - torch.log(1-x_pre_clipped)
            self.logdet = self.logdet + logdet.sum(-1) # [B x *]
        
        return x
        
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
        return self.logdet
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError