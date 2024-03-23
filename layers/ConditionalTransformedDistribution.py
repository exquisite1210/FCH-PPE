import torch.nn as nn

from type_def import *

class ConditionalTransformedDistribution(nn.Module):
    '''
    A distribution transformer implemented by stacking conditional normalizing flows.
    '''
    def __init__(self, base_dist, conditional_transforms: list):
        super(ConditionalTransformedDistribution, self).__init__()
        self.base_dist = base_dist
        self.conditional_transforms = nn.ModuleList(conditional_transforms)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self.base_dist, 'reset_parameters'):
            self.base_dist.reset_parameters()
        for t in self.conditional_transforms:
            t.reset_parameters()
            
    def forward(self, x: Tensor, cond_var: Tensor) -> Tensor:
        '''
        Transform x into z which follows a kwown base distribution.

        Parameters
        ----------
        x : Tensor [B x * x D]
        cond_var : Tensor [B x * x D_c]
            Conditional variable y.
            
        Returns
        -------
        Tensor [B x * x D]
        '''
        for t in self.conditional_transforms:
            x = t(x, cond_var)
        
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
        log_dets = 0.0
        for t in self.conditional_transforms:
            x = t(x, cond_var)
            log_dets = log_dets + t.log_det()
        
        log_prob_x = self.base_dist.log_prob(x).sum(-1) + log_dets
        
        return log_prob_x
    
    def log_det(self) -> Tensor:
        '''
        Compute the log determinant of the Jacobian.
        '''
        log_dets = 0.0
        for t in self.conditional_transforms:
            log_dets = log_dets + t.log_det()
            
        return log_dets
    
    def sample(self, size: list=[1]) -> Tensor:
        '''
        Sample from the learned distribution.
        '''
        raise NotImplementedError