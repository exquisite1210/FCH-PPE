import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from layers.MLP import MLP
from layers.ConditionalNAF_DSF import ConditionalNAF_DSF as CondNAF
from layers.ConditionalAffineFlow import ConditionalAffineFlow as CondAF
from layers.ConditionalTransformedDistribution import ConditionalTransformedDistribution as CondTDist
from utils import init_random_seed

import torch.distributions as dist

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def reset_parameters(self):
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        
    def forward(self,feature):
        mu = self.fc_mu(feature)
        logvar = self.fc_logvar(feature)
        return mu,logvar
    
class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.decoder = MLP(latent_dim, latent_dim, [512], False,
                                        nonlinearity="relu")
    def reset_parameters(self):
        self.decoder.reset_parameters()
    def forward(self,z):
        recons = self.decoder(z)
        return recons

class OursModel(nn.Module):
    def __init__(self,args,drop_ratio = 0.1):
        super(OursModel, self).__init__()


        self.dimZ = args.bit#参数
        self.alpha = 0
        self.device = args.device
        self.args = args
        
        self.encoder = Encoder(latent_dim = args.bit)
        self.decoder = Decoder(latent_dim = args.bit)

        self.label_encodings = nn.Parameter(torch.eye(self.args.num_classes).unsqueeze(0),
                            requires_grad=False)
        
        base_dist = torch.distributions.normal.Normal(torch.zeros(self.dimZ).to(self.device),
                                                torch.ones(self.dimZ).to(self.device))
        self.pos_prototypes = self._create_normalizing_flows(base_dist)
        self.neg_prototypes = self._create_normalizing_flows(base_dist)

        self._reset_parameters()

    def _reset_parameters(self):
        init_random_seed(self.args.seed)
        self.pos_prototypes.reset_parameters()
        self.neg_prototypes.reset_parameters()

  
    def reparametrize(self, mu, logvar):
        sub_std = torch.exp(0.5 * logvar)
        z = dist.Normal(mu, sub_std).rsample()
        return z
    
    def _create_normalizing_flows(self, base_dist):
        flow_trans = []
        flow_trans.append(CondNAF(self.dimZ, self.args.num_classes, [256]))
        flow_trans.append(CondAF(self.dimZ, self.args.num_classes,identity_init=True))
        
        return CondTDist(base_dist, flow_trans)
    
    def _log_density_proto(self, x):
        x_temp = x.unsqueeze(1)

        self.label_encodings = self.label_encodings.to(self.device)
        pos_log_density = self.pos_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]

        neg_log_density = self.neg_prototypes.log_prob(x_temp, self.label_encodings) # [B x Q]

        return torch.stack([neg_log_density, pos_log_density], dim=1)
          
    def forward(self,image):
      
        img_mu, img_logvar = self.encoder(image)

        z = self.reparametrize(img_mu, img_logvar)

        recons  = self.decoder(z)


        log_ins_class_probs_z = self._log_density_proto(z)

        dists_x_z = log_ins_class_probs_z

        return dists_x_z,img_mu, img_logvar,recons
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)

        
class LabelModule(nn.Module):
    def __init__(self, num_classes, bit,drop_ratio = 0.1):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(LabelModule, self).__init__()
        self.module_name = "LabelModule"
        # full-conv layers
        self.label_encoder = MLP(num_classes, 256, [512], False, 
                                 drop_ratio, "relu")
        self.label_fc_mu = nn.Linear(256, bit)
        self.label_fc_logvar = nn.Linear(256, bit)
        self.label_decoder = MLP(bit, 512, [256], False,
                                 nonlinearity="relu")
        self.label_classifier = nn.Linear(512, num_classes)

        self.apply(weights_init)

    def _label_encode(self, target):
        '''
        Encode the input by passing through the encoder network and return the
        latent codes.

        Parameters
        ----------
        target : Tensor
            Input tensor to encode.

        Returns
        -------
        Tuple(Tensor, Tensor)
            Mean and log variance parameters of the latent Gaussian distribution.
        '''
        result = self.label_encoder(target)
        mu = self.label_fc_mu(result)
        logvar = self.label_fc_logvar(result)
        
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        sub_std = torch.exp(0.5 * logvar)
        z = dist.Normal(mu, sub_std).rsample()
        return z
    
    def _label_decode(self, z):
        '''
        Decode the latent codes by passing through the decoder network.

        Parameters
        ----------
        z : Tensor [B x D]
            Latent codes to decode.

        Returns
        -------
        Tensor
            Reconstruction.
        '''
        return self.label_classifier(self.label_decoder(z))
    
    def forward(self, target):
        y_mu, y_logvar = self._label_encode(target)
        z_y = self.reparametrize(y_mu, y_logvar)
        preds_y = self._label_decode(z_y)

        return y_mu, y_logvar,preds_y