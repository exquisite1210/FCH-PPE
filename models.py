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



class ImgModule(nn.Module):
    def __init__(self, image_dim,bit):
        super(ImgModule, self).__init__()
        self.module_name = "image_model" 
        # fc8
        self.classifier = nn.Linear(in_features=image_dim, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01

    def forward(self, x):
        x = x.squeeze()
        x = self.classifier(x)

        return x
    

class TxtModule(nn.Module):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"
        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()

        return x  


LAYER1_NODE = 8192


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.fc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*2, latent_dim)

    def forward(self,feature):
        mu = self.fc_mu(feature)
        logvar = self.fc_logvar(feature)
        return mu,logvar
    
    def reset_parameters(self):
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()

class Decoder(nn.Module):
    def __init__(self,latent_dim,modal_dim):
        super(Decoder, self).__init__()
        self.decoder = MLP(latent_dim, modal_dim, [512], False,
                                        nonlinearity="relu")
    def forward(self,z):
        recons = self.decoder(z)
        return recons
    def reset_parameters(self):
        self.decoder.reset_parameters()


class PCMH(nn.Module):
    def __init__(self,args,img_dim,txt_dim,label_dim):
        super(PCMH, self).__init__()
        self.dimZ = args.bit
        self.alpha = 0
        self.device = args.device
        self.args = args
        self.drop_ratio = 0.1

        self.label_encoder = MLP(label_dim,256,[512],False,self.drop_ratio,'relu')
        self.label_fc_mu = nn.Linear(256, self.args.bit)
        self.label_fc_logvar = nn.Linear(256, self.args.bit)
        self.label_decoder = MLP(self.args.bit, 512, [256], False,
                                 nonlinearity="relu")
        self.label_classifier = nn.Linear(512, label_dim)

        base_dist = torch.distributions.normal.Normal(torch.zeros(self.dimZ).to(self.device),
                                                torch.ones(self.dimZ).to(self.device))
        
        self.pos_prototypes = self._create_normalizing_flows(base_dist)
        self.neg_prototypes = self._create_normalizing_flows(base_dist)

        self.fc_mu = nn.Linear(self.args.bit, self.args.bit)
        self.fc_logvar = nn.Linear(self.args.bit, self.args.bit)


        self.decoder = nn.Sequential(
            nn.Linear(self.args.bit, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, self.args.bit),
            nn.BatchNorm1d(self.args.bit)
            )


        self.to(self.device)

        self._reset_parameters()
        
    def _reset_parameters(self):
        init_random_seed(self.args.seed)
        self.pos_prototypes.reset_parameters()
        self.neg_prototypes.reset_parameters()
 
    def _reparameterize(self, mu, logvar):
        '''
        Reparameterize trick to sample from N(mu, var).

        Parameters
        ----------
        mu : Tensor [B x D]
            Mean of the latent Gaussian.
        logvar : Tensor [B x D]
            Log variance of the latent Gaussian.

        Returns
        -------
        Tensor [B x D]
            Sampled latent codes.
        '''

        sub_std = torch.exp(0.5 * logvar)
        z = dist.Normal(mu, sub_std).rsample()
        return z

    def _create_normalizing_flows(self, base_dist):
        flow_trans = []
        flow_trans.append(CondNAF(self.dimZ, self.args.num_classes, [256]))
        flow_trans.append(CondAF(self.dimZ, self.args.num_classes,identity_init=True))
        
        return CondTDist(base_dist, flow_trans)

    def _log_density_proto(self, x):
        '''
        Compute instance's log probability on positive/negative prototypes of each class label.

        Parameters
        ----------
        x : Tensor [B x D]
            Point at which density is to be evaluated.

        Returns
        -------
        Tensor [B x 2 x Q]
            log density at x.
        '''
        x_temp = x.unsqueeze(1)
        label_encodings = nn.Parameter(torch.eye(self.args.num_classes).unsqueeze(0),
                                    requires_grad=False)
        label_encodings = label_encodings.to(self.device)
        pos_log_density = self.pos_prototypes.log_prob(x_temp, label_encodings) # [B x Q]

        neg_log_density = self.neg_prototypes.log_prob(x_temp, label_encodings) # [B x Q]

        return torch.stack([neg_log_density, pos_log_density], dim=1)
    
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
    
    def forward(self,img_code,label,modal=None):
        # print(img_code)

        y_mu, y_logvar = self._label_encode(label)
        z_y = self._reparameterize(y_mu, y_logvar)
        preds_y = self._label_decode(z_y)
        
        img_mu = self.fc_mu(img_code)
        img_logvar = self.fc_logvar(img_code)



        img_z = self._reparameterize(img_mu, img_logvar)

        recons = self.decoder(img_z)


        log_ins_class_probs_zimg = self._log_density_proto(img_z)


        dists_x_zimg =  log_ins_class_probs_zimg


        return dists_x_zimg, img_mu, img_logvar ,preds_y ,y_mu, y_logvar,recons
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_requires(self,model):
        for p in model.parameters():
            p.requires_grad = True






