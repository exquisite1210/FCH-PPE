import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as FF
from torch.nn import BCELoss, NLLLoss, MSELoss
from model_trainer import ModelTrainer
from utils import *



class MyModelTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model1.cpu().state_dict(),self.model2.cpu().state_dict(),self.model3.cpu().state_dict(),self.model4.cpu().state_dict()
    
    def get_config_optim_img(self):
        return [{'params': self.model1.parameters()},
                {'params': self.model3.parameters()},
                {'params': self.model4.parameters()}
            ]
    
    def get_config_optim_txt(self):
        return [{'params': self.model2.parameters()},
                 {'params': self.model3.parameters()},
                 {'params': self.model4.parameters()}
                 ]
    
    def set_model_params(self, model_parameters1,model_parameters2, model_parameters3,model_parameters4):

        self.model1.load_state_dict(model_parameters1)
        self.model2.load_state_dict(model_parameters2)
        self.model3.load_state_dict(model_parameters3)
        if model_parameters4 != None:
            self.model4.load_state_dict(model_parameters4)
        
    def calc_loss(self,B, F, G, Sim, gamma, eta):
        F = FF.normalize(F)
        G = FF.normalize(G)
    
        theta = torch.matmul(F, G.transpose(0, 1)) / 2
        term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)

        term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
        term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
        loss = term1 + gamma * term2 + eta * term3
        return loss

    def calc_neighbor(self,label1, label2):

        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)

        return Sim
    
    
    def train(self, round_idx,train_data, device, args,other_params):
        local_sample_number,F_buffer,G_buffer,Sim,BB,ones,ones_ = other_params
        train_L,train_x,train_y = train_data

        imgmodel= self.model1
        txtmodel = self.model2
        MMP = self.model3
        labelmodel = self.model4
        self.src_model.load_state_dict(self.model3.cpu().state_dict())

        self.src_model.to(device)


        imgmodel.to(device)
        txtmodel.to(device)
        MMP.to(device)
        labelmodel.to(device)

        F_buffer = F_buffer.to(device)
        G_buffer = G_buffer.to(device)
        BB = BB.to(device)
        ones = ones.to(device)
        ones_ = ones_.to(device)
        Sim = Sim.to(device)


        MMP.train()
        txtmodel.train()
        imgmodel.train()
        labelmodel.train()

        learning_rate = np.linspace(args.learning_rate, np.power(10, -6.), args.rounds*args.train_ep + 1)
        
        args_I = torch.optim.SGD(self.get_config_optim_img(), lr=args.learning_rate)
        args_T = torch.optim.SGD(self.get_config_optim_txt(), lr=args.learning_rate)

        lr_p = learning_rate[round_idx*args.train_ep]

        for param in args_I.param_groups:
            param['lr'] = lr_p
        for param in args_T.param_groups:
            param['lr'] = lr_p
        

        for epoch in range(args.train_ep):
            for epoch_i in range(local_sample_number // args.batch_size):
                args_I.zero_grad()
                index = np.random.permutation(local_sample_number)
                ind = index[0: args.batch_size]
                unupdated_ind = np.setdiff1d(range(local_sample_number), ind)
                sample_L = Variable(train_L[ind, :])
                text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
                text = Variable(text)
                text = text.to(device)
                image = Variable(train_x[ind].type(torch.float))
                image = image.to(device)
                S = self.calc_neighbor(sample_L, train_L)
                S = S.to(device)
                cur_f_o = imgmodel(image)
                F_buffer[ind, :] = cur_f_o.data
                F = Variable(F_buffer)
                G = Variable(G_buffer)
                cur_f = FF.normalize(cur_f_o)
                F = FF.normalize(F)
                G = FF.normalize(G)
                sample_L = sample_L.type(torch.cuda.FloatTensor).to(device)
                theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
                logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))/(self.args.batch_size* local_sample_number)
                quantization_x = torch.sum(torch.pow(BB[ind, :] - cur_f, 2))/(self.args.batch_size* local_sample_number)
                balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))/(self.args.batch_size* local_sample_number)
                loss_cmr = logloss_x + self.args.gamma * quantization_x + self.args.eta * balance_x

 
                dists_x_i, mu_i, logvar_i,recons_i= MMP(cur_f)
                loss_jvae_img = FF.mse_loss(recons_i, cur_f) 
                cons_loss_i = FF.cross_entropy(dists_x_i, sample_L.long())* sample_L.size(1)

                loss_proximal_i = 0
                for pm, ps in zip(MMP.parameters(), self.src_model.parameters()):
                    if isinstance(pm, torch.Tensor) and pm.dtype == torch.bool:
                        continue
                    if isinstance(ps, torch.Tensor) and ps.dtype == torch.bool:
                        continue
                    loss_proximal_i += torch.sum(torch.pow(pm - ps, 2))
                
                y_mu, y_logvar,preds_y = labelmodel(sample_L)
                Recons_loss_i = FF.mse_loss(preds_y.sigmoid(), sample_L, reduction='sum')/self.args.batch_size
                eps = 1e-5
                kl_div = torch.mean(0.5*torch.sum(y_logvar-logvar_i-1+torch.exp(logvar_i-y_logvar)+(y_mu-mu_i)**2/(torch.exp(y_logvar)+eps), dim=1))
                Reg_loss_i = kl_div + FF.multilabel_soft_margin_loss(preds_y,sample_L)* sample_L.size(1)

                loss_x = loss_cmr  + args.deta_2*Recons_loss_i + args.deta_2*Reg_loss_i + args.deta_1*cons_loss_i+ args.deta_3*loss_proximal_i + args.deta_4*loss_jvae_img
                loss_x.backward()
                args_I.step()

            for epoch_t in range(local_sample_number // args.batch_size):
                args_T.zero_grad()
                index = np.random.permutation(local_sample_number)
                ind = index[0: args.batch_size]
                unupdated_ind = np.setdiff1d(range(local_sample_number), ind)
                sample_L = Variable(train_L[ind, :])
                text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
                text = Variable(text)
                text = text.to(device)
                image = Variable(train_x[ind].type(torch.float))
                image = image.to(device)
                S = self.calc_neighbor(sample_L, train_L)
                S = S.to(device)
                cur_g_o = txtmodel(text)
                G_buffer[ind, :] = cur_g_o.data
                F = Variable(F_buffer)
                G = Variable(G_buffer)
                cur_g = FF.normalize(cur_g_o)
                F = FF.normalize(F)
                G = FF.normalize(G)
                sample_L = sample_L.type(torch.cuda.FloatTensor).to(device)
                theta_y = 1.0 / 2 * torch.matmul(cur_g,F.t())
                logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))/(self.args.batch_size* local_sample_number)
                quantization_y = torch.sum(torch.pow(BB[ind, :] - cur_g, 2))/(self.args.batch_size* local_sample_number)
                balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))/(self.args.batch_size* local_sample_number)
                loss_cmr_t = logloss_y + self.args.gamma * quantization_y + self.args.eta * balance_y

                dists_x_t, mu_t, logvar_t ,recons_t = MMP(cur_g)

                loss_jvae_txt = FF.mse_loss(recons_t, cur_g)
                cons_loss_t = FF.cross_entropy(dists_x_t, sample_L.long()) * sample_L.size(1)
                loss_proximal_t = 0
                for pm, ps in zip(MMP.parameters(), self.src_model.parameters()):
                    if isinstance(pm, torch.Tensor) and pm.dtype == torch.bool:
                        continue
                    if isinstance(ps, torch.Tensor) and ps.dtype == torch.bool:
                        continue
                    loss_proximal_t += torch.sum(torch.pow(pm - ps, 2))

                y_mu, y_logvar,preds_y_t = labelmodel(sample_L)
                Recons_loss_t = FF.mse_loss(preds_y_t.sigmoid(), sample_L, reduction='sum')/self.args.batch_size
                eps = 1e-5
                kl_div = torch.mean(0.5*torch.sum(y_logvar-logvar_t-1+torch.exp(logvar_t-y_logvar)
                                            +(y_mu-mu_t)**2/(torch.exp(y_logvar)
                                            +eps), dim=1))
                Reg_loss_t = kl_div + FF.multilabel_soft_margin_loss(preds_y_t,sample_L) * sample_L.size(1)
                
                loss_y = loss_cmr_t   + args.deta_2*Recons_loss_t + args.deta_2*Reg_loss_t + args.deta_1*cons_loss_t+ args.deta_3*loss_proximal_t+ args.deta_4*loss_jvae_txt#  #
                loss_y.backward()
                args_T.step()

            lr = learning_rate[round_idx*args.train_ep+epoch]

            for param in args_I.param_groups:
                param['lr'] = lr
            for param in args_T.param_groups:
                param['lr'] = lr 
        
    def test(self,query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y, device, args):
        imgmodel= self.model1
        txtmodel = self.model2

        imgmodel.to(device)
        txtmodel.to(device)

        imgmodel.eval()
        txtmodel.eval()

        qBX = generate_image_code(args,imgmodel, query_x, args.bit)
        qBY = generate_text_code(args,txtmodel, query_y, args.bit)
        rBX = generate_image_code(args,imgmodel, retrieval_x, args.bit)
        rBY = generate_text_code(args,txtmodel, retrieval_y, args.bit)

        MAP_I2T   = CalcMap(qBX, rBY, query_L.float().cpu(), retrieval_L.float())
        MAP_T2I   = CalcMap(qBY, rBX, query_L.float().cpu(), retrieval_L.float())


        return MAP_I2T, MAP_T2I

