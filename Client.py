import os.path as osp
import torch.nn as nn
import torch.nn.functional as FF
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import copy
from utils import *
from layers.ConditionalNAF_DSF import ConditionalNAF_DSF as CondNAF
from layers.ConditionalAffineFlow import ConditionalAffineFlow as CondAF
from layers.ConditionalTransformedDistribution import ConditionalTransformedDistribution as CondTDist
from torch.distributions import MultivariateNormal
from layers.MLP import MLP
import math
class Client(object):
    def __init__(self,client_id,args,data,device,model_trainer):
        self.id = client_id
        self.args = args
        self.device = device

        self.data = data
        self.model_trainer = model_trainer

    
        self.best = 0.0
        self.local_sample_number = 0

    def get_sample_number(self):
        return self.local_sample_number
    
    def get_split_train(self,data):
        X, Y, L = self.split_data(data)
        train_L = torch.from_numpy(L['train'])
        train_x = torch.from_numpy(X['train'])
        train_y = torch.from_numpy(Y['train'])
        num_train = train_x.shape[0]
        return train_L,train_x,train_y,num_train
    
    def get_split_test(self,data):
        X, Y, L = self.split_data(data)
        query_L = torch.from_numpy(L['query'])
        query_x = torch.from_numpy(X['query'])
        query_y = torch.from_numpy(Y['query'])

        retrieval_L = torch.from_numpy(L['retrieval'])
        retrieval_x = torch.from_numpy(X['retrieval'])
        retrieval_y = torch.from_numpy(Y['retrieval'])
        return query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y
    

    
    def train(self, round_idx,w_global_img,w_global_txt,w_global_pcme,w_lab):

        train_L,train_x,train_y,local_sample_number = self.get_split_train(self.data)
        self.local_sample_number = local_sample_number
        local_training_data = [train_L,train_x,train_y]

        F_buffer = torch.randn(self.local_sample_number, self.args.bit)
        G_buffer = torch.randn(self.local_sample_number, self.args.bit)

        Sim = self.calc_neighbor(train_L, train_L)
        B = torch.sign(F_buffer + G_buffer)
        ones = torch.ones(self.args.batch_size, 1)
        ones_ = torch.ones(self.local_sample_number - self.args.batch_size, 1)

        other_params = [local_sample_number,F_buffer,G_buffer,Sim,B,ones,ones_]

        self.model_trainer.set_model_params(w_global_img,w_global_txt,w_global_pcme,w_lab)
        self.model_trainer.train(round_idx,local_training_data, self.device, self.args,other_params)
        weights_img,weights_txt,weights_pcme,weights_lab = self.model_trainer.get_model_params()

        pMAP_I2T, pMAP_T2I = self.per_validate(round_idx)

        return weights_img,weights_txt,weights_pcme,weights_lab,pMAP_I2T, pMAP_T2I,

    def per_validate(self,epoch):
        query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y = self.get_split_test(self.data)

        MAP_I2T, MAP_T2I  = self.model_trainer.test(query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y, self.device, self.args)

        return MAP_I2T, MAP_T2I

    def local_validate(self,epoch):
        query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y = self.get_split_test(self.data)

        MAP_I2T, MAP_T2I  = self.model_trainer.test(query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y, self.device, self.args)

        return MAP_I2T, MAP_T2I

    def split_data(self,data):
        images,tags,labels = data
        database_size = labels.shape[0]
        query_size = math.ceil(database_size*0.0999250562078)
        training_size = math.ceil(database_size*0.4996252810392)
        database_size2 = math.ceil(database_size*0.9000749437921)
        X = {}
        X['query'] = images[0: query_size]
        X['train'] = images[query_size: training_size + query_size]
        X['retrieval'] = images[query_size: query_size + database_size2]

        Y = {}
        Y['query'] = tags[0: query_size]
        Y['train'] = tags[query_size: training_size + query_size]
        Y['retrieval'] = tags[query_size: query_size + database_size2]

        L = {}
        L['query'] = labels[0: query_size]
        L['train'] = labels[query_size: training_size + query_size]
        L['retrieval'] = labels[query_size: query_size + database_size2]

        return X, Y, L

    def calc_neighbor(self,label1, label2):
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
        return Sim
    
    def calc_loss(self,B, F, G, Sim, gamma, eta):
        F = FF.normalize(F)
        G = FF.normalize(G)
        theta = torch.matmul(F, G.transpose(0, 1)) / 2
        term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
        term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
        term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
        loss = term1 + gamma * term2 + eta * term3
        return loss
    