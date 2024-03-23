import copy
import numpy as np
from Client import Client
from torch.utils.data import DataLoader, Dataset
# from utils import fedavg
import os.path as osp
import torch
import torch.nn as nn
from utils import *

class Server(object):
    def __init__(self, loggers, devices,dict_users_data, images_data, text_data, labels_data,args,model_trainer):
        self.args = args
        self.device = devices
        
        self.client_group = dict_users_data
        self.img_data = images_data
        self.txt_data = text_data
        self.label_data = labels_data

        self.client_list = []
        self.best = 0.0
        self.pbest = 0.0
        self.logger = loggers

        self.model_trainer = model_trainer

        self._setup_clients(model_trainer)

    def _setup_clients(self,model_trainer):
        
        for i in range(self.args.num_users):
            local_data = self._assign_data_for_each_client(client_id=i)
            client = Client(client_id=i,
                            args=self.args,
                            data=local_data,
                            device=self.device,
                            model_trainer = model_trainer
                            )
            
            self.client_list.append(client)
    
    def _assign_data_for_each_client(self, client_id):
        local_idxs = np.array(self.client_group[client_id])

        local_img = self.img_data[local_idxs]
        local_txt = self.txt_data[local_idxs]
        local_label = self.label_data[local_idxs]
        local_data = [local_img,local_txt,local_label]
        return local_data
     
    def _client_sampling(self,round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes
        
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def train(self):
        w_global_img,w_global_txt,w_global_pcme,w_global_lab= self.model_trainer.get_model_params()
        w_locals_lab = []
        for i in range(self.args.num_users):
            w_locals_lab.append(w_global_lab)

        for round_idx in range(self.args.rounds):
            self.logger.info("################################Communication round : %d################################"%(round_idx+1))
            w_locals_img = []
            w_locals_txt = []
            w_locals_pcme = []
            # client_num_per_round = np.random.randint(low=2,high=16)
            client_num_per_round = 15
            client_indexes = self._client_sampling(round_idx, self.args.num_users,
                                                   client_num_per_round)
            p_all_i2t_map = 0.0
            p_all_t2i_map = 0.0
            for client_id in client_indexes:
                w_img,w_txt,w_pc,w_lab,pMAP_I2T, pMAP_T2I= self.client_list[client_id].train(round_idx,copy.deepcopy(w_global_img),copy.deepcopy(w_global_txt),copy.deepcopy(w_global_pcme),copy.deepcopy(w_locals_lab[client_id]))
                p_all_i2t_map += pMAP_I2T 
                p_all_t2i_map += pMAP_T2I

                w_locals_img.append((self.client_list[client_id].get_sample_number(), copy.deepcopy(w_img)))
                w_locals_txt.append((self.client_list[client_id].get_sample_number(), copy.deepcopy(w_txt)))
                w_locals_pcme.append((self.client_list[client_id].get_sample_number(), copy.deepcopy(w_pc)))
                w_locals_lab[client_id] = copy.deepcopy(w_lab)
            
            self.logger.info('----------------local test----------------')
            p_MAP_I2T_ave = p_all_i2t_map / self.args.num_users
            P_MAP_T2I_ave = p_all_t2i_map / self.args.num_users
            self.logger.info('average pMAP_I2T_ave:'+str(p_MAP_I2T_ave))
            self.logger.info('average pMAP_T2I_ave:'+str(P_MAP_T2I_ave))
            if p_MAP_I2T_ave + P_MAP_T2I_ave > self.pbest:
                self.pbest = p_MAP_I2T_ave + P_MAP_T2I_ave

            w_global_img = self._aggregate(w_locals_img)
            w_global_txt = self._aggregate(w_locals_txt)
            w_global_pcme = self._aggregate(w_locals_pcme)
            
            self.model_trainer.set_model_params(w_global_img,w_global_txt,w_global_pcme,None)

            self.logger.info('----------------Global test----------------')
            self.local_validate_on_all_clients(round_idx)


    def local_validate_on_all_clients(self, round_idx):
        MAP_I2T_all, MAP_T2I_all = 0.0,0.0
   
        for idx, client in enumerate(self.client_list):
            MAP_I2T, MAP_T2I = client.local_validate(epoch = round_idx)
            MAP_I2T_all += MAP_I2T
            MAP_T2I_all += MAP_T2I
        MAP_I2T_ave = MAP_I2T_all / self.args.num_users
        MAP_T2I_ave = MAP_T2I_all / self.args.num_users
        self.logger.info('average MAP_I2T_ave:'+str(MAP_I2T_ave))
        self.logger.info('average MAP_T2I_ave:'+str(MAP_T2I_ave))
        if MAP_I2T_ave + MAP_T2I_ave > self.best:
            self.best = MAP_I2T_ave + MAP_T2I_ave
        
    def save_checkpoints(self):
        file_name = '%s_%d_bit_latest.pth' % (str(self.args.dataset) , self.args.bit)
        model_img_params,model_txt_params,_,_ = self.model_trainer.get_model_params()

        ckp_path = osp.join(self.args.save_model_path, file_name)
        obj = {
            'ImgNet': model_img_params,
            'TxtNet': model_txt_params,
        }
        torch.save(obj, ckp_path)
        print('**********Save the trained model successfully.**********')