#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import random
import numpy as np
import h5py
import scipy.io as scio
from models import *
import logging
from Server import Server
from MyModelTrainer import MyModelTrainer
from DAH import OursModel,LabelModule
# from kmodes.kmodes import KModes

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--rounds', type=int, default=50, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=15, help="number of users: K")
    parser.add_argument('--train_ep', type=int, default=25, help="the number of local episodes: E")
    parser.add_argument('--device', default="cuda", type=str, help="cpu, cuda, or others")
    parser.add_argument('--gpu', default=3, type=int, help="index of gpu")
    parser.add_argument('--alpha', type=float, default=0.1, help='noniid')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size")
    parser.add_argument('--client_num_per_round', type=int, default=15,help='client_num_per_round: C')
    parser.add_argument('--num_classes', type=int, default=24, help="local batch size")
    #优化参数
    parser.add_argument('--learning_rate', type=float, default=10**(-1.5), metavar='N',help='learning_rate')#10**(-1.5)
    parser.add_argument('--save_model_path', default='./', help='path to save_model')
    parser.add_argument('--data_dir', type=str, default="./datasets/MSCOCO/MS-COCO.mat", help="name of dataset, default: './data/'")
    parser.add_argument('--dataset', type=str, default='MIR-FLICKR25K', help="name of dataset, e.g. MIR-FLICKR25K,MSCOCO,NUS-WIDE")
    parser.add_argument('--method', type=str, default='Ours', help="name of method, e.g. FedAvg,Local")
    #局部哈希参数
    parser.add_argument('--bit', type=int, default=16, help="hash code length")
    parser.add_argument('--gamma', type=float, default=1.0, help='DCMH')
    parser.add_argument('--eta', type=float, default=1.0, help='DCMH')
    parser.add_argument('--use_gpu', type=bool, default=True, help='gpu')

    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')#annealing_factor = 1.0
    parser.add_argument('--annealing_factor', type=float, default=1,
                            help='multipler for text reconstruction [default: 10]')
    
    parser.add_argument('--deta_1', type=float, default=0.3, help='hyper-parameter')
    parser.add_argument('--deta_2', type=float, default=0.7, help='hyper-parameter')
    parser.add_argument('--deta_3', type=float, default=0.001, help='hyper-parameter')
    parser.add_argument('--deta_4', type=float, default=0.001, help='hyper-parameter')
    return parser

def one_hot(x,class_count):
    return torch.eye(class_count)[x,:]

def dirichlet_split_noniid(train_labels, alpha, n_clients,n_classes):
    # print(len(train_labels))
    labelss = []
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            if train_labels[i][j] == 1:
                labelss.append(j)
    labels = np.array(labelss)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(labels==y).flatten() 
           for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

def load_data(dataset):
    if dataset == 'MIR-FLICKR25K':
        images = np.load("./imagesMIR-FLICKR25K.npy")
        tags = np.load("./tagsMIR-FLICKR25K.npy")
        labels = np.load("./labelsMIR-FLICKR25K.npy")
        client_group = np.load('./flickr_15_15_client_0.5_idcs.npy',allow_pickle=True)

    if dataset == 'MSCOCO':
        images = np.load("./imagesMSCOCO.npy")
        tags = np.load("./tagsMSCOCO.npy")
        labels = np.load("./labelsMSCOCO.npy")
        client_group = np.load('./MSCOCO_15_15_client_0.5_idcs.npy',allow_pickle=True)
    
    if dataset == 'NUSWIDE':
        images = np.load("./imagesNUSWIDE.npy")
        tags = np.load("./tagsNUSWIDE.npy")
        labels = np.load("./labelsNUSWIDE.npy")
        client_group = np.load("./NUSWIDE_15_15_client_0.5_idcs.npy",allow_pickle=True)

    return images, tags, labels,client_group

def load_pretrain_model(path):
    return scio.loadmat(path)

def custom_model_trainer(args,model_img,model_txt,model_pcme,model_label):
    return MyModelTrainer(model_img,model_txt,model_pcme,model_label,args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)


    images,tags,labels,client_group= load_data(args.dataset)
    print('...loading and splitting data finish')
    
    txt_feat_len = tags.shape[1]
    image_dim = 4096
    global_model_Img = ImgModule(image_dim,args.bit)
    global_model_Txt = TxtModule(txt_feat_len,args.bit)
    label_dim = 24
    gloabl_model_label = LabelModule(args.num_classes, args.bit)
    global_model_PCME = OursModel(args=args)
    My_trainer = custom_model_trainer(args, global_model_Img,global_model_Txt,global_model_PCME,gloabl_model_label)
    logger = logging.getLogger('experiment')
    file_handler = logging.FileHandler('./result/%s_%s_experiments_%.1f_%.1f_%d_%d.log'%(args.dataset,args.method,args.deta_1,args.deta_2,args.num_users,args.bit), mode='w')
    logger.addHandler(file_handler)
    logger.info('---------------------------%d---------------------------------'%(args.bit))
    logger.info('---------------------------%s---------------------------------'%(args))

    server = Server(
        loggers = logger,
        devices = args.device,
        dict_users_data = client_group,
        images_data = images,
        text_data = tags,
        labels_data = labels,
        args=args,
        model_trainer = My_trainer
        )
    
#     server.setup_clients()
    server.train()



