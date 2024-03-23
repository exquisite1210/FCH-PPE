#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import copy
import numpy as np
import h5py
import random
import scipy.io as scio

def load_pretrain_model(path):
    return scio.loadmat(path)

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

def load_data(path):
    file = h5py.File(path)
    images = file['IAll'][:]
    labels = file['LAll'][:]
    tags = file['TAll'][:]
    images = images.transpose(3,2,0,1)
    labels = labels.transpose(1,0)
    tags = tags.transpose(1,0)

    file.close()
    
    return images, tags, labels

def one_hot(x,class_count):
    return torch.eye(class_count)[x,:]

def split_data(images, tags, labels, database_size_input, query_size_input,train_size_input):
    query_size = query_size_input
    training_size = train_size_input
    database_size = database_size_input

    X = {}
    X['query'] = images[0: query_size]
    X['train'] = images[query_size: training_size + query_size]
    X['retrieval'] = images[query_size: query_size + database_size]

    Y = {}
    Y['query'] = tags[0: query_size]
    Y['train'] = tags[query_size: training_size + query_size]
    Y['retrieval'] = tags[query_size: query_size + database_size]

    L = {}
    L['query'] = labels[0: query_size]
    L['train'] = labels[query_size: training_size + query_size]
    L['retrieval'] = labels[query_size: query_size + database_size]

    return X, Y, L
    


def fedavg(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num,averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params

def init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def CalcMap(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

def generate_image_code(args,img_model, X, bit):
    batch_size = args.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if args.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if args.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    image = image.cpu()
    return B


def generate_text_code(args,txt_model, Y, bit):
    batch_size = args.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if args.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if args.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    text = text.cpu()
    return B