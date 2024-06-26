from pickletools import float8
import random
from functools import reduce
import math,pickle
import copy
import torch
from tqdm import tqdm
import time
from torch._C import device
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed
import scipy.stats
import threading
import json,os
def neg_sample(seq, labels, num_item,sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs




class CE:
    def __init__(self, model, args):
        self.model = model
        self.embedding_sharing = args.embedding_sharing
        self.enable_sample = args.enable_sample     
        self.num_item = args.num_item
        self.user_num = args.user_num
        self.sample_ratio = args.samples_ratio
        self.device = args.device
        self.net = args.model
        self.alpha = args.alpha
        self.args = args

        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
        #self.ce = self.ls
        self.path = args.load_path + 'class-{}-tau{}'.format(str(args.class_num),str(args.tau))

        # if args.graph + args.MA + args.use_pop + args.noise > 1:
        #     print('设置了多种方法，程序退出')
        #     exit()
        

        if args.loss == 'desorec':

            self.enhance = pickle.load(open(self.path+'soft','rb'))
            f1 = open(self.path+'index.json')
            f2 = open(self.path+'hash.json')
            I = json.load(f1)
            hash = json.load(f2)
            self.I = []
            self.H = []
            self.soft = []
            self.h = []
            m = 0
            for i in tqdm(self.enhance):
                i = torch.tensor(i,device='cuda')
                m += i.max(dim=-1)[0].sum().item()
            print('Z_m',m / self.user_num)
            for i in tqdm(range(self.user_num)):
                self.I.append(I[str(i)])
                cluster, id = I[str(i)]
                x = torch.tensor(self.enhance[cluster][id],requires_grad=False)
                self.soft.append(x)
                self.h.append(torch.LongTensor(hash[str(cluster)]))
                #x = x.cuda()
                #m += x.max(dim=-1)[0].mean().item()
                # self.soft = np.array(self.soft)
                # self.h = np.array(self.h)

            print('len',len(self.soft))

            for i in range(args.class_num):
                self.H.append(hash[str(i)])
     
        
       


    def index_to_label(self,labels):
        
        ll = torch.zeros(labels.shape[0], self.num_item + 1).to(self.device)
        
        hash = []
        L = []
        i0 = []
        idx = 0
        label_list = labels.cpu().tolist()
        for index in label_list:
            cluster, id = self.I[index]
            L.append(self.soft[index])
            hash.append(self.h[index])
            i0.extend([idx]*len(self.H[cluster]))
            idx += 1

        i0 = torch.LongTensor(i0).to(self.device)
        hash = torch.cat(hash,dim = 0).to(self.device)
        L = torch.cat(L,dim = 0).to(self.device)
        i = (i0,hash)
        ll.index_put_(i,L)
        return ll

    def compute_enhance(self, batch):
        args = self.args
        seqs,index,l = batch
        soft_label = self.index_to_label(index)
        outputs = self.model(seqs) # B * L * N
            
        one = torch.zeros(outputs.shape,device='cuda')
        one[torch.arange(len(l),device='cuda'),l.view(-1)]=1
        b = one.bool()
        p = outputs.softmax(dim = -1)
        p_b = p[b].view(-1) 
        p = p[~b].view(p.shape[0],-1)
        p = F.normalize(p,1,-1)

        labels_b = soft_label[b].view(-1)
        labels = soft_label[~b].view(soft_label.shape[0],-1)
        labels = F.normalize(labels,1,-1)
        labels_b = labels_b * args.ld1 + 1 - args.ld1

        loss1 = - torch.mean(labels_b*p_b.log() + (1-labels_b) * ((1-p_b+1e-12).log()))
        loss2 = F.kl_div((p +1e-12).log(), labels, reduction='batchmean')

        loss = args.ld2 * loss1 + (1-args.ld2) * loss2
        return loss

         

    def compute(self, batch):
        
        seqs, _ , labels = batch
        pred = self.model(seqs)  # B * L * N
       
        

        pred = pred.view(-1, pred.shape[-1])  # (B*L) * N
        labels = labels.view(-1)
        loss = self.ce(pred, labels)


        return loss

