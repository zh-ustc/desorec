import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data


# use 0 to padding

def neg_sample(seq, labels, num_item,sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs


class TrainDataset(Data.Dataset):
    def __init__(self, args):
        self.cold = args.cold
        self.enable_sample =args.enable_sample
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.max_len = args.max_len


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index):
      
        


        seq = self.data[index, -self.max_len - 3:-3].tolist()
        labels = [self.data[index,-3].tolist()]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
           
        return torch.LongTensor(seq), index,torch.LongTensor(labels)




class EvalDataset(Data.Dataset):
    def __init__(self,  args, mode='test'):
        
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.max_len = args.max_len
        self.mode = mode


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        if self.mode == 'test': t = 1
        elif self.mode == 'valid': t = 2
        seq = self.data[index, :-t]
        pos = self.data[index, -t]
        
        seq = list(seq)
       
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        answers = [pos]
        return torch.LongTensor(seq), torch.LongTensor(answers)
        
