from dataset import TrainDataset, EvalDataset
from process import Trainer
#from args import args
import pandas as pd
import torch
import torch.utils.data as Data
from model.GRU4rec import GRU4Rec
import os
import random
import pickle
import numpy as np
import torch.nn.functional as F
# torch.backends.cudnn.benchmark = True
# CUDA_LAUNCH_BLOCKING=1

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic= False
    torch.cuda.manual_seed_all(seed)

def main(args):
    seed_everything(0)
    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset = EvalDataset(args, mode='valid')
    val_loader = Data.DataLoader(val_dataset, batch_size=args.test_batch_size)

    test_dataset = EvalDataset(args, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    
    print('dataset initial ends')


    if args.model == 'GRU4Rec':
        model = GRU4Rec(args)

    trainer = Trainer(args, model, train_loader, test_loader,val_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    from args import args
    main(args)
