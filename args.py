import argparse
import time
import os
import json
import pandas as pd
import torch
parser = argparse.ArgumentParser()



#1 train args  
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--lr_decay_rate', type=float, default=1)
parser.add_argument('--lr_decay_steps', type=int, default=1250)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--get_user', type=int, default=0)
parser.add_argument('--graph', type=int, default=0)
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--dataname', type=str, default='ce')
parser.add_argument('--early_stop', type=int, default=2)
parser.add_argument('--ld1', type=float, default=0.5)
parser.add_argument('--ld2', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=1) 
parser.add_argument('--class_num', type=int, default=2048)
parser.add_argument('--metric_ks', type=list, default=[10,20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='None') #
parser.add_argument('--load_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_len', type=int, default=20) #
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--train_batch_size', type=int, default=256) 
parser.add_argument('--test_batch_size', type=int, default=256)

#2 model args 
parser.add_argument('--model', type=str, default='sasrec', choices=['mlp', 'sasrec', 'deepfm', 'GRU4Rec']) # 模型选择，模型名见choices
parser.add_argument('--d_model', type=int, default=128) # 
parser.add_argument('--eval_per_steps', type=int, default=2) 
parser.add_argument('--enable_res_parameter', type=int, default=1)  
parser.add_argument('--enable_sample', type=int, default=0) #
parser.add_argument('--neg_samples', type=int, default=0) # 
parser.add_argument('--sample_strategy', type=str, default='raw_random') # 
parser.add_argument('--sampled_evaluation', type=int, default=0) #
parser.add_argument('--samples_ratio', type=float, default=0.1) # 
parser.add_argument('--output_style', type=str, default='avg') #
parser.add_argument('--embedding_sharing', type=int, default=0) # 




# gru
parser.add_argument('--dropout', type=float, default=0.1) 
parser.add_argument('--num_layers', type=int, default=4) 
parser.add_argument('--hidden_size', type=int, default=128) 


args = parser.parse_args()
# other args

DATA = pd.read_csv(args.data_path, header=None)
args.user_num = len(DATA)
args.eval_per_steps = args.user_num // args.train_batch_size
num_item = DATA.max().max()
del DATA
args.num_item = int(num_item)

if args.save_path == 'None' and args.get_user == 0:
    loss_str = args.loss_type
    path_str = 'Model-' + args.model +'_le_share-' + str(args.le_share)+'-le_res-' + str(args.le_res)+'-onehot-' + str(args.onehot) +'-le_t-' + str(args.le_t)+ '-output_style-' + str(args.output_style) + \
               '_Lr-' + str(args.lr) + '_Loss-' + loss_str + '_sample-' + str(args.enable_sample)+ '_soft_target-' + str(args.soft_target)
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()

# DIV_COE = (torch.arange(0, args.max_len).repeat(args.d_model, 1).transpose(0, 1) + 1).to(args.device)


# def avg_user_representation(pred):
#     # pred in shape B * L * D
#     pred_sum = torch.cumsum(pred, dim=1)
#     pred_new = pred_sum / DIV_COE
#     return pred_new
