import argparse
import time
import os
import json
import pandas as pd
import torch
parser = argparse.ArgumentParser()
# dataset and dataloader args


#1 train args  训练中各种设定batch ，lr等
parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'le','bce']) # 确定寻来你的loss 如果验证le4rec使用le loss
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--lr_decay_rate', type=float, default=1)
parser.add_argument('--lr_decay_steps', type=int, default=1250)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--get_user', type=int, default=0)
parser.add_argument('--graph', type=int, default=0)
parser.add_argument('--cold', type=int, default=0)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--use_pop', type=int, default=0)
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--dataname', type=str, default='ce')
parser.add_argument('--early_stop', type=int, default=2)
parser.add_argument('--ld1', type=float, default=0.5)
parser.add_argument('--ld2', type=float, default=0.5)
parser.add_argument('--MA', type=int, default=0)
parser.add_argument('--JS', type=int, default=0)
parser.add_argument('--clean', type=int, default=1)
parser.add_argument('--class_num', type=int, default=2048)
parser.add_argument('--metric_ks', type=list, default=[10,20])
parser.add_argument('--best_metric', type=str, default='NDCG@10')
parser.add_argument('--data_path', type=str, default='/data/zhangh/S/bert4-rec/dataset/movielens20_new.csv')
parser.add_argument('--save_path', type=str, default='None') #设为None则根据预设的参数自动生成文件夹
parser.add_argument('--load_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_len', type=int, default=20) # 序列的长度
parser.add_argument('--mask_prob', type=float, default=0.3) # Bert模型使用MIP方式训练，mask的比率
parser.add_argument('--beta', type=float, default=4)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--smoothing', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=256) 
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--caser_nh', type=int, default=16)
parser.add_argument('--caser_nv', type=int, default=8)
#2 model args 模型通用的设置
parser.add_argument('--model', type=str, default='sasrec', choices=['mlp','GCN','SVD','bert', 'sasrec', 'nextitnet', 'nfm', 'deepfm', 'GRU4Rec','caser']) # 模型选择，模型名见choices
parser.add_argument('--d_model', type=int, default=128) # 输入embedding的长度
parser.add_argument('--eval_per_steps', type=int, default=2) 
parser.add_argument('--enable_res_parameter', type=int, default=1)  # 是否增加残差结构
parser.add_argument('--enable_sample', type=int, default=0) # 1使用采样版本 0full的版本 与loss一起选择相对应
parser.add_argument('--neg_samples', type=int, default=0) # 1使用采样版本 0full的版本 与loss一起选择相对应
parser.add_argument('--sample_strategy', type=str, default='raw_random') # 1使用采样版本 0full的版本 与loss一起选择相对应
parser.add_argument('--sampled_evaluation', type=int, default=0) #确定测试时是否采样
parser.add_argument('--samples_ratio', type=float, default=0.1) # 如果使用采样策略每次采样多少，仅在训练时使用，测试时仅采样100个负样本
parser.add_argument('--output_style', type=str, default='avg') #仅在sasrec nextinet 和 GRURec使用  经过模型输出的特征，是使用最后一项，还是使用avg pooling .last or avg。
parser.add_argument('--embedding_sharing', type=int, default=0) # 如何获得各类别的分数， 0使用linear作为分类头，1直接与embedding矩阵相乘


#3 各个模型不同的设置
# bert sasrec args
parser.add_argument('--attn_heads', type=int, default=4) # 指定使用几头注意力
parser.add_argument('--dropout', type=float, default=0.1)  # 模型中使用的dropout参数
parser.add_argument('--d_ffn', type=int, default=512) # FPN中的维度
parser.add_argument('--bert_layers', type=int, default=2) # bert sasrec的深度


# nextinet args
parser.add_argument('--kernel_size', type=int, default=3) # 卷积核的大小
parser.add_argument('--block_num', type=int, default=32) # nextinet的模型深度
parser.add_argument('--dilations', type=list, default=[1,4]) #空洞卷积的大小

# FM超参
parser.add_argument('--nfm_layers', type=list, default=[128,128]) #  第二个值要和d_model 一致  FM中的linear曾额参数
parser.add_argument('--dfm_layers', type=list, default=[512,128]) # 第二个值要和d_model 一致
parser.add_argument('--drop_prob', type=list, default=[0.1,0.1]) # 第一个是FM的超参，第二个是mlp的超参
parser.add_argument('--act_function', default='relu', type=str , help='mlp模型中使用的activate function relu sigmoid tanh')

# gru超参
parser.add_argument('--num_layers', type=int, default=4) #  有几层GRU
parser.add_argument('--hidden_size', type=int, default=128) #  GRU中隐藏层的维度


#4 LE4Rec label enhance的参数设置
parser.add_argument('--tau', type=float, default=1) 
parser.add_argument('--le_t', type=float, default=1) # KL中的temporature
parser.add_argument('--lamda', type=float, default=0.2) 
parser.add_argument('--lamda_b', type=float, default=0.4) 
parser.add_argument('--le_share', type=str, default='unshare') # le loss中使用哪种方式计算各类别分数 share unshare 
parser.add_argument('--soft_target', type=str, default='unshare', choices=['share', 'unshare', 'mlp', 'cos','euclid']) ## le loss中使用哪种方式计算得到soft_target的值
parser.add_argument('--onehot', type=int, default=1) # le loss中是否使用onehot增强soft_target

parser.add_argument('--le_res', type=float, default=0.1) # target 和soft_target 产生的loss所占用的比例
parser.add_argument('--le_res_type', type=str, default='lp', choices=['hp', 'lp']) #自己设定的  hp hyper-parameters ,可学习的 lp learnable-parameters
parser.add_argument('--mlp_hiddent', type=list, default=[128,1024]) # 第一个值和d_model一致，输出自动给根据数据的项目数量自动计算.soft_target选定为mlp时需设定这个

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
