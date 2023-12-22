
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_


class GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.device = args.device
        self.get_user = args.get_user
        # load parameters info
        self.embedding_size = args.d_model
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_prob = args.dropout

        # define layers and loss
        self.item_embedding = nn.Embedding(args.num_item+1, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
    
        # Output
        self.predict=args.le_share
        if args.le_share== "unshare" or args.soft_target=='unshare':
            self.output = nn.Linear(args.d_model, args.num_item+1)

        # self.output_style = args.output_style
        # if self.output_style == 'avg':
        #     self.DIV_COE = (torch.arange(0, args.max_len).repeat(args.d_model, 1).transpose(0, 1) + 1).to(args.device)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)


    
    def log2feats(self,x):
        item_seq_emb = self.item_embedding(x)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def feat2pred(self,x, target): # 使用该函数获得predict，计算各feature对应各类别的分数
        if target=='share':
            x = F.linear(x, self.item_embedding.weight)
        elif target=='unshare':
            x = self.output(x)
        # 在le loss中使用的soft target
        elif target == 'mlp':
            x = self.mlp(x)
        elif target== 'cos':
            x = cos(x.view(-1,x.size(-1)),self.item_embedding.weight).view(x.size(0),x.size(1),-1)
        elif target == 'euclid':
            x = euclid(x.view(-1,x.size(-1)),self.item_embedding.weight).view(x.size(0),x.size(1),-1)
        return x   # B * L * D --> (B * L)* N 
    
    def avg_user_representation(self,pred):
        # pred in shape B * L * D
        pred_sum = torch.cumsum(pred, dim=1)
        pred_new = pred_sum / self.DIV_COE
        return pred_new
    def forward(self, x):
        
        x = self.log2feats(x)
        # item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        # gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        # gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        # seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        # if self.output_style == 'avg' :
        #     x = self.avg_user_representation(x)
        # x = x[labels>0]
        x = x[:,-1,:]
        if self.get_user:
            return x
        return self.feat2pred(x, self.predict)#, self.feat2pred(x, self.soft_target)   # B * L * D --> (B * L)* N
    

