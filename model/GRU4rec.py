
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

        self.output = nn.Linear(args.d_model, args.num_item+1)

      
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

    def feat2pred(self,x): # 使用该函数获得predict，计算各feature对应各类别的分数
        x = self.output(x)
        return x   # B * L * D --> (B * L)* N 
    
  
    def forward(self, x):
        x = self.log2feats(x)
        x = x[:,-1,:]
        if self.get_user:
            return x
        return self.feat2pred(x) 
    

