import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):  # Stage-2 of Attention mechanism (FeedForward)
    def __init__(
        self,
        d_model, dFF
    ):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.dFF = dFF

        self.FF1 = nn.Linear(self.d_model, self.dFF)
        self.FF2 = nn.Linear(self.dFF, self.d_model)
        
    def forward(self, x):
        x = self.FF1(x)
        x = F.relu(x)
        x = self.FF2(x)

        return x

class SMHA(nn.Module):  #Self Multi-Head Attention
    def __init__(
        self,
        d_model, dQK, dV, HeadsNum,
        mask, pos, pos_Num
        # maxlength=42
    ):
        super(SMHA, self).__init__()
        self.d_model = d_model
        self.dQK = dQK
        self.dV = dV
        self.HeadsNum = HeadsNum
        self.mask = mask.to('cuda')
        self.pos = pos.to('cuda')
        self.pos_Num = pos_Num
        # self.maxlength = maxlength

        # Weights 
        self.W_Q = torch.nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.HeadsNum, self.dQK, self.dQK))
            )
        )
        self.W_K = torch.nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.HeadsNum, self.dQK, self.dQK))
            )
        )
        self.W_V = torch.nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.HeadsNum, self.dQK, self.dV))
            )
        )
        self.W_Z = torch.nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.HeadsNum*self.dV, self.d_model))
            )
        )

        self.Pos_K = nn.Embedding(self.pos_Num, self.dQK)
        self.Pos_V = nn.Embedding(self.pos_Num, self.dV)

    def forward(self, x):
        self.mask = self.mask.to('cuda')
        self.pos = self.pos.to('cuda')

        BS, TS, _ = x.size()
        
        x = x.view(
            BS, TS, self.HeadsNum, self.d_model//self.HeadsNum
        ).permute(0,2,1,3)

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        x = Q @ K.permute(0,1,3,2)

        x += torch.einsum(
            'bhqd,qkd->bhqk', Q, self.Pos_K(self.pos)
        )

        x = x + self.mask
        x = x / torch.sqrt(torch.tensor(self.dQK))
        x = F.softmax(x, dim=-1)

        V = x @ V

        x = V + torch.einsum(
            'bhqv,qvd->bhqd', x, self.Pos_V(self.pos)
        )

        x = x.permute(0,2,1,3)
        x = x.contiguous().view(BS, TS, self.HeadsNum*self.dV)

        x = x @ self.W_Z
        
        return x

    def __repr__(self):
        return (
            f'SMHA('
            f'\n\tW_Q shape = {tuple(self.W_Q.shape)},'
            f'\n\tW_K shape = {tuple(self.W_K.shape)},'
            f'\n\tW_V shape = {tuple(self.W_V.shape)},'
            f'\n\tW_Z shape = {tuple(self.W_Z.shape)}'
            f'\n and also there are 2 Embeddings)'
        )

class Decoder_Block(nn.Module):
    def __init__(
        self,
        d_model, dQK, dV, HeadsNum, dFF,
        mask, pos, pos_Num,
        # dropout
    ):
        super(Decoder_Block, self).__init__()
        self.d_model = d_model
        self.dFF = dFF
        self.dQK = dQK
        self.dV = dV
        self.HeadsNum = HeadsNum
        self.mask = mask
        self.pos = pos
        self.pos_Num = pos_Num
        # self.dropout = dropout

        # Layers
        self.SMHA = SMHA(
            self.d_model, self.dQK, self.dV, self.HeadsNum,
            self.mask, self.pos, self.pos_Num
            )
        self.FFN = FFN(self.d_model, self.dFF)
        self.Norm1 = nn.RMSNorm(self.d_model)
        self.Norm2 = nn.RMSNorm(self.d_model)

        # self.Norm1 = nn.LayerNorm(self.d_model)
        # self.Norm2 = nn.LayerNorm(self.d_model)
        # self.Drop1 = nn.Dropout(self.dropout)
        # self.Drop2 = nn.Dropout(self.dropout)

    def forward(self, x):
        # sub_out = self.SMHA(x)
        # x = self.Norm1(x + self.Drop1(sub_out))
        # sub_out = self.FFN(x)
        # x = self.Norm2(x + self.Drop2(sub_out))

        ######################
        x = x + self.SMHA(self.Norm1(x))
        x = x + self.FFN(self.Norm2(x))
        return x

'''
# class Embeddings_Layer(nn.Module):
#     def __init__(
#         self, BLocksNum,
#         classes_num, position,
#         max_TS, symbols_Num,
#         d_model, mask
#     ):
#         super(Decoder, self).__init__()
#         self.BLocksNum = BLocksNum
#         self.classes_num = classes_num
#         self.position = position
#         self.max_TS = max_TS
#         self.symbols_Num = symbols_Num
#         self.d_model = d_model
#         self.mask = mask

#         # self.position = self.position.to('cuda')

#         self.Tokens_Embeddings = nn.Embedding(self.classes_num, self.d_model)
#         self.Position_Embeddings = nn.Embedding(self.max_TS, self.d_model)
#         self.Symbols_Embeddings = nn.Embedding(self.symbols_Num, self.d_model)
'''

class Decoder(nn.Module):
        
    def __init__(
        self, BLocksNum, classes_num,
        max_TS, symbols_Num, context_size,
        d_model, dQK, dV, HeadsNum, dFF,
        mask, pos, pos_Num,
        cutoffs, div_value,
        # dropout
    ):
        super(Decoder, self).__init__()
        self.BLocksNum = BLocksNum
        self.classes_num = classes_num
        
        self.max_TS = max_TS
        self.symbols_Num = symbols_Num
        self.context_size = context_size

        self.d_model = d_model
        self.dQK = dQK
        self.dV = dV
        self.HeadsNum = HeadsNum
        self.dFF = dFF

        self.mask = mask
        self.pos = pos
        self.pos_Num = pos_Num

        self.cutoffs = cutoffs
        self.div_value = div_value
        # self.dropout = dropout

        self.Sentence_Embeddings = nn.Embedding(1, self.d_model)
        self.Tokens_Embeddings = nn.Embedding(self.classes_num, self.d_model)
        self.Symbols_Embeddings = nn.Embedding(self.symbols_Num, self.d_model)
        
        self.BlocksList = nn.ModuleList(
            [
                Decoder_Block(
                    self.d_model, self.dQK, self.dV, self.HeadsNum, self.dFF,
                    self.mask, self.pos, self.pos_Num
                    # self.dropout,
                ) for i in range(BLocksNum)
            ]
        )

        # self.Norm = nn.LayerNorm(self.d_model)
        self.Norm = nn.RMSNorm(self.d_model)

        self.Softmax = nn.AdaptiveLogSoftmaxWithLoss(
            self.d_model, self.classes_num, self.cutoffs, self.div_value
        )
        
    def forward(self, x, y=None, ignored=None, PNT=False):
        s = self.Sentence_Embeddings(x[:, 0:1])
        sym = self.Symbols_Embeddings(x[:, 1:2])
        x = self.Tokens_Embeddings(x[:, 2:])

        x = torch.cat((s, sym, x), 1)
        
        for i in torch.arange(self.BLocksNum):
            x = self.BlocksList[i](x)

        x = self.Norm(x)

        s = x[:, 0].contiguous()

        if PNT:
            Pad, Weekend, missing = ignored

            x = x[:, 2+self.context_size:].contiguous(
                ).view(-1, x.shape[-1])
            y = y[:, 1+self.context_size:].contiguous(
                ).view(-1)

            x = x[y!=Weekend]
            y = y[y!=Weekend]
            x = x[y!=missing]
            y = y[y!=missing]

            output, PNT_loss = self.Softmax(x, y)

            PNT_loss = PNT_loss * y.size()[0]

            with torch.no_grad():
                preds = self.Softmax.log_prob(x)

            return output, PNT_loss, preds, s
        
        else:
            return s
        
def Generate_Positional_Tokens(max_TS, a_min=None, a_max=None):
    sentence = np.concatenate(
        [
            ['$', '*'],
            np.char.add('#', np.arange(max_TS,0,-1).astype(str))
        ],
        axis=0
    )[np.newaxis, :]
    
    symbol = ['@'] * (max_TS+2)
    symbol[1] = '*'
    symbol = np.array(symbol)[np.newaxis, :]

    data = np.concatenate(
        [
            np.array(['@'] * (max_TS))[:, np.newaxis],
            np.array(['*'] * (max_TS))[:, np.newaxis],
        ],
        axis=1
    )

    if a_min is None and a_max is None:
        data = np.concatenate(
            [
                data,
                np.arange(max_TS)[:, np.newaxis] - np.arange(max_TS)[np.newaxis, :]
            ], axis=1
        )
    else:
        data = np.concatenate(
            [
                data,
                np.clip(
                    np.arange(max_TS)[:, np.newaxis] - np.arange(max_TS)[np.newaxis, :],
                    a_min=a_min, a_max=a_max
                )
            ], axis=1
        )
    
    Position = np.concatenate([sentence, symbol, data], axis=0)

    Unique_Positional_Tokens = np.unique(Position)
    Tokens_Number = Unique_Positional_Tokens.shape[0]

    Tokens_id = np.arange(Tokens_Number)

    Positional_Tokens_Dictionary = dict(
        zip(
            Unique_Positional_Tokens,
            Tokens_id
        )
    )
    Reverse_Positional_Tokens_Dictionary = dict(
        zip(
            Tokens_id,
            Unique_Positional_Tokens
        )
    )

    Position = np.vectorize(
        Positional_Tokens_Dictionary.get
    )(Position)

    Position = torch.tensor(Position)
    Tokens_Number = torch.tensor(Tokens_Number)

    return Position, Tokens_Number, Positional_Tokens_Dictionary, Reverse_Positional_Tokens_Dictionary

def Generate_Mask_Tokens(max_TS):
    sentence = np.ones((1, 2+max_TS))
    symbol = np.zeros((1, 2+max_TS))
    symbol[0,1] = 1
    
    data = np.concatenate(
        [
            np.zeros((max_TS,1)),
            np.ones((max_TS,1)),
            np.tri(max_TS, max_TS, 0)
        ],
        axis=1
    )
    
    mask = np.concatenate([sentence, symbol, data], axis=0)
    mask[mask==0] = -np.inf
    mask[mask==1] = 0

    mask = mask.astype(np.float32)
    
    mask = torch.tensor(mask[np.newaxis, np.newaxis, :, :])

    return mask

'''
BS = 3
TS = 4
max_TS = TS
d_model, dQK, dV, HeadsNum, dFF = 10, 5, 5, 2, 20
BLocksNum, classes_num = 2, 100
symbols_Num, context_size = 2, TS
cutoffs, div_value = [5, 20], 1.0 #4.0

mask = Generate_Mask_Tokens(max_TS).to('cuda')

pos, pos_Num, Pos_Dict, Rev_Pos_Dict = Generate_Positional_Tokens(
    max_TS=max_TS, a_min=-1, a_max=None
)
pos = pos.to('cuda')
pos_Num = pos_Num.to('cuda')

# input = np.random.random((BS, max_TS+2, d_model)).astype(np.float32)
input = np.concatenate(
    [
        # np.repeat(0, BS).reshape(BS,1),
        np.zeros((BS, 1)),
        np.random.randint(0, symbols_Num, (BS, 1)),
        np.random.randint(0, classes_num, (BS, max_TS))
    ],
    axis=-1 
).astype(int)

input = torch.tensor(input).to('cuda')
print('input')
print(input.size())
print(input)

# SMHA_Block = SMHA(
#     d_model, dQK, dV, HeadsNum,
#     mask, pos, pos_Num
# ).to('cuda')
# print(SMHA_Block)

# Dec_Block = Decoder_Block(
#     d_model, dQK, dV, HeadsNum, dFF,
#     mask, pos, pos_Num
# ).to('cuda')
# print(Dec_Block)

print('model')
Decoder_Model = Decoder(
    BLocksNum, classes_num,
    max_TS, symbols_Num, context_size,
    d_model, dQK, dV, HeadsNum, dFF,
    mask, pos, pos_Num,
    cutoffs, div_value
).to('cuda')
print(Decoder_Model)

# output = SMHA_Block(input)
# output = Dec_Block(input)
output = Decoder_Model(input)

print('output')
print(output.size())
print(output)
'''


