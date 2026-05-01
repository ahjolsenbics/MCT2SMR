import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F


def Metaclones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MetaDecoder(nn.Module):
    def __init__(self, args, layer, N):
        super(MetaDecoder, self).__init__()
        self.args = args
        self.layers = Metaclones(layer, N)
        self.norm = MetaLayerNorm(layer.d_model)
        self.mask_linear = nn.Linear(8000, args.d_model)

    def forward(self, x, meta_token, src_mask, tgt_mask, memory):
        # x[1,i-1,2048] hidden_states[1,8000,2048] tgt_mask[1,i-1,i-1] memory[i-1,6144]
        for layer in self.layers: # ModuleList:3
            x = layer(x, meta_token, src_mask, tgt_mask, memory)
        return self.norm(x)  # x[1,i-1,2048]

class MetaDecoderLayer(nn.Module):
    def __init__(self, args,d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(MetaDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.d_model = d_model
        self.sublayer = Metaclones(MetaConditionalSublayerConnection(args.total_vocab_size, d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, meta_token, src_mask, tgt_mask, memory):
        m = meta_token
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        MD_out = self.sublayer[2](x, self.feed_forward, memory)

        return MD_out

class MetaSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(MetaSublayerConnection, self).__init__()
        self.norm = MetaLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        s_out = x + self.dropout(sublayer(self.norm(x)))  # x:[1,8000,2048]
        return s_out

class MetaLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(MetaLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MetaConditionalSublayerConnection(nn.Module):
    def __init__(self, total_vocab_size, d_model, dropout, rm_num_slots, rm_d_model):
        super(MetaConditionalSublayerConnection, self).__init__()
        self.norm = MetaConditionalLayerNorm(total_vocab_size, d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        csc_out = x + self.dropout(sublayer(self.norm(x, memory)))  # [1,i,2048]
        return csc_out


class MetaConditionalLayerNorm(nn.Module):
    def __init__(self, total_vocab_size, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(MetaConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, rm_d_model),  # [in_features:6144-->out_features:2048]
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, rm_d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(rm_d_model, rm_d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)       # [1,i-1,1]
        std = x.std(-1, keepdim=True)         # [1,i-1,1]
        delta_gamma = self.mlp_gamma(memory)  # [1,i-1,2048]
        delta_beta = self.mlp_beta(memory)    # [1,i-1,2048]
        gamma_hat = self.gamma.clone()        # [2048]的全为1的矩阵
        beta_hat = self.beta.clone()          # [2048]的零矩阵
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)  # [1,2048]
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)  # [1,i-1,2048]
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)    # [1,2048]
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)    # [1,i-1,2048]
        gamma_hat += delta_gamma  # [1,i-1,2048]
        beta_hat += delta_beta    # [1,i-1,2048]
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat    # [1,i-1,2048]


def Meta_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)   # decode时, key:[1,8,i-1/8000,256] query:[1,8,i-1,256] value:[1,8,i-1/8000,256]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # decode scores:[1,8,i-1,i-1]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # decode scores:[1,8,i-1,i-1]
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MetaMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MetaMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = Metaclones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)   # decode时, query:[1,i-1,2048] key:[1,i-1/8000,2048] value:[1,i-1/8000,2048]  第一轮时qkv值一样，第二轮key和value的一样
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]    # decode时, query:[1,8,i-1,256]  key:[1,8,i-1/8000,256]value:[1,8,i-1/8000,256]

        x, self.attn = Meta_attention(query, key, value, mask=mask, dropout=self.dropout)  # encode时 x:[1,8,8000,256]  self.attn:[1,8,8000,8000]  decode时 x:[1,8,i-1,256]  self.attn:[1,8,i-1,i-1/8000]

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # encode时 x:[1,8000,2048]  decode时 x:[1,i-1,2048]
        out = self.linears[-1](x)  # encode时 out:[1,8000,2048]  decode时 out:[1,i-1,2048]
        return out

class MetaPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MetaPositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MetaPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(MetaPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
