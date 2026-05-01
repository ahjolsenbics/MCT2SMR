from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import reshape
from paddle.distributed.fleet.fleet_executor_utils import origin
from skimage.color.rgb_colors import black
from timm.models.vision_transformer import PatchEmbed
from .att_model import pack_wrapper, AttModel
from .Meta_Decdoer import MetaDecoder, MetaDecoderLayer, MetaMultiHeadedAttention, MetaPositionwiseFeedForward
from .fusion_decoder import FusionDecoder, FusionDecoderLayer, FusionMultiHeadedAttention, FusionPositionwiseFeedForward


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)   # decode时, key:[1,8,i-1/8000,256] query:[1,8,i-1,256] value:[1,8,i-1/8000,256]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # decode scores:[1,8,i-1,i-1]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # scores:[1,8,i-1,i-1]  mask:[1,1,i-1,i-1]
    p_attn = F.softmax(scores, dim=-1)  # decode scores:[1,8,i-1,i-1]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, args, encoder, decoder, src_embed, tgt_embed, rm, meta_decoder):
        super(Transformer, self).__init__()
        self.args = args
        self.refine = args.refine
        self.num_slots = args.rm_num_slots
        self.d_model = args.rm_d_model
        self.total_vocab_size = args.total_vocab_size

        self.decoder = decoder
        self.encoder = encoder
        self.meta_decoder = meta_decoder
        # self.fuion_decoder = FusionDecoder

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.num_meta_token =1
        self.num_meta_dim = 20
        self.meta_linear = nn.Linear(self.d_model, self.num_meta_dim)
        self.logit = nn.Linear(self.d_model, self.total_vocab_size)
        self.re_logit = nn.Linear(self.total_vocab_size, self.d_model)

    def forward(self, src, tgt, src_mask, tgt_mask):
        output = self.encode(src, src_mask)  # [1,8000,2048]
        outputs = self.decode(output, src_mask, tgt, tgt_mask)

        return outputs

    def encode(self, src, src_mask):
        encode_out = self.encoder(self.src_embed(src), src_mask)
        return encode_out  # [1,8000,2048]

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, mode=None):
        tgt_embed = self.tgt_embed(tgt)
        if mode is None:
            summary_tgt = torch.sum(tgt_embed, dim=1).unsqueeze(1)
            tgt_embed = torch.cat([tgt_embed, summary_tgt], dim=1)

        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.refine, tgt_embed, memory)
        decode_out = self.decoder(tgt_embed, hidden_states, src_mask, tgt_mask, memory)  # tgt_embed[1,i-1,2048] src_mask[1,1,8000]全1 tgt_mask[1,i-1,i-1]  memory[1,i-1,6144]

        return decode_out  # train: [1,i-1,2048]  test:[1,1,2048]

    def meta_decode(self, meta_token, tgt):
        tgt_mask = (tgt.data > 0)
        tgt_mask[:, 0] += True
        tgt_mask = tgt_mask.unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(tgt_mask)

        src_mask = meta_token.new_ones(meta_token.shape[:2], dtype=torch.long)
        src_mask = src_mask.unsqueeze(-2)

        memory = self.rm.init_memory(meta_token.size(0)).to(meta_token)
        memory = self.rm(self.refine, self.tgt_embed(tgt), memory)
        meta_decode_out = self.meta_decoder(self.tgt_embed(tgt), meta_token, src_mask, tgt_mask, memory)

        return meta_decode_out

    # def fusion_decode(self, fusion_hidden_states, src_mask, tgt, tgt_mask):
    #     memory = self.rm.init_memory(fusion_hidden_states.size(0)).to(fusion_hidden_states)
    #     memory = self.rm(self.refine, self.tgt_embed(tgt), memory)
    #     fusion_out = self.fusion_decoder(self.tgt_embed(tgt), fusion_hidden_states, src_mask, tgt_mask, memory)
    #
    #     return fusion_out


class Encoder(nn.Module):
    def __init__(self, args, layer, N):
        super(Encoder, self).__init__()
        self.args = args
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # [1,8000,2048]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # [1,8000,2048]
        out = self.sublayer[1](x, self.feed_forward)  # x:[1,8000,2048]
        return out


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        s_out = x + self.dropout(sublayer(self.norm(x)))  # x:[1,8000,2048]
        return s_out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, args, layer, N):
        super(Decoder, self).__init__()
        self.args = args
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        # x[1,i-1,2048] hidden_states[1,8000,2048] tgt_mask[1,i-1,i-1] memory[i-1,6144]
        for layer in self.layers:  # ModuleList:3
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)  # x[1,i-1,2048]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):

        m = hidden_states  # x[1,i-1,2048] hidden_states[1,8000,2048] src_mask[1,1,8000] tgt_mask[1,i-1,i-1] memory[1,i-1,6144]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory) # x[1,i-1,2048]  q[1,i-1,2048] k[1,i-1,2048] v[1,i-1,2048]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)  # x[1,i-1,2048]  q[1,i-1,2048] k[1,8000,2048] v[1,8000,2048]
        d_out = self.sublayer[2](x, self.feed_forward, memory)  # d_out[1,i,2048]
        return d_out


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        csc_out = x + self.dropout(sublayer(self.norm(x, memory)))  # [1,i,2048]
        return csc_out


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),  # [in_features:6144-->out_features:2048]
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)  # [1,i-1,1]
        std = x.std(-1, keepdim=True)  # [1,i-1,1]
        delta_gamma = self.mlp_gamma(memory)  # [1,i-1,2048]
        delta_beta = self.mlp_beta(memory)    # [1,i-1,2048]
        gamma_hat = self.gamma.clone()  # [2048]的全为1的矩阵
        beta_hat = self.beta.clone()  # [2048]的零矩阵
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)  # [1,2048]
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)  # [1,i-1,2048]
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)    # [1,2048]
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)    # [1,i-1,2048]
        gamma_hat += delta_gamma  # [1,i-1,2048]
        beta_hat += delta_beta    # [1,i-1,2048]
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat    # [1,i-1,2048]


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)   # decode时, query:[1,i-1,2048] key:[1,i-1/8000,2048] value:[1,i-1/8000,2048]  第一轮时qkv值一样，第二轮key和value的一样
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]    # decode时, query:[1,8,i-1,256]  key:[1,8,i-1/8000,256]value:[1,8,i-1/8000,256]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # encode时 x:[1,8,8000,256]  self.attn:[1,8,8000,8000]  decode时 x:[1,8,i-1,256]  self.attn:[1,8,i-1,i-1/8000]

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # encode时 x:[1,8000,2048]  decode时 x:[1,i-1,2048]
        out = self.linears[-1](x)  # encode时 out:[1,8000,2048]  decode时 out:[1,i-1,2048]
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)  # 3阶单元矩阵
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots  # 2048 -3 = 2045
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)  # 前三列的对角线为1，其它内容都为0，剩下2045列全为0，共3行，2048列
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory  # q[1,3,2048]
        k = torch.cat([memory, input.unsqueeze(1)], 1)  # k[1,4,2048]
        v = torch.cat([memory, input.unsqueeze(1)], 1)  # v[1,4,2048]
        next_memory = memory + self.attn(q, k, v)   # [1,3,2048]
        next_memory = next_memory + self.mlp(next_memory)  # [1,3,2048]

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)    # [1,3,2048]
        forget_gate = torch.sigmoid(forget_gate)  # [1,3,2048]

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory  # [1,3,2048]
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)  # [1,6144]

        return next_memory

    def forward(self, refine, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):  # 这里控制根据gt，来控制预测的输出长度和内容
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)  # lsit:i-1  根据输入gt长度为准
        outputs = torch.stack(outputs, dim=1)  # [1,i-1,6144]
        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        meta_attn = MetaMultiHeadedAttention(self.num_heads, self.d_model)
        meta_ff = MetaPositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(self.args,
            Encoder(self.args, EncoderLayer(self.d_model, c(attn), c(ff), self.dropout),self.num_layers),
            Decoder(self.args,
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),rm,
            MetaDecoder(self.args,
                MetaDecoderLayer(self.args, self.d_model, c(meta_attn), c(meta_attn), c(meta_ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.meta_num_layers))

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.meta_num_layers = args.meta_num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        self.tgt_vocab = self.total_vocab_size

        self.model = self.make_model(self.tgt_vocab)

        self.logit = nn.Linear(self.d_model, self.tgt_vocab)
        self.re_logit = nn.Linear(self.tgt_vocab, self.d_model)

        self.block1_model = self.model
        self.block2_model = self.model
        self.block3_model = self.model
        self.block4_model = self.model
        self.block5_model = self.model
        self.block6_model = self.model

        with torch.no_grad():
            block1_checkpoint_path = './block1_checkpoint/meta_models/meta_model_30.pth'
            block2_checkpoint_path = './block2_checkpoint/meta_models/meta_model_30.pth'
            block3_checkpoint_path = './block3_checkpoint/meta_models/meta_model_30.pth'
            block4_checkpoint_path = './block4_checkpoint/meta_models/meta_model_30.pth'
            block5_checkpoint_path = './block5_checkpoint/meta_models/meta_model_30.pth'
            block6_checkpoint_path = './block6_checkpoint/meta_models/meta_model_30.pth'
        
        
            block1_checkpoint = torch.load(block1_checkpoint_path, map_location='cpu') 
            block2_checkpoint = torch.load(block2_checkpoint_path, map_location='cpu')
            block3_checkpoint = torch.load(block3_checkpoint_path, map_location='cpu')
            block4_checkpoint = torch.load(block4_checkpoint_path, map_location='cpu')
            block5_checkpoint = torch.load(block5_checkpoint_path, map_location='cpu')
            block6_checkpoint = torch.load(block6_checkpoint_path, map_location='cpu')
        
            self.block1_model.load_state_dict(block1_checkpoint['state_dict'])
            self.block2_model.load_state_dict(block2_checkpoint['state_dict'])
            self.block3_model.load_state_dict(block3_checkpoint['state_dict'])
            self.block4_model.load_state_dict(block4_checkpoint['state_dict'])
            self.block5_model.load_state_dict(block5_checkpoint['state_dict'])
            self.block6_model.load_state_dict(block6_checkpoint['state_dict'])
        
            self.block1_model.eval()
            self.block2_model.eval()
            self.block3_model.eval()
            self.block4_model.eval()
            self.block5_model.eval()
            self.block6_model.eval()

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):   # [1,8000,2048] [1,1,2048]

        att_feats, att_masks, _, _, _ = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)  # [1,8000,2048]

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks, orgin_seq=None, block_ids=None, blocktext_len_list=None):  # att_feats[1,8000,3072] att_masks=None  seq[1,i]
        att_feats, att_masks = self.clip_att(att_feats, att_masks)            # att_feats[1,8000,3072] att_masks=None
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)        # att_feats[1,8000,2048]

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)  # [1, 8000]
        att_masks = att_masks.unsqueeze(-2)  # [1,1,8000]

        origin_block_ids_masks = []
        if block_ids is not None:
            for index in range(block_ids.shape[1]):
                seq = block_ids[:, index]
                seq = seq[:, :max(blocktext_len_list[:, index])]
                seq_mask = (seq.data > 0)
                seq_mask[:, 0] += True
                new_mata_mask = torch.tensor([[True]], dtype=torch.bool,device=seq_mask.device)  # 初始化meta_token的mask
                seq_mask = torch.cat([seq_mask, new_mata_mask], dim=1)  # 添加meta_token的mask

                seq_mask = seq_mask.unsqueeze(-2)
                seq_mask = seq_mask & subsequent_mask(seq.size(-1) + 1).to(seq_mask)
                origin_block_ids_masks.append(seq_mask)

        if orgin_seq is not None:
            orgin_seq = orgin_seq[:, :-1]
            orgin_seq_masks = (orgin_seq.data > 0)
            orgin_seq_masks[:, 0] += True
            orgin_seq_masks = orgin_seq_masks.unsqueeze(-2)
            orgin_seq_masks = orgin_seq_masks & subsequent_mask(orgin_seq.size(-1)).to(orgin_seq_masks)
        else:
            orgin_seq_masks = None

        return att_feats, att_masks, orgin_seq, orgin_seq_masks, origin_block_ids_masks

    def _prepare_fusion_feature_forward(self, fusion_att_feats):

        fusion_att_masks = fusion_att_feats.new_ones(fusion_att_feats.shape[:2], dtype=torch.long)
        fusion_att_masks = fusion_att_masks.unsqueeze(-2)

        return fusion_att_masks

    def _load_meta_outputs(self, att_feats, att_masks, seg_feats, block_ids, org_block_ids_masks, blocktext_len_list):

        orgin_b1_out = self.block1_model.decode(att_feats, att_masks, block_ids[:, 0, :][:, :max(blocktext_len_list[:, 0])], org_block_ids_masks[0])
        orgin_b2_out = self.block2_model.decode(att_feats, att_masks, block_ids[:, 1, :][:, :max(blocktext_len_list[:, 1])], org_block_ids_masks[1])
        orgin_b3_out = self.block3_model.decode(att_feats, att_masks, block_ids[:, 2, :][:, :max(blocktext_len_list[:, 2])], org_block_ids_masks[2])
        orgin_b4_out = self.block4_model.decode(att_feats, att_masks, block_ids[:, 3, :][:, :max(blocktext_len_list[:, 3])], org_block_ids_masks[3])
        orgin_b5_out = self.block5_model.decode(att_feats, att_masks, block_ids[:, 4, :][:, :max(blocktext_len_list[:, 4])], org_block_ids_masks[4])
        orgin_b6_out = self.block6_model.decode(att_feats, att_masks, block_ids[:, 5, :][:, :max(blocktext_len_list[:, 5])], org_block_ids_masks[5])

        meta_tokens = torch.cat([orgin_b1_out[:, -1:, :], orgin_b2_out[:, -1:, :],
                                 orgin_b3_out[:, -1:, :], orgin_b4_out[:, -1:, :], orgin_b5_out[:, -1:, :], orgin_b6_out[:, -1:, :]], dim=1)

        fusion_att_feats = att_feats + seg_feats

        fusion_att_feats = torch.cat([fusion_att_feats, meta_tokens], dim=1)

        fusion_att_masks = self._prepare_fusion_feature_forward(fusion_att_feats)

        return fusion_att_feats, fusion_att_masks

    def _origin_forward(self, fc_feats, att_feats, seg_feats, seq, block_seq, blocktext_len_list, att_masks=None):
        att_feats, att_masks, seq, seq_masks, origin_block_seq_masks = self._prepare_feature_forward(att_feats, att_masks, seq, block_seq, blocktext_len_list)  # att_feats[1,8000,2048] seq[1,i-1] att_masks[1,1,8000] seq_masks[1,i-1,i-1}
        att_feats = self.model.encode(att_feats, att_masks)

        fusion_att_feats, fusion_att_masks = self._load_meta_outputs(att_feats, att_masks, seg_feats, block_seq, origin_block_seq_masks, blocktext_len_list)


        fusion_out = self.model.decode(fusion_att_feats, fusion_att_masks, seq, seq_masks, mode='fusion')

        outputs = F.log_softmax(self.logit(fusion_out), dim=-1)

        return outputs

    def meta_core(self, it, memory, state, mask, b_num):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        if b_num == 0:
            origin_out = self.block1_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))
        elif b_num == 1:
            origin_out = self.block2_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))
        elif b_num == 2:
            origin_out = self.block3_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))
        elif b_num == 3:
            origin_out = self.block4_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))
        elif b_num == 4:
            origin_out = self.block5_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))
        elif b_num == 5:
            origin_out = self.block6_model.decode(memory, mask, ys, subsequent_mask(ys.size(1) + 1).to(memory.device))

        meta_token = origin_out[:, -1]

        return origin_out[:, -2], [ys.unsqueeze(0)], meta_token.unsqueeze(0)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        origin_out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), mode='fusion')

        return origin_out[:, -1], [ys.unsqueeze(0)]
