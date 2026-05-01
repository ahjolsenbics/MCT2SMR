

import torch
import torch.nn as nn
import numpy as np
import os
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from ctvit.ctvit import CTViT
from modules.UNet3D import UNet3D
from modules.tools import to_one_hot_3d
from modules.asymmetric_phase_perception_aggregator import DynamicPhaseAwareCrossAttention
from torch.nn import MaxPool1d
from modules.optimizers import build_optimizer, build_lr_scheduler


class MCT2SMR(nn.Module):
    def __init__(self, args, tokenizer):
        super(MCT2SMR, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.load_device1 = args.load_device1
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.dim = args.dim
        self.codebook_size = args.codebook_size
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.spatial_depth = args.spatial_depth
        self.temporal_depth = args.temporal_depth
        self.dim_head = args.dim_head
        self.heads = args.heads

        vit = CTViT(
                    dim = self.dim,
                    codebook_size = self.codebook_size,
                    image_size = self.image_size,
                    patch_size = self.patch_size,
                    temporal_patch_size = self.temporal_patch_size,
                    spatial_depth = self.spatial_depth,
                    temporal_depth = self.temporal_depth,
                    dim_head = self.dim_head,
                    heads = self.heads,
                    device=self.device
                )

        self.to_patch_emb = nn.Sequential(
                Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1=self.patch_size, p2=self.patch_size,
                          pt=self.temporal_patch_size),
                nn.LayerNorm(self.out_channels * self.patch_size * self.patch_size * self.temporal_patch_size),
                nn.Linear(self.out_channels * self.patch_size * self.patch_size * self.temporal_patch_size, self.dim),
                nn.LayerNorm(self.dim)
            )

        self.DynamicPhaseAwareCrossAttention = DynamicPhaseAwareCrossAttention(args, self.dim)


        self.visual_extractor_C = VisualExtractor(vit, args)
        self.visual_extractor_A = VisualExtractor(vit, args)
        self.visual_extractor_P = VisualExtractor(vit, args)
        self.visual_extractor_V = VisualExtractor(vit, args)

        with torch.no_grad():

            C_phase_path = "./visual_extractor_checkpoint/C_pahse/C_register_30.pth"
            A_phase_path = "./visual_extractor_checkpoint/A_pahse/A_register_30.pth"
            P_phase_path = "./visual_extractor_checkpoint/P_pahse/P_register_30.pth"
            V_phase_path = "./visual_extractor_checkpoint/V_pahse/V_register_30.pth"

            C_phase_checkpoint = torch.load(C_phase_path, map_location='cpu')
            A_phase_checkpoint = torch.load(A_phase_path, map_location='cpu')
            P_phase_checkpoint = torch.load(P_phase_path, map_location='cpu')
            V_phase_checkpoint = torch.load(V_phase_path, map_location='cpu')

            self.visual_extractor_C.load_state_dict(C_phase_checkpoint['state_dict'])
            self.visual_extractor_A.load_state_dict(A_phase_checkpoint['state_dict'])
            self.visual_extractor_P.load_state_dict(P_phase_checkpoint['state_dict'])
            self.visual_extractor_V.load_state_dict(V_phase_checkpoint['state_dict'])

            self.visual_extractor_C.eval()
            self.visual_extractor_A.eval()
            self.visual_extractor_P.eval()
            self.visual_extractor_V.eval()

        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.forward = self.forward_ct2rep

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def features_load(self, images):
        with torch.no_grad():
            att_feats_C, fc_feats_C = self.visual_extractor_C(images[:, 0, :, :, :].unsqueeze(1))
            att_feats_A, fc_feats_A = self.visual_extractor_A(images[:, 1, :, :, :].unsqueeze(1))
            att_feats_P, fc_feats_P = self.visual_extractor_P(images[:, 2, :, :, :].unsqueeze(1))
            att_feats_V, fc_feats_V = self.visual_extractor_V(images[:, 3, :, :, :].unsqueeze(1))

            att_feats = (att_feats_C, att_feats_A, att_feats_P, att_feats_V)
            fc_feats = (fc_feats_C.unsqueeze(1), fc_feats_A.unsqueeze(1), fc_feats_P.unsqueeze(1), fc_feats_V.unsqueeze(1))

        return att_feats, fc_feats

    def segment_load(self, masks):
        seg_feats = []
        with torch.no_grad():
            for index in range(masks.shape[1]):                # [1,4,240,480,480]
                mask = masks[:, index, :, :, :].unsqueeze(1)   # [1,1,240,480,480]
                mask = to_one_hot_3d(mask.to(torch.long), self.out_channels, mask.device)  # [1,5,240,480,480]
                segfeats = self.to_patch_emb(mask)             # [1, 20, 20, 20, 512]
                segfeats, _ = pack([segfeats], 'b * d')        # [1, 8000, 512]  _:[(20, 20, 20)]  将多维张量重新排列并展平为指定的形状，同时保持批量维度和特征维度不变

                seg_feats.append(segfeats)

        seg_feats = torch.cat(seg_feats, dim=-1)          # [1, 8000, 2048]

        return seg_feats

    def forward_ct2rep(self, images, seg_images=None, block_seq=None, blocktext_len_list=None, targets=None, mode='origin_train'):

        att_feats, fc_feats = self.features_load(images)
        seg_feats = self.segment_load(seg_images)

        att_feats = self.DynamicPhaseAwareCrossAttention(att_feats)
        fc_feats = self.DynamicPhaseAwareCrossAttention(fc_feats)

        if mode == 'origin_train':
            output = self.encoder_decoder(fc_feats, att_feats, seg_feats, targets, block_seq, blocktext_len_list, mode='origin_forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, seg_feats, mode='meta_sample')
        else:
            raise ValueError
        return output


