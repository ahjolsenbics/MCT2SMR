import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.out_channels = args.out_channels
        self.patch_height = self.patch_size
        self.patch_width = self.patch_size
        self.dim = args.dim

        self.to_patch_emb = nn.Sequential(  # 240, 480, 480   :   12 24 24
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1=self.patch_size, p2=self.patch_size,
                      pt=self.temporal_patch_size),
            nn.LayerNorm(self.out_channels * self.patch_width * self.patch_height * self.temporal_patch_size),
            nn.Linear(self.out_channels * self.patch_size * self.patch_size * self.temporal_patch_size, self.dim),
            nn.LayerNorm(self.dim)
        )


    def patch_reshape(self, patch_feats):

        patch_feats = patch_feats.permute(0, 4, 1, 2, 3)             # [1, 20, 20, 20, 512] -->   [1, 512, 20, 20, 20]
        batch_size, feat_size, _, _, _ = patch_feats.shape                                # batch=1, feat_size=512
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        return patch_feats

    def forward(self, masks):
        mask_patch_feats = self.to_patch_emb(masks)
        mask_patch_feats = self.patch_reshape(mask_patch_feats)

        return mask_patch_feats
