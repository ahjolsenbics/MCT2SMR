from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils

import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from ctvit.attention import Attention, Transformer, ContinuousPositionBias


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def cast_tuple(val, l=1):
    return val if isinstance(val, tuple) else (val,) * l


# discriminator

class DiscriminatorBlock(nn.Module):
    def __init__(
            self,
            input_channels,
            filters,
            downsample=True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)

        if exists(self.downsample):
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            channels=3,
            attn_res_layers=(16,),
            max_dim=512
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(math.log2(min_image_resolution) - 2)
        attn_res_layers = cast_tuple(attn_res_layers, num_layers)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_block = None
            if image_resolution in attn_res_layers:
                attn_block = Attention(dim=out_chan)

            attn_blocks.append(attn_block)

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            nn.LeakyReLU(0.1),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        for block, attn_block in zip(self.blocks, self.attn_blocks):
            x = block(x)

            if exists(attn_block):
                x, ps = pack([x], 'b c *')
                x = rearrange(x, 'b c n -> b n c')
                x = attn_block(x) + x
                x = rearrange(x, 'b n c -> b c n')
                x, = unpack(x, ps, 'b c *')

        return self.to_logits(x)


# ctvit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder

class MASKViT(nn.Module):
    def __init__(
            self,
            *,
            dim,
            codebook_size,
            image_size,
            patch_size,
            temporal_patch_size,
            spatial_depth,
            temporal_depth,
            dim_head=64,
            heads=8,
            channels=5,
            attn_dropout=0.,
            ff_dropout=0.,
            device="cuda:0"
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.device = device

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads, device=device)  # 计算空间相对位置偏置

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1=patch_height, p2=patch_width,
                      pt=temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)


    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
            self,
            tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=self.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        return tokens


    def forward(
            self,
            video,
            mask=None,
            return_only_codebook_ids=False,
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        tokens = self.to_patch_emb(video)

        shape = tokens.shape
        *_, h, w, _ = shape

        tokens = self.encode(tokens)

        # quantize
        tokens, packed_fhw_shape = pack([tokens], 'b * d')
        vq_mask = None
        if exists(mask):
            vq_mask = repeat(mask, 'b f -> b (f h w)', h=h, w=w) # Simplified mask calculation based on context.
        tokens, indices, commit_loss = self.vq(tokens, mask=vq_mask)
        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        return tokens