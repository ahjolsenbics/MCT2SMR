import torch
from monai.networks.nets import UNet
import torch.nn.functional as F


class UNet3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims, channels, strides, num_res_units):
        super(UNet3D, self).__init__()

        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

    def forward(self, x):
        out = self.model(x)

        return out
