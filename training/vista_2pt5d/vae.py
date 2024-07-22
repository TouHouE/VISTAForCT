import re
from typing import Callable, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F
import monai.data
import numpy as np
from icecream import ic
ic.configureOutput(includeContext=True)


def get_vista_hidden_size(vista_type):
    if (scale_char := re.split(r'[-_]', vista_type.lower())[-1]) == 'b':
        return 768
    elif scale_char == 'l':
        return 1024
    elif scale_char == 'h':
        return 1280
    else:
        raise NotImplementedError(f'Vista Type: {vista_type} not implement now.')


class UpResBlock(nn.Module):
    GROUP = 8

    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=(2., 2.))
        )
        self.block = nn.Sequential(
            nn.GroupNorm(self.GROUP, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(self.GROUP, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        # ic(x.shape)
        x = self.upsample(x)
        # ic(x.shape)
        buf_x = self.block(x)
        # ic(buf_x.shape)
        x += buf_x
        return x


class VAEDecoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            raw_image_shape: Sequence[Union[int, int, int]],
            latent_dim: int = 256,
            estimate_std: bool = True,
            default_std: float | int = .3,
            act_func: Callable = nn.ReLU,
            image_loss_func: Callable = nn.MSELoss,
            return_pseudo_image: bool = False
    ):
        """
        I try to simulate the vae decoder that SegResnet for vista2p5d segmentation model
        @param in_channels(int): The vision encoder output dimension.
        @param raw_image_shape(Sequence[int, int, int]): contains the original image shape with (c, h, w)
        @param latent_dim(int): The size of the representation embedding
        @param estimate_std(bool): shall we predict std?
        @param default_std(float | int): use this value when you don't predict std. default=0.3
        @param act_func(Callable): given a constructor or a function. default: nn.ReLU
        @param image_loss_func(Callable): given a constructor or a function. default: nn.MSELoss
        @param return_pseudo_image(bool): shall we return that reconstructed image?
        """
        super().__init__()
        self.return_pseudo_image: bool = return_pseudo_image
        self.raw_image_shape: tuple[int, int, int] = raw_image_shape    # channel, height, width
        self.image_embryo_hw = [length // 8 for length in raw_image_shape[-2:]] # 26, 26
        self.latent_dim: int = latent_dim   # 256
        self.in_channels: int = in_channels     # 256
        self.embryo_channels: int = in_channels // 4    # 64
        self.estimate_std = estimate_std
        self.default_std = default_std
        self.embryo_tot_element = (np.prod(self.image_embryo_hw) // 4) * self.embryo_channels

        # The image size is reduced by patch embedding, not downsampling, because the vision encoder is ViT.
        # Fortunately, the patch size in Vista2pt5d is setting as 16, it can as down sample 4 times.
        # Based on this point, I DO NOT reduce the image shape at this stage.
        self.down_sample = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.embryo_channels, 3, padding=1),
            nn.GroupNorm(8, self.embryo_channels),
            nn.ReLU()
        )
        self.prefix_up_sample = nn.Sequential(
            nn.Conv2d(self.embryo_channels, self.in_channels, 1),
            nn.UpsamplingBilinear2d(scale_factor=(2., 2.)),
            nn.GroupNorm(8, self.in_channels),
            act_func(),
        )

        self.mean_estimator = nn.Linear(self.embryo_tot_element, latent_dim)
        self.std_estimator = nn.Linear(self.embryo_tot_element, latent_dim)
        self.embryo_image_generator = nn.Linear(latent_dim, self.embryo_tot_element)
        self.act = act_func()
        self.image_loss_func = image_loss_func()
        self.decoder_layer = nn.Sequential(
            *[UpResBlock(self.in_channels // 2 ** i) for i in range(3)]
        )
        self.channel_aligner = nn.Sequential(
            nn.GroupNorm(8, self.in_channels // 2 ** 3),
            nn.ReLU(),
            nn.Conv2d(self.in_channels // 2 ** 3, raw_image_shape[0], 1, 1)
        )

    def forward(self, encoded_x: torch.Tensor | monai.data.MetaTensor, raw_image: torch.Tensor | monai.data.MetaTensor):
        """
            Given an image embedding after passthrough ViT and an original image before ViT forward.
        @param encoded_x:
        @param raw_image:
        @return:
        """
        ic(encoded_x.shape)
        if len(encoded_x.shape) == 3:
            encoded_x = encoded_x.unsqueeze(0)
        encoded_x = self.down_sample(encoded_x)
        # print(f'encoded_x.shape: {encoded_x.shape}')
        restore_shape = encoded_x.shape
        flat_x = encoded_x.view(-1, self.mean_estimator.in_features)
        mean = self.mean_estimator(flat_x)
        noise = torch.randn_like(mean, requires_grad=False)

        if self.estimate_std:
            std = self.std_estimator(flat_x)
            std = F.softplus(std)
            reg_loss = .5 * torch.mean(mean ** 2 + std ** 2 - torch.log(1e-8 + std ** 2) - 1)

        else:
            std = self.std_estimator
            reg_loss = torch.mean(mean ** 2)

        flat_z = mean + std * noise
        flat_z = self.act(self.embryo_image_generator(flat_z))
        flat_z = flat_z.view(restore_shape)
        flat_z = self.prefix_up_sample(flat_z)
        flat_z = self.decoder_layer(flat_z)
        pseudo_image = self.channel_aligner(flat_z)
        # ic(pseudo_image.shape)
        loss_image = self.image_loss_func(raw_image, pseudo_image)
        loss_tot = reg_loss + loss_image
        return loss_tot





if __name__ == '__main__':
    image_shape = [27, 256, 256]
    dx = torch.randn(image_shape)[None, ...]
    embed = torch.randn((1, 256, 16, 16))
    model = VAEDecoder(
        256, image_shape, 'vit-l'
    )
    print(model)
    loss = model(embed, dx)
