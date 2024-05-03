from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import spectral_norm

from fvcore.nn import flop_count, parameter_count

from torchinfo import summary


def feature_loss(fmap_r, fmap_g):
    loss = 0
    total_n_layers = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            total_n_layers += 1
            loss += torch.mean(torch.abs(rl - gl))

    return loss / total_n_layers


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss

    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l

    return loss


class PeriodDiscriminator(torch.nn.Module):
    """
    A discriminator module that operates at a specific period.

    Args:
        period (int): The period at which the discriminator operates.
        kernel_size (int): The size of the kernel for convolutional layers.
        stride (int): The stride for convolutional layers.
        use_spectral_norm (bool): Whether to use spectral normalization on the layers.
    """

    def __init__(
        self, period, kernel_size=5, stride=3, use_spectral_norm=False, hidden=32
    ):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.norm_layer = weight_norm if use_spectral_norm else spectral_norm

        # The discriminator is composed of a series of convolutions
        # The number of channels increases as the audio signal moves through the layers
        self.layers = nn.ModuleList(
            [
                self.norm_layer(
                    nn.Conv2d(
                        1,
                        hidden,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        hidden,
                        hidden * 4,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        hidden * 4,
                        hidden * 16,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        hidden * 16,
                        hidden * 32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        hidden * 32, hidden * 32, (kernel_size, 1), 1, padding=(2, 0)
                    )
                ),
            ]
        )
        self.conv_post = self.norm_layer(
            nn.Conv2d(hidden * 32, 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x):
        feature_map = []

        # Convert 1D audio signal to 2D by segmenting with respect to the period
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.layers:
            x = layer(x)
            x = nn.GELU()(x)
            # Store the feature map after each layer
            feature_map.append(x)

        x = self.conv_post(x)

        feature_map.append(x)

        # Flatten the output for the final discriminator score
        # Which means that the discriminator will output a single score for the input
        x = torch.flatten(x, 1, -1)

        return x, feature_map

    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, hidden=32, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period, hidden=hidden) for period in periods]
        )

    def forward(self, y, y_hat):
        # y: real audio, y_hat: generated audio
        y_d_rs = []  # Discriminator outputs for real audio
        y_d_gs = []  # Discriminator outputs for generated audio
        fmap_rs = []  # Feature maps for real audio
        fmap_gs = []  # Feature maps for generated audio
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    @torch.no_grad()
    def flops(self, shape=(1, 121890)):
        model = deepcopy(self)
        model.cuda().eval()
        supported_ops = {
            "aten::gelu": None,  # as relu is in _IGNORED_OPS
        }

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_hat = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(
            model=model, inputs=(input, input_hat), supported_ops=supported_ops
        )

        statics = summary(model, input_data=[input, input_hat], verbose=0)
        del model, input, input_hat
        torch.cuda.empty_cache()

        # Return the number of parameters and FLOPs
        return (
            f"{statics}\nparams {params/1e6:.2f}M, GFLOPs {sum(Gflops.values()):.2f}\n"
        )
