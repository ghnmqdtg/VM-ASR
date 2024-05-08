from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Newer versions of PyTorch
    from torch.nn.utils.parametrizations import weight_norm, spectral_norm
except:
    # Older versions of PyTorch
    from torch.nn.utils import weight_norm
    from torch.nn.utils.parametrizations import spectral_norm


from fvcore.nn import flop_count, parameter_count

from torchinfo import summary


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
        y_real = []  # Discriminator outputs for real audio
        y_gen = []  # Discriminator outputs for generated audio
        feature_map_real = []  # Feature maps for real audio
        feature_map_gen = []  # Feature maps for generated audio
        for i, disc in enumerate(self.discriminators):
            y_r, feature_map_r = disc(y)
            y_g, feature_map_g = disc(y_hat)
            y_real.append(y_r)
            feature_map_real.append(feature_map_r)
            y_gen.append(y_g)
            feature_map_gen.append(feature_map_g)

        return y_real, y_gen, feature_map_real, feature_map_gen

    @torch.no_grad()
    def flops(self, shape=(1, 122640)):
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


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, hidden=128):
        super(ScaleDiscriminator, self).__init__()
        self.norm_layer = weight_norm if use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                self.norm_layer(
                    nn.Conv1d(
                        1,
                        hidden,
                        kernel_size=15,
                        stride=1,
                        padding=7,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel_size=41,
                        stride=4,
                        groups=4,
                        padding=20,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden,
                        hidden * 2,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden * 2,
                        hidden * 4,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden * 4,
                        hidden * 8,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden * 8,
                        hidden * 8,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                self.norm_layer(
                    nn.Conv1d(
                        hidden * 8,
                        hidden * 8,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    )
                ),
            ]
        )

        self.conv_post = self.norm_layer(
            nn.Conv1d(
                hidden * 8,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x):
        feature_map = []
        for layer in self.convs:
            x = layer(x)
            x = nn.GELU()(x)
            feature_map.append(x)
        x = self.conv_post(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self, hidden=128):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(hidden=hidden),
                ScaleDiscriminator(hidden=hidden),
                ScaleDiscriminator(hidden=hidden),
            ]
        )
        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, y, y_hat):
        # y: real audio, y_hat: generated audio
        y_real = []  # Discriminator outputs for real audio
        y_gen = []  # Discriminator outputs for generated audio
        feature_map_real = []  # Feature maps for real audio
        feature_map_gen = []  # Feature maps for generated audio
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_r, feature_map_r = disc(y)
            y_g, feature_map_g = disc(y_hat)
            y_real.append(y_r)
            feature_map_real.append(feature_map_r)
            y_gen.append(y_g)
            feature_map_gen.append(feature_map_g)

        return y_real, y_gen, feature_map_real, feature_map_gen

    @torch.no_grad()
    def flops(self, shape=(1, 122640)):
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


if __name__ == "__main__":
    # Test the discriminator
    model = MultiPeriodDiscriminator(hidden=32).to("cuda")
    print(model.flops())

    del model

    model = MultiScaleDiscriminator(hidden=128).to("cuda")
    print(model.flops())
