import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba import VSSBlock

try:
    from base import BaseModel
except:
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Now you can import BaseModel
    from base.base_model import BaseModel


class DualStreamBlock(BaseModel):
    """
    Base class for the dual-stream model.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Stream for magnitude
        self.mag_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Stream for phase
        self.phase_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Set activation function
        self.activation = nn.ReLU()

    def forward(self, mag, phase):
        # Stream for magnitude
        mag = self.activation(self.mag_conv(mag))
        # Stream for phase
        phase = self.activation(self.phase_conv(phase))

        # Interaction between streams
        combined_mag = mag + phase
        combined_phase = phase + mag

        return combined_mag, combined_phase


class DualStreamModel(BaseModel):
    """
    Dual-Stream model for learning the magnitude and phase of the image.
    """

    def __init__(self, in_channels=1, out_channels=1, num_blocks=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.out_channels = out_channels

        # Define the dual-stream blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = DualStreamBlock(in_channels, out_channels)
            self.blocks.append(block)
        # Set activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Clone the input for residual connection
        residual = x.clone()
        # Input dimension: (batch_size, mag_(0)or_phase(1), in_channels, H, W)
        # Get the magnitude and phase
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]
        # Forward pass through the dual-stream blocks
        for block in self.blocks:
            mag, phase = block(mag, phase)
        # Apply LayerNorm and activation function for the output
        mag = self.activation(mag)
        phase = self.activation(phase)
        # Residual connection
        mag = mag + residual[:, 0, :, :, :]
        phase = phase + residual[:, 1, :, :, :]

        return mag, phase


class ToyUNet(BaseModel):
    """
    A simple toy U-Net model that has same shape of input and output.
    """

    def __init__(
        self, in_channels=1, out_channels=1, scale=2, dropout=0.0, batchnorm=False
    ):
        super().__init__()
        # Define the channel sizes and kernel size for each layer
        channels_down = [in_channels, 32, 64, 128, 128]
        channels_up = [128, 128, 64, 32, out_channels]

        # Activation function
        self.activation = nn.ReLU()

        # Define the downscaling network
        self.down_net = nn.ModuleList()
        for i in range(len(channels_down) - 1):
            down_block = nn.Sequential()
            conv = nn.Conv2d(
                channels_down[i],
                channels_down[i + 1],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            )
            down_block.append(conv)
            down_block.append(self.activation)
            if batchnorm:
                down_block.append(nn.BatchNorm2d(channels_down[i + 1]))
            self.down_net.append(nn.Sequential(*down_block))

        # Define the bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                channels_down[-1],
                channels_down[-1],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels_up[0],
                channels_up[0] * scale**2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=scale),
        )

        # Define the upscaling network with PixelShuffle and skip connections
        self.up_net = nn.ModuleList()
        for i in range(len(channels_up) - 1):
            up_block = nn.Sequential()
            conv = nn.Conv2d(
                channels_up[i] * scale,
                channels_up[i + 1] * scale**2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            )
            up_block.append(conv)
            up_block.append(self.activation)
            up_block.append(nn.PixelShuffle(upscale_factor=scale))
            if batchnorm:
                up_block.append(nn.BatchNorm2d(channels_up[i + 1]))
            self.up_net.append(nn.Sequential(*up_block))

    def forward(self, x):
        # Residual connection
        y = x.clone()
        # Downscaling path
        down_outputs = []
        for down_block in self.down_net:
            x = down_block(x)
            # print(x.shape)
            down_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        # print(x.shape)

        # Upscaling path
        for up_block in self.up_net:
            # Concatenate the skip connection
            x = torch.cat((x, down_outputs.pop()), dim=1)
            x = up_block(x)
            # print(x.shape)

        # Add the residual connection
        x += y

        return x


class DualStreamUNet(BaseModel):
    """
    Dual-Stream model for learning the magnitude and phase of the image.
    """

    def __init__(
        self, in_channels=1, out_channels=1, scale=2, dropout=0.0, batchnorm=False
    ):
        super().__init__()
        # Define magnitude and phase networks
        self.mag_net = ToyUNet(in_channels, out_channels, scale, dropout, batchnorm)
        self.phase_net = ToyUNet(in_channels, out_channels, scale, dropout, batchnorm)

    def forward(self, x):
        # Clone the input for residual connection
        residual = x.clone()
        # Input dimension: (batch_size, mag_(0)or_phase(1), in_channels, H, W)
        # Get the magnitude and phase
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]
        # Forward pass through the dual-stream blocks
        mag = self.mag_net(mag)
        phase = self.phase_net(phase)
        # Residual connection
        mag = mag + residual[:, 0, :, :, :]
        phase = phase + residual[:, 1, :, :, :]

        return mag, phase


class InteractingUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_down):
        super().__init__()
        padding_dict = {3: 1}
        # Basic convolutional block with option for downscaling or upscaling
        self.conv_mag = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_dict[kernel_size],
            stride=2 if is_down else 1,
        )
        self.conv_phase = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_dict[kernel_size],
            stride=2 if is_down else 1,
        )
        self.activation = nn.ReLU()
        self.up_sample_mag = (
            nn.PixelShuffle(upscale_factor=2) if not is_down else nn.Identity()
        )
        self.up_sample_phase = (
            nn.PixelShuffle(upscale_factor=2) if not is_down else nn.Identity()
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, mag, phase):
        # Pass magnitude through the block
        mag = self.conv_mag(mag)
        mag = self.activation(mag)

        # Pass phase through the block
        phase = self.conv_phase(phase)
        phase = self.activation(phase)

        # Batchnorm
        mag = self.batchnorm(mag)
        phase = self.batchnorm(phase)

        # Upsample if this is an upsampling block
        mag = self.up_sample_mag(mag)
        phase = self.up_sample_phase(phase)

        # Interact magnitude and phase by adding them
        return mag + phase, phase + mag


class DualStreamInteractiveUNet(BaseModel):
    """
    Dual-Stream model for learning the magnitude and phase of the image.
    """

    def __init__(
        self, in_channels=1, out_channels=1, scale=2, dropout=0.0, batchnorm=False
    ):
        super().__init__()
        # Define down and up blocks
        channels_down = [in_channels, 32, 64, 128, 256, 256]
        channels_up = [256, 256, 128, 64, 32, out_channels]
        kernels_down = [3, 3, 3, 3, 3]
        # The reverse of the downscaling kernels
        kernels_up = kernels_down[::-1]
        # Activation function
        self.activation = nn.ReLU()

        # Define the downscaling network
        self.down_unet_blocks = nn.ModuleList()
        for i in range(len(channels_down) - 1):
            down_block = InteractingUNetBlock(
                channels_down[i], channels_down[i + 1], kernels_down[i], is_down=True
            )
            self.down_unet_blocks.append(down_block)

        # Define the bottleneck
        self.bottleneck_mag = nn.Sequential(
            nn.Conv2d(
                channels_down[-1],
                channels_down[-1],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            self.activation,
            nn.Conv2d(
                channels_up[0],
                channels_up[0] * scale**2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            self.activation,
            nn.PixelShuffle(upscale_factor=scale),
        )

        self.bottleneck_phase = nn.Sequential(
            nn.Conv2d(
                channels_down[-1],
                channels_down[-1],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            self.activation,
            nn.Conv2d(
                channels_up[0],
                channels_up[0] * scale**2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            self.activation,
            nn.PixelShuffle(upscale_factor=scale),
        )

        # Define the upscaling network with PixelShuffle and skip connections
        self.up_unet_blocks = nn.ModuleList()
        for i in range(len(channels_up) - 1):
            up_block = InteractingUNetBlock(
                channels_up[i] * scale,
                channels_up[i + 1] * scale**2,
                kernels_up[i],
                is_down=False,
            )
            self.up_unet_blocks.append(up_block)

    def forward(self, x):
        # Clone the input for residual connection
        residual = x.clone()
        # Input dimension: (batch_size, mag_(0)or_phase(1), in_channels, H, W)
        # Get the magnitude and phase
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]
        # Skip connections
        skip_connections = []
        # Forward pass through the dual-stream blocks
        for block in self.down_unet_blocks:
            mag, phase = block(mag, phase)
            skip_connections.append((mag, phase))
        # Bottleneck
        mag = self.bottleneck_mag(mag)
        phase = self.bottleneck_phase(phase)
        # Forward pass through the dual-stream blocks
        for block in self.up_unet_blocks:
            # Add the skip connection
            mag_skip, phase_skip = skip_connections.pop()
            # Concatenate the skip connection
            mag = torch.cat((mag, mag_skip), dim=1)
            phase = torch.cat((phase, phase_skip), dim=1)
            mag, phase = block(mag, phase)

        # Residual connection
        mag = mag + residual[:, 0, :, :, :]
        phase = phase + residual[:, 1, :, :, :]

        return mag, phase


class InteractingVSSBlock(nn.Module):

    def __init__(self, hidden_dim=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vss_layers_mag = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            ssm_d_state=1,
            ssm_ratio=2,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_checkpoint=False,
        )
        self.vss_layers_phase = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            ssm_d_state=1,
            ssm_ratio=2,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_checkpoint=False,
        )

    def forward(self, mag, phase):
        # Pass magnitude through the block
        mag = self.vss_layers_mag(mag)
        # Pass phase through the block
        phase = self.vss_layers_phase(phase)
        return mag + phase, phase + mag


class DualStreamInteractiveVSS(BaseModel):
    """
    Integrating VSSBlock
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem_mag = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.stem_phase = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.vss_layers = nn.ModuleList()
        for i in range(3):
            self.vss_layers.append(InteractingVSSBlock(hidden_dim=3))
        self.tail_mag = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.tail_phase = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Clone the input for residual connection
        residual = x.clone()
        # Input dimension: (batch_size, mag_(0)or_phase(1), in_channels, H, W)
        # Get the magnitude and phase
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]

        # Forward pass through the dual-stream blocks
        mag = self.stem_mag(mag).permute(0, 2, 3, 1)
        phase = self.stem_phase(phase).permute(0, 2, 3, 1)

        for block in self.vss_layers:
            mag, phase = block(mag, phase)

        mag = self.tail_mag(mag.permute(0, 3, 1, 2))
        phase = self.tail_phase(phase.permute(0, 3, 1, 2))

        # Residual connection
        mag = mag + residual[:, 0, :, :, :]
        phase = phase + residual[:, 1, :, :, :]

        return mag, phase


if __name__ == "__main__":
    model = ToyUNet().to("cuda")
    print(model)
    x = torch.rand(128, 1, 512, 128).to("cuda")
    y = model(x)
    print(y.shape)
