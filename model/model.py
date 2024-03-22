import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

try:
    from base import BaseModel
    from .vmamba import VSSM, VSSBlock
except:
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Now you can import BaseModel
    from base.base_model import BaseModel
    from vmamba import VSSM, VSSBlock


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


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm2d is a wrapper for LayerNorm to support 2D input.
    """

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    """
    Wrapper for torch.permute. Used in nn.Sequential.
    """

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False
        )
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """
    Patch expansion module.

    SRC: Mamba-UNet
    """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class MambaUNet(BaseModel):
    """
    Mamba-based U-Net model. Inherits from BaseModel and VSSM.
    """

    def __init__(
        self,
        patch_size=4,
        in_chans=1,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # ==============================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",  # "BN", "LN2D"
        patchembed_version: str = "ch1",  # "v1", "v2"
        patchembed_reverse_version: str = "v1",  # "v1"
        downsample_version: str = "v1",  # "v1", "v2", "v3"
        upsample_version: str = "v1",  # "v1"
        use_checkpoint=False,
        **kwargs,
    ):
        # Initialize the BaseModel and VSSM
        super().__init__()
        # Default norm layer is LayerNorm, can be changed to BatchNorm or LayerNorm2D
        self.channel_first = norm_layer.lower() in ["bn", "ln2d"]
        self.num_layers = len(depths)
        if isinstance(dims, int):
            # If dims is an integer, use it as the base dimension for all layers
            # and scale it by 2^i for the i-th layer
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule (dpr = drop path rate)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # norm_layer originally is a string, we use it to get the layer as a class
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        # Select the patch embedding module
        _make_patch_embed = dict(
            # v1=self._make_patch_embed,
            # v2=self._make_patch_embed_v2,
            ch1=self._make_patch_embed_ch1,
        ).get(patchembed_version, None)

        _make_patch_embed_reverse = dict(
            v1=self._male_patch_embed_reverse_ch1,
        ).get(patchembed_reverse_version, None)

        # Pass parameters to chosen patch embed
        self.patch_embed = _make_patch_embed(
            in_chans,
            dims[0],
            patch_size,
            patch_norm,
            norm_layer,
            channel_first=self.channel_first,
        )

        print(f"patchembed_version: {patchembed_version} -> {_make_patch_embed}")

        # Get the downsample version
        _make_downsample = dict(
            v1=PatchMerging2D,
            # v2=self._make_downsample,
            # v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        print(f"downsample_version: {downsample_version} -> {_make_downsample}")

        # Get the upsample version
        _make_upsample = dict(
            v1=PatchExpand,
        ).get(upsample_version, None)

        self.layers_encoder = nn.ModuleList()
        self.layers_latent = nn.ModuleList()
        self.layers_decoder = nn.ModuleList()
        # self.num_layers is "stage" in the VMamba paper
        # Here we have 4 stages, each stage has a different number of input and ouptut dimensions
        # Encoders
        for i_layer in range(self.num_layers):
            print(f"Downsample layer {i_layer}")
            print(self.dims[i_layer])
            downsample = (
                _make_downsample(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer,
                    channel_first=self.channel_first,
                )
                if (i_layer < self.num_layers - 1)
                # The last layer does not need downsample
                else nn.Identity()
            )
            self.layers_encoder.append(
                self.VSSLayer(
                    dim=self.dims[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    sampler=downsample,
                    channel_first=self.channel_first,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                )
            )
        # Latent layer
        self.layers_latent.append(
            self.VSSLayer(
                dim=self.dims[self.num_layers - 1],
                drop_path=dpr[
                    sum(depths[: self.num_layers - 1]) : sum(depths[: self.num_layers])
                ],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                sampler=nn.Identity(),
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )
        )
        # Decoders
        for i_layer in range(self.num_layers - 1, -1, -1):
            print(f"Upsample layer {i_layer}")
            print(
                self.dims[i_layer + 1]
                if i_layer < self.num_layers - 1
                else self.dims[self.num_layers - 1]
            )
            upsample = (
                _make_upsample(
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer,
                )
                if (i_layer < self.num_layers - 1)
                else nn.Identity()
            )
            self.layers_decoder.append(
                self.VSSLayer(
                    dim=(
                        self.dims[i_layer + 1]
                        if i_layer < self.num_layers - 1
                        else self.dims[self.num_layers - 1]
                    ),
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    sampler=upsample,
                    channel_first=self.channel_first,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                )
            )

        self.patch_embed_reverse = _make_patch_embed_reverse(
            in_chans,
            dims[0],
            patch_size,
            patch_norm,
            norm_layer,
            channel_first=self.channel_first,
        )

    def forward(self, x):
        x = x[:, 0, :, :, :]
        print(f"Input shape: {x.shape}")
        # Clone the input for residual connection
        residual = x.clone()
        # Patch embedding
        x = self.patch_embed(x)
        # Skip connections
        skip_connections = []
        print(f"Patch embedding shape: {x.shape}")
        # Encoder
        for i, layer in enumerate(self.layers_encoder):
            x = layer(x)
            print(f"Encoder({i}): {x.shape}")
            # skip_connections.append(x)
        # Latent layer
        for i, layer in enumerate(self.layers_latent):
            x = layer(x)
        print(f"Latent({i}): {x.shape}")
        # print(len(skip_connections))
        # Decoder
        for i, layer in enumerate(self.layers_decoder):
            x = layer(x)
            print(f"Decoder({i}): {x.shape}")
            # Concatenate the skip connection
            # x = torch.cat((x, skip_connections.pop()), dim=-1)
        x = self.patch_embed_reverse(x)
        print(f"Output shape: {x.shape}")
        print(x + residual)
        return x + residual, x

    @staticmethod
    # Static method does not recieve the instance as the first argument (self)
    def _make_patch_embed_ch1(
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        channel_first=False,
    ):
        # VMamba set the patch size to 4, we follow this convention
        assert patch_size == 4
        # If channel_first is True, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            # Permute the dimensions if channel_first is False and patch_norm is True
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 2, 3, 1)
            ),
            norm_layer(embed_dim // 2) if patch_norm else nn.Identity(),
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 3, 1, 2)
            ),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            # Permute the dimensions if channel_first is False and patch_norm is True
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _male_patch_embed_reverse_ch1(
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        channel_first=False,
    ):
        # If channel_first is True, then Norm and Output are both channel_first
        return nn.Sequential(
            # Permute the dimensions if channel_first is False and patch_norm is True
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 3, 1, 2)
            ),
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # Permute the dimensions if channel_first is False and patch_norm is True
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 2, 3, 1)
            ),
            norm_layer(embed_dim // 2) if patch_norm else nn.Identity(),
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 3, 1, 2)
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                embed_dim // 2,
                in_chans,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # Permute the dimensions if channel_first is False and patch_norm is True
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(in_chans) if patch_norm else nn.Identity()),
            # Permute the dimensions if channel_first is False and patch_norm is True
            (
                nn.Identity()
                if (channel_first and (not patch_norm))
                else Permute(0, 3, 1, 2)
            ),
        )

    @staticmethod
    def VSSLayer(
        dim=96,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        sampler=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                )
            )

        return nn.Sequential(
            OrderedDict(
                blocks=nn.Sequential(
                    *blocks,
                ),
                sampler=sampler,
            )
        )


if __name__ == "__main__":
    x = torch.rand(8, 1, 512, 128).to("cuda")
    # x = torch.rand(24, 3, 224, 224).to("cuda")

    model = MambaUNet(
        depths=[2, 2, 2, 2],
        dims=[2, 8, 16, 32],
        in_chans=x.shape[1],
    ).to("cuda")
    # model = VSSM(in_chans=1).to("cuda")
    print(model)

    for i in range(1):
        y = model(x)
