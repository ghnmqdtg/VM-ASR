import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import spectral_norm
from fvcore.nn import flop_count, parameter_count
from torchinfo import summary

try:
    from base import BaseModel
    from .vmamba import VSSBlock, selective_scan_flop_jit
except:
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Now you can import BaseModel
    from base.base_model import BaseModel
    from vmamba import VSSBlock, selective_scan_flop_jit


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
    """
    Patch merging module.

    input: (B, H, W, C) -> output: (B, H/2, W/2, 4*C)
    """

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


class PatchExpanding(nn.Module):
    """
    Patch expansion module that doubles the spatial dimensions and halves the channel count.

    input: (B, H, W, C) -> output: (B, 2H, 2W, C/2)
    """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        if norm_layer is not None:
            self.norm = norm_layer(dim // dim_scale)
        else:
            self.norm = nn.Identity()

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
        patchembed_version: str = "v2",  # "v1", "v2"
        downsample_version: str = "v1",  # "v1", "v2", "v3"
        upsample_version: str = "v1",  # "v1"
        output_version: str = "v2",  # "v1", "v2", "v3"
        concat_skip=False,
        use_checkpoint=False,
        **kwargs,
    ):
        # Initialize the BaseModel and VSSM
        super().__init__()
        # Default norm layer is LayerNorm, can be changed to BatchNorm or LayerNorm2D
        self.channel_first = norm_layer.lower() in ["bn", "ln2d"]
        self.num_layers = len(depths)
        self.depths = depths
        if isinstance(dims, int):
            # If dims is an integer, use it as the base dimension for all layers
            # and scale it by 2^i for the i-th layer
            dims = [int(dims * 2**i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule (dpr = drop path rate)
        self.concat_skip = concat_skip

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
            v1=self._make_patch_embed_v1,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)

        _make_output_layer = dict(
            v1=self._make_output_layer_v1,
            v2=self._make_output_layer_v2,
            v3=self._make_output_layer_v3,
        ).get(output_version, None)

        # Pass parameters to chosen patch embed
        self.patch_embed = _make_patch_embed(
            in_chans,
            dims[0],
            patch_size,
            patch_norm,
            norm_layer,
            channel_first=self.channel_first,
        )

        # print(f"patchembed_version: {patchembed_version} -> {_make_patch_embed}")

        # Get the downsample version
        _make_downsample = dict(
            v1=PatchMerging2D,
        ).get(downsample_version, None)

        # print(f"downsample_version: {downsample_version} -> {_make_downsample}")

        # Get the upsample version
        _make_upsample = dict(
            v1=PatchExpanding,
        ).get(upsample_version, None)

        self.layers_encoder = nn.ModuleList()
        self.layers_latent = nn.ModuleList()
        self.layers_decoder = nn.ModuleList()
        # self.num_layers is "stage" in the VMamba paper

        # Encoders
        for i_layer in range(self.num_layers):
            if len(self.dims) == 5:
                downsample = _make_downsample(
                    # Here we have 4 stages, each stage has a different number of input and ouptut dimensions
                    # The original VMamba code didn't set the downsampling for the last stage
                    # So we add extra dim for the last stage
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer,
                    channel_first=self.channel_first,
                )
            else:
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
                    drop_path=self.dpr[
                        sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                    ],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    sampler=downsample,
                    channel_first=self.channel_first,
                    concat_skip=False,
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
        if len(self.dims) == 5:
            self.layers_latent.append(
                self.VSSLayer(
                    dim=(
                        self.dims[self.num_layers]
                        if len(self.dims) == 5
                        else self.dims[self.num_layers - 1]
                    ),
                    drop_path=self.dpr[
                        sum(self.depths[: self.num_layers - 1]) : sum(
                            self.depths[: self.num_layers]
                        )
                    ],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    sampler=nn.Identity(),
                    channel_first=self.channel_first,
                    concat_skip=False,
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
        # num_layers is 4, so we iterate from 3 to 0 intuitively
        # But we iterate from 4 to 1 because we add the downsample layer to the output of the last encoder layer
        # Therefore, there is an extra dim for the last stage (we set it in config), so we iterate from 4 to 1
        # The inversed dim would be 128 -> 64 -> 32 -> 16 ->8, The 8 is not used as input dim here. It's the output dim for the last decoder
        for i_layer in range(self.num_layers, 0, -1):
            if len(self.dims) == 5:
                upsample = _make_upsample(
                    self.dims[i_layer],
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                upsample = (
                    _make_upsample(
                        self.dims[i_layer],
                        dim_scale=2,
                        norm_layer=norm_layer,
                    )
                    if i_layer < self.num_layers
                    else nn.Identity()
                )

            self.layers_decoder.append(
                self.VSSLayer(
                    dim=(
                        self.dims[i_layer]
                        if len(self.dims) == 5
                        else (
                            self.dims[i_layer]
                            if i_layer < self.num_layers - 1
                            else self.dims[self.num_layers - 1]
                        )
                    ),
                    drop_path=self.dpr[
                        sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                    ],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    sampler=upsample,
                    channel_first=self.channel_first,
                    concat_skip=(
                        self.concat_skip
                        if len(self.dims) == 5
                        else (self.concat_skip if i_layer < self.num_layers else False)
                    ),
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

        self.output_layer = _make_output_layer(
            in_chans=in_chans,
            dim=dims[0],
            norm_layer=None,
            sampler=_make_upsample,
            use_checkpoint=use_checkpoint,
            concat_skip=self.concat_skip,
            channel_first=self.channel_first,
            # =================
            # Used for output versions that require VSSLayer
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
            # Used for output versions that require VSSLayer
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
        )

        self.apply(self._init_weights)

    def forward(self, x):
        verbose = True
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]
        # Clone the input for residual connection
        if verbose:
            print(f"Input shape: {mag.shape}")
        residual_mag = mag.clone()
        # Skip connections
        skip_connections = []
        # Patch embedding
        mag = self.patch_embed(mag)
        skip_connections.append(mag)
        if verbose:
            print(f"Patch embedding shape: {mag.shape}")
        if len(self.dims) == 5:
            for i, layer in enumerate(self.layers_encoder):
                mag = layer(mag)
                skip_connections.append(mag)
                if verbose:
                    print(f"Encoder layer {i} shape: {mag.shape}")
            # Latent layer
            for i, layer in enumerate(self.layers_latent):
                mag = layer(mag)
                if verbose:
                    print(f"Latent layer {i} shape: {mag.shape}")
            if verbose:
                # Print shape of each item in skip_connections
                for i, skip in enumerate(skip_connections):
                    print(f"Skip connection {i} shape: {skip.shape}")
            # Decoder
            for i, layer in enumerate(self.layers_decoder):
                # Concatenate or add skip connection
                mag = (
                    torch.cat((mag, skip_connections.pop()), dim=-1)
                    if self.concat_skip
                    else (mag + skip_connections.pop())
                )
                mag = layer(mag)
                if verbose:
                    print(f"Decoder layer {i} shape: {mag.shape}")

            # Concatenate or add skip connection
            mag = (
                torch.cat((mag, skip_connections.pop()), dim=-1)
                if self.concat_skip
                else (mag + skip_connections.pop())
            )
            if verbose:
                print(f"Output layer input shape: {mag.shape}")

            # Output layer
            mag = self.output_layer(mag)

            if verbose:
                print(f"Patch output shape: {mag.shape}")
        else:
            # Encoder
            for i, layer in enumerate(self.layers_encoder):
                mag = layer(mag)
                if i < self.num_layers - 1:
                    skip_connections.append(mag)
                if verbose:
                    print(f"Encoder layer {i} shape: {mag.shape}")
            if verbose:
                # Print shape of each item in skip_connections
                for i, skip in enumerate(skip_connections):
                    print(f"Skip connection {i} shape: {skip.shape}")
            # Decoder
            for i, layer in enumerate(self.layers_decoder):
                # Concatenate or add skip connection
                if i != 0:
                    mag = (
                        torch.cat((mag, skip_connections.pop()), dim=-1)
                        if self.concat_skip
                        else (mag + skip_connections.pop())
                    )

                mag = layer(mag)
                if verbose:
                    print(f"Decoder layer {i} shape: {mag.shape}")

            # Concatenate or add skip connection
            mag = (
                torch.cat((mag, skip_connections.pop()), dim=-1)
                if self.concat_skip
                else (mag + skip_connections.pop())
            )
            if verbose:
                print(f"Output layer input shape: {mag.shape}")

            # Output layer
            mag = self.output_layer(mag)

            if verbose:
                print(f"Patch output shape: {mag.shape}")

        return mag + residual_mag, phase

    @staticmethod
    def _make_patch_embed_v1(
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        channel_first=False,
    ):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            ),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    # Static method does not recieve the instance as the first argument (self)
    def _make_patch_embed_v2(
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
    def _make_output_layer_v1(
        in_chans=3,
        dim=96,
        concat_skip=False,
        **kwargs,
    ):
        """
        Output layer v1: Using two ConvTranspose2d for upscaling.

        The first ConvTranspose2d decreases input channel size by half, and the second decreases it to the input channel size.
        """
        # If channel_first is True, then Norm and Output are both channel_first
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            (
                nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
                if concat_skip
                else nn.Identity()
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                dim,
                dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                dim // 2,
                in_chans,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def _make_output_layer_v2(
        self,
        in_chans=3,
        dim=96,
        sampler=nn.Identity(),
        concat_skip=False,
        use_checkpoint=False,
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
        """
        Output layer v2: Using VSSLayer + PatchExpanding for upscaling.
        """

        return nn.Sequential(
            self.VSSLayer(
                dim=dim,
                drop_path=self.dpr[
                    sum(self.depths[: self.num_layers - 1]) : sum(
                        self.depths[: self.num_layers]
                    )
                ],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.LayerNorm,
                sampler=sampler(
                    dim,
                    dim_scale=2,
                    norm_layer=None,
                ),
                channel_first=self.channel_first,
                concat_skip=concat_skip,
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
            ),
            self.VSSLayer(
                dim=dim // 2,
                drop_path=self.dpr[
                    sum(self.depths[: self.num_layers - 1]) : sum(
                        self.depths[: self.num_layers]
                    )
                ],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.LayerNorm,
                sampler=sampler(
                    dim // 2,
                    dim_scale=2,
                    norm_layer=None,
                ),
                channel_first=self.channel_first,
                concat_skip=False,
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
            ),
            Permute(0, 3, 1, 2),
            (
                nn.Conv2d(dim // 4, in_chans, kernel_size=1, stride=1, padding=0)
                if dim // 4 != in_chans
                else nn.Identity()
            ),
        )

    def _make_output_layer_v3(
        self,
        in_chans=1,
        dim=96,
        use_checkpoint=False,
        sampler=nn.Identity(),
        concat_skip=True,
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
        """
        Output layer v3: Using Conv2d (skip connection) + PatchExpanding + VSSLayer for upscaling.
        """

        return nn.Sequential(
            self.VSSLayer(
                dim=dim,
                drop_path=self.dpr[-1:],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.Identity,
                # Input: (B, H, W, C) -> Output: (B, 2 * H, 2 * W, C // 2)
                sampler=sampler(
                    dim,
                    dim_scale=2,
                    norm_layer=nn.LayerNorm,
                ),
                channel_first=self.channel_first,
                concat_skip=concat_skip,
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
            ),
            # Refine the output
            self.VSSLayer(
                dim=dim // 2,
                drop_path=self.dpr[-1:],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.LayerNorm,
                # Input: (B, 2 * H, 2 * W, C // 2) -> Output: (B, 4 * H, 4 * W, C // 4)
                sampler=sampler(
                    dim // 2,
                    dim_scale=2,
                    norm_layer=nn.LayerNorm,
                ),
                channel_first=self.channel_first,
                concat_skip=False,  # We already handle the skip connection above
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
            ),
            # Input: (B, 4 * H, 4 * W, C // 4) -> Output: (B, 4 * H, 4 * W, C)
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim // 4, in_chans, kernel_size=1, stride=1, padding=0),
            Permute(0, 2, 3, 1),
            # Refine the output
            self.VSSLayer(
                dim=in_chans,
                drop_path=self.dpr[-1:],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.Identity,
                sampler=nn.Identity(),
                channel_first=self.channel_first,
                concat_skip=False,  # We already handle the skip connection above
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
            ),
            Permute(0, 3, 1, 2),
        )

    @staticmethod
    def VSSLayer(
        dim=96,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        sampler=nn.Identity(),
        channel_first=False,
        concat_skip=False,
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
        skip_handler = nn.Identity()
        if concat_skip:
            # If concat_skip is True, then then input dimension is doubled
            # We use a Conv2d layer to reduce the dimension back to the original
            skip_handler = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0),
                Permute(0, 2, 3, 1),
            )

        # If channel first, then Norm and Output are both channel_first
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
                [
                    ("skip_handler", skip_handler),
                    ("blocks", nn.Sequential(*blocks)),
                    ("sampler", sampler),
                ]
            )
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def flops(self, shape=(2, 1, 512, 128)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::gelu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(
            model=model, inputs=(input,), supported_ops=supported_ops
        )
        statics = summary(model, input_size=input.shape, verbose=0)
        del model, input
        torch.cuda.empty_cache()

        # Return the number of parameters and FLOPs
        return (
            f"{statics}\nparams {params/1e6:.2f}M, GFLOPs {sum(Gflops.values()):.2f}\n"
        )


class DualStreamInteractiveMambaUNet(MambaUNet):
    """
    InteractiveVSSLayers
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
        patchembed_version: str = "v2",  # "v1", "v2"
        downsample_version: str = "v1",  # "v1", "v2", "v3"
        upsample_version: str = "v1",  # "v1"
        output_version: str = "v2",  # "v1", "v2", "v3"
        concat_skip=False,
        use_checkpoint=False,
        **kwargs,
    ):
        # Initialize the MambaUNet
        super().__init__(
            patch_size,
            in_chans,
            depths,
            dims,
            ssm_d_state,
            ssm_ratio,
            ssm_dt_rank,
            ssm_act_layer,
            ssm_conv,
            ssm_conv_bias,
            ssm_drop_rate,
            ssm_init,
            forward_type,
            mlp_ratio,
            mlp_act_layer,
            mlp_drop_rate,
            gmlp,
            drop_path_rate,
            patch_norm,
            norm_layer,
            patchembed_version,
            downsample_version,
            upsample_version,
            output_version,
            concat_skip,
            use_checkpoint,
            **kwargs,
        )
        # Deep copy patch embedding, encoder, latent, decoder and output in MambaUNet
        # Create magnitude stream
        self.patch_embed_mag = deepcopy(self.patch_embed)
        self.layers_encoder_mag = deepcopy(self.layers_encoder)
        self.layers_latent_mag = (
            deepcopy(self.layers_latent) if len(self.dims) == 5 else None
        )
        self.layers_decoder_mag = deepcopy(self.layers_decoder)
        self.output_layer_mag = deepcopy(self.output_layer)
        # Create phase stream
        self.patch_embed_phase = deepcopy(self.patch_embed)
        self.layers_encoder_phase = deepcopy(self.layers_encoder)
        self.layers_latent_phase = (
            deepcopy(self.layers_latent) if len(self.dims) == 5 else None
        )
        self.layers_decoder_phase = deepcopy(self.layers_decoder)
        self.output_layer_phase = deepcopy(self.output_layer)

        # Delete layers in MambaUNet to save memory
        del self.patch_embed
        del self.layers_encoder
        del self.layers_latent
        del self.layers_decoder
        del self.output_layer

        self.apply(self._init_weights)

    def forward(self, x):
        # Loop through the magnitude and phase streams
        mag = x[:, 0, :, :, :]
        phase = x[:, 1, :, :, :]
        # Clone the input for residual connection
        residual_mag = mag.clone()
        residual_phase = phase.clone()
        # Skip connections
        skip_connections = []
        # Patch embedding
        mag = self.patch_embed_mag(mag)
        phase = self.patch_embed_phase(phase)
        skip_connections.append((mag, phase))
        if len(self.dims) == 5:
            # Encoders (zip is used to iterate over two lists at the same time)
            for i, (encoder_mag, encoder_phase) in enumerate(
                zip(self.layers_encoder_mag, self.layers_encoder_phase)
            ):
                mag = encoder_mag(mag)
                phase = encoder_phase(phase)
                skip_connections.append((mag, phase))
                # Interacting
                mag = mag + phase
                phase = phase + mag
            # Latent layer
            for i, (latent_mag, latent_phase) in enumerate(
                zip(self.layers_latent_mag, self.layers_latent_phase)
            ):
                mag = latent_mag(mag)
                phase = latent_phase(phase)
            # Decoders
            for i, (decoder_mag, decoder_phase) in enumerate(
                zip(self.layers_decoder_mag, self.layers_decoder_phase)
            ):
                # Concatenate or add skip connection
                mag_skip, phase_skip = skip_connections.pop()
                if self.concat_skip:
                    mag = decoder_mag(torch.cat((mag, mag_skip), dim=-1))
                    phase = decoder_mag(torch.cat((phase, phase_skip), dim=-1))
                else:
                    mag = decoder_mag(mag + mag_skip)
                    phase = decoder_phase(phase + phase_skip)

                # Interacting
                mag = mag + phase
                phase = phase + mag

            # Concatenate or add skip connection for the output layer
            mag_skip, phase_skip = skip_connections.pop()
            if self.concat_skip:
                mag = self.output_layer_mag(torch.cat((mag, mag_skip), dim=-1))
                phase = self.output_layer_phase(torch.cat((phase, phase_skip), dim=-1))
            else:
                mag = self.output_layer_mag(mag + mag_skip)
                phase = self.output_layer_phase(phase + phase_skip)
        else:
            # Encoders (zip is used to iterate over two lists at the same time)
            for i, (encoder_mag, encoder_phase) in enumerate(
                zip(self.layers_encoder_mag, self.layers_encoder_phase)
            ):
                mag = encoder_mag(mag)
                phase = encoder_phase(phase)
                if i < self.num_layers - 1:
                    skip_connections.append((mag, phase))
                # Interacting
                mag = mag + phase
                phase = phase + mag

            # Decoders
            for i, (decoder_mag, decoder_phase) in enumerate(
                zip(self.layers_decoder_mag, self.layers_decoder_phase)
            ):
                if i != 0:
                    # Concatenate or add skip connection
                    mag_skip, phase_skip = skip_connections.pop()
                    if self.concat_skip:
                        mag = decoder_mag(torch.cat((mag, mag_skip), dim=-1))
                        phase = decoder_mag(torch.cat((phase, phase_skip), dim=-1))
                    else:
                        mag = decoder_mag(mag + mag_skip)
                        phase = decoder_phase(phase + phase_skip)
                else:
                    mag = decoder_mag(mag)
                    phase = decoder_phase(phase)

                # Interacting
                mag = mag + phase
                phase = phase + mag

            # Concatenate or add skip connection for the output layer
            mag_skip, phase_skip = skip_connections.pop()
            if self.concat_skip:
                mag = self.output_layer_mag(torch.cat((mag, mag_skip), dim=-1))
                phase = self.output_layer_phase(torch.cat((phase, phase_skip), dim=-1))
            else:
                mag = self.output_layer_mag(mag + mag_skip)
                phase = self.output_layer_phase(phase + phase_skip)

        # Residual connection
        return mag + residual_mag, phase + residual_phase

    @torch.no_grad()
    def flops(self, shape=(2, 1, 512, 128)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::gelu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(
            model=model, inputs=(input,), supported_ops=supported_ops
        )
        statics = summary(model, input_size=input.shape, verbose=0)
        del model, input
        torch.cuda.empty_cache()

        # Return the number of parameters and FLOPs
        return (
            f"{statics}\nparams {params/1e6:.2f}M, GFLOPs {sum(Gflops.values()):.2f}\n"
        )


class PeriodDiscriminator(torch.nn.Module):
    """
    A discriminator module that operates at a specific period.

    Args:
        period (int): The period at which the discriminator operates.
        kernel_size (int): The size of the kernel for convolutional layers.
        stride (int): The stride for convolutional layers.
        use_spectral_norm (bool): Whether to use spectral normalization on the layers.
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
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
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(self.get_padding(5, 1), 0),
                    )
                ),
                self.norm_layer(
                    nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))
                ),
            ]
        )
        self.conv_post = self.norm_layer(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

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
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(2),
                PeriodDiscriminator(3),
                PeriodDiscriminator(5),
                PeriodDiscriminator(7),
                PeriodDiscriminator(11),
            ]
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


if __name__ == "__main__":
    from tqdm import tqdm
    import time

    # x = torch.rand(12, 2, 1, 512, 128).to("cuda")
    # model = MambaUNet(
    #     in_chans=1,
    #     depths=[2, 2, 2, 2],
    #     dims=[8, 16, 32, 64],
    #     # dims=[32, 64, 128, 256],
    #     ssm_d_state=1,
    #     ssm_ratio=2.0,
    #     ssm_dt_rank="auto",
    #     ssm_conv=3,
    #     ssm_conv_bias=False,
    #     forward_type="v5",
    #     mlp_ratio=4.0,
    #     patchembed_version="v2",
    #     downsample_version="v1",
    #     upsample_version="v1",
    #     output_version="v3",
    #     concat_skip=False,
    # ).to("cuda")

    # print(model.flops())

    x = torch.rand(12, 2, 1, 512, 128).to("cuda")
    model = DualStreamInteractiveMambaUNet(
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[8, 16, 32, 64],
        # dims=[32, 64, 128, 256],
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v5",
        mlp_ratio=4.0,
        patchembed_version="v2",
        downsample_version="v1",
        upsample_version="v1",
        output_version="v3",
        concat_skip=False,
    ).to("cuda")

    print(model.flops())
