from .model import *
from .vmamba import *
from .discriminator import *

from utils.stft import *


def get_model(config):
    models = {"generator": None}

    if config.MODEL.NAME == "DualStreamInteractiveMambaUNet":
        generator = DualStreamInteractiveMambaUNet(
            in_chans=config.MODEL.VSSM.IN_CHANS,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.DIMS,
            # ==============================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_dt_rank=(
                "auto"
                if config.MODEL.VSSM.SSM_DT_RANK == "auto"
                else int(config.MODEL.VSSM.SSM_DT_RANK)
            ),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # =========================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            gmlp=config.MODEL.VSSM.GMLP,
            # =========================
            drop_path_rate=config.MODEL.VSSM.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            upsample_version=config.MODEL.VSSM.UPSAMPLE,
            output_version=config.MODEL.VSSM.OUTPUT,
            concat_skip=config.MODEL.VSSM.CONCAT_SKIP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # =========================
            # FFT related parameters
            n_fft=config.DATA.STFT.N_FFT,
            hop_length=config.DATA.STFT.HOP_LENGTH,
            win_length=config.DATA.STFT.WIN_LENGTH,
            spectro_scale=config.DATA.STFT.SCALE,
        )
        models["generator"] = generator

    if config.TRAIN.ADVERSARIAL:
        if "mpd" in config.TRAIN.ADVERSARIAL.DISCRIMINATORS:
            discriminator = MultiPeriodDiscriminator()
            models["mpd"] = discriminator

    return models
