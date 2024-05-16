import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 24
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = "data/"
# Dataset name
_C.DATA.DATASET = "VCTK_092"
# Microphone id
_C.DATA.MIC_ID = "mic1"
# Resampler
# ! Don't use sox, the model learns nothing from the resampled data
_C.DATA.RESAMPLER = "scipy"
# Shuffle data
_C.DATA.SHUFFLE = True
# Number of workers for dataloading
_C.DATA.NUM_WORKERS = 2
# Quantity of data to use
_C.DATA.USE_QUANTITY = 0.1
# Train/test_split on VCTK dataset
_C.DATA.TRAIN_SPLIT = [100, 8]
# Train/validation split
_C.DATA.VALID_SPLIT = 0.1
# Max length of audio clip
# Target sample rate
_C.DATA.TARGET_SR = 48000
# Random resampling
_C.DATA.RANDOM_RESAMPLE = [8000, 48000] if _C.DATA.TARGET_SR == 48000 else [2000, 16000]
# Set the weight to randomly choose the SR
# The lower SR has higher probability to be chosen
_C.DATA.WEIGHTED_SR = CN()
_C.DATA.WEIGHTED_SR.ENABLE = False
_C.DATA.WEIGHTED_SR.RANGES = (
    [(8000, 16000), (16000, 24000), (24000, 48000)]
    if _C.DATA.TARGET_SR == 48000
    else [(2000, 8000), (8000, 12000), (12000, 16000)]
)
_C.DATA.WEIGHTED_SR.WEIGHTS = [0.5, 0.3, 0.2]
# Length of audio clip
_C.DATA.SEGMENT = 2.555
# White noise to pad to the short audio
_C.DATA.PAD_WHITENOISE = 1e-32
# STFT parameters
_C.DATA.STFT = CN()
_C.DATA.STFT.N_FFT = 1024
_C.DATA.STFT.HOP_LENGTH = 240 if _C.DATA.TARGET_SR == 48000 else 80
_C.DATA.STFT.WIN_LENGTH = 1024
_C.DATA.STFT.SCALE = "log2"
# Random low pass filter
_C.DATA.LPF = CN()
_C.DATA.LPF.MULTIFILTER = False
_C.DATA.LPF.LPF_TRAIN = [
    ("cheby1", 6),
    ("cheby1", 8),
    ("cheby1", 10),
    ("cheby1", 12),
    ("bessel", 6),
    ("bessel", 12),
    ("ellip", 6),
    ("ellip", 12),
]
_C.DATA.LPF.LPF_TEST = [("cheby1", 6)]

# Flac to wav
_C.DATA.FLAC2WAV = CN()
_C.DATA.FLAC2WAV.SRC_SR = 48000
_C.DATA.FLAC2WAV.SRC_PATH = _C.DATA.DATA_PATH
_C.DATA.FLAC2WAV.DST_PATH = "VCTK-Corpus-0.92/wav48_silence_trimmed_wav"
_C.DATA.FLAC2WAV.TIMESTAMPS = "./vctk-silence-labels/vctk-silences.0.92.txt"

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "VM_ASR"
# Model name
_C.MODEL.NAME = "VM_ASR_BASIC"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME_PATH = None
# Dropout rate
_C.MODEL.DROP_RATE = 0.0

# VSSM parameters
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.IN_CHANS = 1
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.DEPTHS = [2, 2, 2, 2]
_C.MODEL.VSSM.DIMS = 16
_C.MODEL.VSSM.SSM_D_STATE = 1
_C.MODEL.VSSM.SSM_RATIO = 1.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v04_ondwconv3"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.GMLP = False
_C.MODEL.VSSM.DROP_PATH_RATE = 0.1
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "LN"
_C.MODEL.VSSM.PATCHEMBED = "v2"
_C.MODEL.VSSM.DOWNSAMPLE = "v1"
_C.MODEL.VSSM.UPSAMPLE = "v1"
_C.MODEL.VSSM.OUTPUT = "v3"
_C.MODEL.VSSM.CONCAT_SKIP = True
# If the dim is not 5, we use the last layer of the encoder and the first layer of the decoder as the latent layer (4 VSS blocks in total)
# This option is to drop the last encoder layer, and only use the first decoder layer as the latent layer (only 2 VSS blocks in total)
_C.MODEL.VSSM.DROP_LAST_ENCODER = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.EARLY_STOPPING = 10
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BASE_LR = 1e-3
_C.TRAIN.MAX_LR = 1e-3
_C.TRAIN.MIN_LR = 1e-5
_C.TRAIN.CYCLE_MULT = 1.0
_C.TRAIN.ENABLE_GAN = False
_C.TRAIN.LOSSES = CN()
_C.TRAIN.METRICS = ["snr", "lsd", "lsd_hf", "lsd_lf"]
_C.TRAIN.LOSSES.GEN = ["multi_resolution_stft"]
_C.TRAIN.LOSSES.STFT_LOSS = CN()
# The factor controls how much the STFT loss should be enforced
# SC stands for spectral convergence loss
_C.TRAIN.LOSSES.STFT_LOSS.SC_FACTOR = 0.5
# MAG stands for Log STFT magnitude loss
_C.TRAIN.LOSSES.STFT_LOSS.MAG_FACTOR = 0.5
# Emphasize high frequency in STFT loss
_C.TRAIN.LOSSES.STFT_LOSS.EMPHASIZE_HIGH_FREQ = False
_C.TRAIN.LOW_FREQ_REPLACEMENT = False

# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Gradient clipping norm
_C.TRAIN.CLIP_GRAD = CN()
_C.TRAIN.CLIP_GRAD.ENABLE = False
_C.TRAIN.CLIP_GRAD.MAX_NORM = 1.0

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
# Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Adversarial training
_C.TRAIN.ADVERSARIAL = CN()
_C.TRAIN.ADVERSARIAL.ENABLE = False
_C.TRAIN.ADVERSARIAL.DISCRIMINATORS = [""]
# The feature loss lambda controls how much the similarity between the
# generated and the original features should be enforced
_C.TRAIN.ADVERSARIAL.MPD_HIDDEN = 32
_C.TRAIN.ADVERSARIAL.FEATURE_LOSS_LAMBDA = 100
_C.TRAIN.ADVERSARIAL.ONLY_FEATURE_LOSS = False
_C.TRAIN.ADVERSARIAL.ONLY_ADVERSARIAL_LOSS = False
# GAN loss type
# ! wgan or wgan-gp does not wor, use lsgan instead
_C.TRAIN.ADVERSARIAL.GAN_LOSS_TYPE = "lsgan"
_C.TRAIN.ADVERSARIAL.GP_LAMBDA = 10

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.RESULTS_DIR = "results"
_C.TEST.OVERLAP = 2000  # Overlap in samples
_C.TEST.SAVE_RESULT = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.DEBUG = False
_C.DEBUG_OUTPUT = "debug"
# Number of GPUs to use
_C.N_GPU = 1
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = "logs"
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Monitor mode for model performance, set to "off" to disable
_C.MONITOR = "min lsd"
# Frequency to save checkpoint in epoch (-1 for disabled)
_C.SAVE_EPOCH_FREQ = -1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 123
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Sync to wandb
_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = _C.MODEL.TYPE
_C.WANDB.ENTITY = None
_C.WANDB.MODE = "online"  # online/offline/disabled
_C.WANDB.LOG = "all"  # gradients/parameters/all/None
_C.WANDB.RESUME = False
_C.WANDB.TAGS = []
# Tensorboard settings
_C.TENSORBOARD = CN()
_C.TENSORBOARD.ENABLE = True
_C.TENSORBOARD.LOG_ITEMS = ["audio", "waveform", "spectogram"]


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    # Merge from specific arguments
    if _check_args("batch_size"):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args("resume"):
        config.MODEL.RESUME_PATH = args.resume
        if config.MODEL.RESUME_PATH is not None and not config.EVAL_MODE:
            config.WANDB.RESUME = True
    if _check_args("accumulation_steps"):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args("disable_amp"):
        config.AMP_ENABLE = False
    if _check_args("output"):
        config.OUTPUT = args.output
    if _check_args("tag"):
        config.TAG = args.tag
    if _check_args("eval"):
        config.EVAL_MODE = True
    if _check_args("throughput"):
        config.THROUGHPUT_MODE = True
    if _check_args("optim"):
        config.TRAIN.OPTIMIZER.NAME = args.optim
    if _check_args("target_sr"):
        if args.target_sr not in [16000, 48000]:
            raise ValueError("Target sample rate should be 16000 or 48000")

    # Output folder
    if config.MODEL.RESUME_PATH is None:
        config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    else:
        config.OUTPUT = config.MODEL.RESUME_PATH

    # Update configs based on the target SR
    if config.DATA.TARGET_SR == 48000:
        config.DATA.RANDOM_RESAMPLE = [8000, 48000]
        config.DATA.STFT.HOP_LENGTH = 240
        config.DATA.WEIGHTED_SR.RANGES = [(8000, 16000), (16000, 24000), (24000, 48000)]
    else:
        config.DATA.RANDOM_RESAMPLE = [2000, 16000]
        config.DATA.STFT.HOP_LENGTH = 80
        config.DATA.WEIGHTED_SR.RANGES = [(2000, 8000), (8000, 12000), (12000, 16000)]

    if _check_args("input_sr"):
        if config.DATA.TARGET_SR == 48000 and args.input_sr >= config.DATA.TARGET_SR:
            raise ValueError(
                f"Input sample rate should be less than {config.DATA.TARGET_SR}"
            )
        config.DATA.RANDOM_RESAMPLE = [args.input_sr]

    # Update low pass filter config
    if not config.EVAL_MODE:
        if not config.DATA.LPF.MULTIFILTER:
            config.DATA.LPF.LPF_TRAIN = [config.DATA.LPF.LPF_TRAIN[0]]

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
