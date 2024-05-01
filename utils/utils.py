import json
import wandb
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch import distributed as dist


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        # Set columns to float
        self._data = self._data.astype("float")
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def _get_wandb_config(config):
    wandb_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                wandb_config[f"{k}.{kk}"] = vv
        else:
            wandb_config[k] = v
    return wandb_config


def init_wandb_run(config):
    # Check if the distributed environment is available
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = -1
        world_size = -1

    # Initialize wandb
    # Logging multiple GPUs training is not supported yet
    if rank == 0 or not dist.is_available():
        experiment_name = f"{config.MODEL.TYPE}/{config.MODEL.NAME}"
        if config.TENSORBOARD.ENABLE:
            wandb.tensorboard.patch(root_logdir=config.OUTPUT, pytorch=True)

        wandb.init(
            project=config.WANDB.PROJECT,
            entity=config.WANDB.ENTITY,
            name=experiment_name,
            group=config.TAG,
            config=_get_wandb_config(config),
            dir=config.OUTPUT,
            resume=config.MODEL.RESUME,
            # Set id with the world size and rank
            id=f"{world_size}_{rank}",
            mode=config.WANDB.MODE,
            tags=config.WANDB.TAGS,
        )
    else:
        wandb.init(mode="disabled")
