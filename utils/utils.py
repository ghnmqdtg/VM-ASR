import os
import glob
import wandb
import torch
import pandas as pd
from itertools import repeat

import torch.nn.functional as F


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

        # # TESTING: Save or update the metric to ./metric.csv
        # self._data.to_csv("./metric.csv")

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # If the key is not in the index, add it
        if key not in self._data.index:
            # Initialize new row for the new key
            self._data.loc[key] = [0, 0, 0]

        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

        # # TESTING: Save or update the metric to ./metric.csv
        # self._data.to_csv("./metric.csv")

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def get_keys(self):
        return self._data.index.tolist()


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

    experiment_name = f"{config.TAG}"
    if config.TENSORBOARD.ENABLE:
        wandb.tensorboard.patch(root_logdir=config.OUTPUT, pytorch=True)

    wandb.init(
        project=config.WANDB.PROJECT,
        entity=config.WANDB.ENTITY,
        name=experiment_name,
        group=config.WANDB.TAGS[0],
        config=_get_wandb_config(config),
        dir=config.OUTPUT,
        resume=True if config.MODEL.RESUME_PATH else False,
        mode=config.WANDB.MODE,
        tags=config.WANDB.TAGS,
    )


def load_from_path(models, optimizer, config, logger):
    """
    Load best models from the specified folder.
    """
    start_epoch = 1
    resume_path = config.MODEL.RESUME_PATH

    logger.info(f"Loading checkpoint from folder: {resume_path}")
    # Check if there are any checkpoints
    if not config.EVAL_MODE and not config.INFERENCE_MODE:
        # Training mode: Try to load the latest checkpoint, if not found, train from scratch
        # Get all the .pth files that has "best" in the name
        checkpoint_files = glob.glob(os.path.join(resume_path, "*best*.pth"))
        if len(checkpoint_files) == 0:
            logger.info(
                f"No best checkpoints found in the folder. Training from scratch ..."
            )
            return (models, optimizer, config, start_epoch)
        else:
            for file in checkpoint_files:
                try:
                    checkpoint = torch.load(file)
                    # Get the model type
                    model_type = file.split("-")[2].split(".")[0]
                    key = "generator" if model_type == "G" else model_type
                    # Load the model
                    models[key].load_state_dict(checkpoint["state_dict"])
                    optimizer_key = (
                        "generator" if model_type == "G" else "discriminator"
                    )
                    optimizer[optimizer_key].load_state_dict(checkpoint["optimizer"])
                    # Update the configurations
                    if key == "generator":
                        # Update the generator config
                        config = checkpoint["config"]
                        config.defrost()
                        config.MODEL.RESUME_PATH = resume_path
                        config.freeze()
                        start_epoch = checkpoint["epoch"] + 1
                    # Log the loaded model
                    logger.info(f"Loaded {key} from {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
                    exit(1)
    else:
        # Evaluation mode: Load the best checkpoint, if not found, exit
        # Get all the .pth files that has "best" in the name
        checkpoint_files = glob.glob(os.path.join(resume_path, "*best-G*.pth"))
        if len(checkpoint_files) == 0:
            logger.error(
                f"No best checkpoints found in the folder. Please check the path: {resume_path}"
            )
            exit(1)
        else:
            try:
                # Load the model
                checkpoint = torch.load(checkpoint_files[0])
                models["generator"].load_state_dict(checkpoint["state_dict"])
                # Log the loaded model
                logger.info(f"Loaded generator from {checkpoint_files[0]}")
            except:
                logger.error(
                    f"Error loading {checkpoint_files[0]}, the .pth file might be corrupted"
                )
                exit(1)

    return (models, optimizer, config, start_epoch)
