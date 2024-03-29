import os
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import wandb


# Fix random seeds for reproducibility
SEED = 9527
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    run_id = str(config.log_dir).split("/")[-1]
    run_name = f'{config["arch"]["type"]}-{run_id}'
    # Check if wandb is enabled
    if config["trainer"]["sync_wandb"]:
        wandb.tensorboard.patch(
            root_logdir=os.path.join("./", config.log_dir),
            pytorch=True,
            tensorboard_x=False,
        )
        wandb.init(project=config["name"], name=run_name, config=config)

    logger = config.get_logger("train")

    # Set seed for reproducibility

    # Setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # Build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    # Show number of parameters and FLOPs
    logger.info(model)
    # Show model summary
    logger.info(model.flops(shape=data_loader.data_shape))
    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    criterion = {crit: getattr(module_loss, crit) for crit in config["loss"]}
    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    # Build optimizer, learning rate scheduler. Delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()

    if config["trainer"]["sync_wandb"]:
        # Finish logging
        wandb.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Mamba ASR")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
