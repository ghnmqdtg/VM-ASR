import os
import json
import time
import wandb
import random
import argparse
import itertools
import numpy as np
from data_loader.data_loaders import get_loader

from trainer.trainer import Trainer
from logger.logger import create_logger

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from model import get_model
import model.metric as module_metric
from utils.optimizer import get_optimizer
from utils.lr_scheduler import get_scheduler
from utils.utils import init_wandb_run, prepare_device

from config import get_config


def parse_option():
    parser = argparse.ArgumentParser(
        "VM-ASR training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--output",
        default="logs",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument(
        "--tag",
        default=time.strftime("%Y%m%d%H%M%S", time.localtime()),
        help="tag of experiment",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    # To make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    # Create output folder
    os.makedirs(config.OUTPUT, exist_ok=True)


def main(config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # Setup device with single/multiple GPUs
    device, device_ids = prepare_device(dist.get_rank())
    model = get_model(config)
    metrics = [getattr(module_metric, met) for met in config.TRAIN.METRICS]

    if config.WANDB.ENABLE:
        init_wandb_run(config)

    if not config.EVAL_MODE:
        data_loader_train, data_loader_val = get_loader(config, logger)

        # Initialize the optimizer and learning rate scheduler
        optimizers = {"generator": None, "discriminator": None}
        lr_schedulers = {"generator": None, "discriminator": None}
        # Get the optimizer and lr_scheduler for the generator
        optimizers["generator"] = get_optimizer(config, model["generator"], logger)
        lr_schedulers["generator"] = get_scheduler(
            config,
            optimizers["generator"],
            len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
        )
        # Get the optimizer and lr_scheduler for the discriminator
        if config.TRAIN.ADVERSARIAL.ENABLE:
            # There would be more than one discriminators if specified
            optimizers["discriminator"] = get_optimizer(
                config, itertools.chain(*model["discriminator"].parameters()), logger
            )
            lr_schedulers["discriminator"] = get_scheduler(
                config,
                optimizers["discriminator"],
                len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
            )

        # Log the model, optimizer and lr_scheduler
        logger.info(f"Model: {model}")
        logger.info(f"Optimizer: {optimizers}")
        logger.info(f"LR Scheduler: {lr_schedulers}")

        trainer = Trainer(
            model=model,
            metric_ftns=metrics,
            optimizer=optimizers,
            config=config,
            device=device,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_val,
            lr_scheduler=lr_schedulers,
            amp=config.AMP_ENABLE,
            gan=config.TRAIN.ADVERSARIAL.ENABLE,
            logger=logger,
        )

        trainer.train()
    else:
        data_loader_test = get_loader(config)
        logger.info(f"TESTING: ({len(data_loader_test)} files)")
        # Test the trained model
        return NotImplementedError("Testing is not implemented yet")

    if config.WANDB.ENABLE:
        wandb.finish()


if __name__ == "__main__":
    args, config = parse_option()

    # Init distributed env first
    setup_dist()
    # Set the random seed for reproducibility
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
    )
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
