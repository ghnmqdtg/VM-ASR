import os
import json
import time
import glob
import wandb
import random
import argparse
import numpy as np
from data_loader.data_loaders import get_loader

from trainer.trainer import Trainer
from trainer.tester import Tester
from trainer.inferencer import Inferencer
from logger.logger import create_logger

import torch
import torch.backends.cudnn as cudnn

from model import get_model
import model.metric as module_metric
from utils.optimizer import get_optimizer
from utils.lr_scheduler import get_scheduler
from utils.utils import init_wandb_run, prepare_device

from config import get_config


def parse_option():
    parser = argparse.ArgumentParser(
        "VM-ASR training, evaluation, and inference script", add_help=False
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
    parser.add_argument(
        "--input_sr",
        type=int,
        help="the input sample rate (if set, the random resample will be disabled)",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        help="the target sample rate",
    )
    parser.add_argument("--resume", type=str, help="path to checkpoint for models")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
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
        "--inference", action="store_true", help="Perform inference only"
    )
    parser.add_argument(
        "--input", type=str, help="Input file or directory for inference"
    )
    # TODO: Add throughput mode
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # Setup device with single/multiple GPUs
    device, device_ids = prepare_device(config.N_GPU)
    # Get the model
    models = get_model(config)
    # Set the models to device
    models = {k: v.to(device) for k, v in models.items()}
    # Get the metrics
    metrics = [getattr(module_metric, met) for met in config.TRAIN.METRICS]

    if config.WANDB.ENABLE and not config.EVAL_MODE and not config.INFERENCE_MODE:
        init_wandb_run(config)

    # Inference mode
    if config.INFERENCE_MODE:
        logger.info(f"Starting inference...")
        logger.info(f"Loading checkpoint from {config.MODEL.RESUME_PATH}")
        # Make sure we only have the generator model for inference
        models = {"generator": models["generator"]}

        # Create the inferencer
        inferencer = Inferencer(
            models=models,
            metric_ftns=metrics,
            config=config,
            device=(device, device_ids),
            logger=logger,
        )

        # Check if input is specified
        if args.input is None:
            logger.error("Input path must be specified for inference mode")
            return

        # Run inference on the specified input
        if os.path.isfile(args.input):
            inferencer.infer_file(args.input)
        elif os.path.isdir(args.input):
            inferencer.infer_directory(args.input)
        else:
            logger.error(f"Input path does not exist: {args.input}")

        logger.info("Inference completed successfully")
        return

    # Evaluation mode
    elif config.EVAL_MODE:
        logger.info(f"Starting evaluation ...")
        logger.info(f"Loading checkpoint from {config.MODEL.RESUME_PATH}")
        # Remove models except the generator
        models = {"generator": models["generator"]}
        data_loader_test = get_loader(config, logger)
        logger.info(f"TESTING: ({len(data_loader_test)} files)")
        # Test the trained model
        tester = Tester(
            models=models,
            metric_ftns=metrics,
            config=config,
            device=(device, device_ids),
            data_loader=data_loader_test,
            logger=logger,
        )

        tester.evaluate()

    # Training mode
    else:
        data_loader_train, data_loader_val = get_loader(config, logger)
        # Initialize the optimizer and learning rate scheduler
        optimizers = {"generator": None, "discriminator": None}
        lr_schedulers = {"generator": None, "discriminator": None}
        # Get the optimizer and lr_scheduler for the generator
        optimizers["generator"] = get_optimizer(config, models["generator"], logger)
        lr_schedulers["generator"] = (
            get_scheduler(
                config,
                optimizers["generator"],
                len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
            )
            if config.TRAIN.ACCUMULATION_STEPS > 1
            else get_scheduler(config, optimizers["generator"], len(data_loader_train))
        )
        # Get the optimizer and lr_scheduler for the discriminator
        if config.TRAIN.ADVERSARIAL.ENABLE:
            if config.TRAIN.ADVERSARIAL.DISCRIMINATORS is not None:
                # There would be more than one discriminators if specified
                # models["discriminator"] is a dict with keys as discriminator names, we need to iterate over them
                optimizers["discriminator"] = get_optimizer(
                    config,
                    [
                        models[disc_name]
                        for disc_name in config.TRAIN.ADVERSARIAL.DISCRIMINATORS
                    ],
                    logger,
                )
                lr_schedulers["discriminator"] = (
                    get_scheduler(
                        config,
                        optimizers["discriminator"],
                        len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
                    )
                    if config.TRAIN.ACCUMULATION_STEPS > 1
                    else get_scheduler(
                        config, optimizers["discriminator"], len(data_loader_train)
                    )
                )
            else:
                # Log an error if the discriminator is not specified
                logger.error(
                    "Adversarial training is enabled but the discriminator is not specified in the config file. Please specify the discriminator in the config file."
                )

        trainer = Trainer(
            models=models,
            metric_ftns=metrics,
            optimizers=optimizers,
            config=config,
            device=(device, device_ids),
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_val,
            lr_schedulers=lr_schedulers,
            amp=config.AMP_ENABLE,
            gan=config.TRAIN.ADVERSARIAL.ENABLE,
            logger=logger,
        )

        trainer.train()

    if config.WANDB.ENABLE:
        wandb.finish()


def validate_resume_path(config):
    assert os.path.exists(
        config.MODEL.RESUME_PATH
    ), f"Folder not found, please check the path: {config.MODEL.RESUME_PATH}"
    if config.EVAL_MODE or config.INFERENCE_MODE:
        # There must be a checkpoint for evaluation or inference
        assert (
            glob.glob(os.path.join(config.MODEL.RESUME_PATH, "*.pth")) != []
        ), f"No checkpoint found in the folder. Please check the path: {config.MODEL.RESUME_PATH}"


def setup_test(config):
    # Evaluate the trained model with the test dataset
    assert (
        len(config.TAG.split("_")) == 2
    ), "TAG should be in format {input_sr}_{target_sr}"
    input_sr, target_sr = config.TAG.split("_")
    # Example: "./results/16k_DeciData_MPD_WGAN_Local/16000/2000"
    output_dir = os.path.join(
        config.TEST.RESULTS_DIR,
        os.path.basename(config.MODEL.RESUME_PATH),
        target_sr,
        input_sr,
    )
    # Remove the existing output directory
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # Update config
    config.defrost()
    config.OUTPUT = output_dir
    config.freeze()
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    return config, logger


def setup_inference(config):
    # Setup for inference mode
    assert (
        len(config.TAG.split("_")) == 2
    ), "TAG should be in format {input_sr}_{target_sr}"
    input_sr, target_sr = config.TAG.split("_")
    # Create inference output directory
    output_dir = os.path.join(
        config.INFERENCE.RESULTS_DIR,
        os.path.basename(config.MODEL.RESUME_PATH),
        target_sr,
        input_sr,
    )
    os.makedirs(output_dir, exist_ok=True)
    # Update config
    config.defrost()
    config.OUTPUT = output_dir
    config.freeze()
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    return config, logger


if __name__ == "__main__":
    args, config = parse_option()

    # Create output folder
    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(config.DEBUG_OUTPUT, exist_ok=True)
    # Set the random seed for reproducibility
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if config.MODEL.RESUME_PATH is not None:
        validate_resume_path(config)
        if config.INFERENCE_MODE:
            config, logger = setup_inference(config)
        elif config.EVAL_MODE:
            config, logger = setup_test(config)
        else:
            logger = create_logger(
                output_dir=config.MODEL.RESUME_PATH,
                name=f"{config.MODEL.NAME}",
                load_existing=True,
            )
            logger.info(f"Resume training from {config.MODEL.RESUME_PATH}")
    else:
        logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

    main(config)
