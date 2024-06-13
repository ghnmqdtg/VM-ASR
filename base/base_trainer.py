import os
import math
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from prettytable import PrettyTable

from utils import load_from_path


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, models, metric_ftns, optimizer, config, logger):
        self.config = config
        self.logger = logger

        self.models = models
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = config.TRAIN.EPOCHS
        self.monitor = config.MONITOR

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = config.TRAIN.EARLY_STOPPING
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.log_dir = config.OUTPUT

        # setup visualization writer instance
        self.writer = TensorboardWriter(
            self.log_dir, self.logger, config.TENSORBOARD.ENABLE
        )

        if self.config.MODEL.RESUME_PATH is not None:
            self._resume_checkpoint()
        else:
            # Log info to indicate that the model is loaded. The training will start from scratch
            self.logger.info("Resume is not enabled. Training from scratch ...")

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        self.logger.info("Start training...")
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Train the model for an epoch
            self._train_epoch(epoch)
            # Check if do validation
            if self.do_validation:
                self._valid_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(self.epoch_log)

            # print logged informations to the screen
            self._log_epoch(log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        for key in self.models.keys():
            # The key would be either "generator" or names of the discriminators
            model = self.models[key]
            if model is None:
                continue
            model_name = "G" if key == "generator" else key
            model_type = "generator" if key == "generator" else "discriminator"
            state = {
                "name": model_name,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": self.optimizer[model_type].state_dict(),
                "monitor_best": self.mnt_best,
                "config": self.config,
            }
            # Save the latest checkpoint
            filename = os.path.join(self.log_dir, f"checkpoint-latest-{model_name}.pth")
            torch.save(state, filename)
            self.logger.info(
                f"Saving the latest {model_type} on epoch {epoch}: {filename} ..."
            )

            # Save the checkpoint with epoch number if SAVE_EPOCH is True
            if (
                self.config.SAVE_EPOCH_FREQ != -1
                and epoch % self.config.SAVE_EPOCH_FREQ == 0
            ):
                filename = os.path.join(
                    self.log_dir, f"checkpoint-epoch-{epoch}-{model_name}.pth"
                )
                torch.save(state, filename)
                self.logger.info(
                    f"Saving {model_type} on epoch {epoch}: {filename} ..."
                )

            # Save the best checkpoint
            if save_best:
                filename = os.path.join(
                    self.log_dir, f"checkpoint-best-{model_name}.pth"
                )
                torch.save(state, filename)
                self.logger.info(f"Saving best {model_type}: {filename} ...")

    def _resume_checkpoint(self):
        """
        Load a model checkpoint from the provided path.
        """
        self.models, self.optimizer, config, start_epoch = load_from_path(
            self.models, self.optimizer, self.config, self.logger
        )
        if self.config.FINETUNE:
            self.logger.info("Using pretrained model ...")
        else:
            self.config = config
            self.start_epoch = start_epoch
            self.logger.info(
                f"Resuming training from epoch {self.start_epoch} / {self.epochs} ..."
            )

    def _log_epoch(self, logs):
        """
        Log the training results of an epoch
        """
        table = PrettyTable()
        table.field_names = ["Metric", "Training", "Validation"]

        for key, value in logs.items():
            # If key is "epoch", set it to be title
            if key == "epoch":
                table.title = f"Epoch {value} / {self.epochs}"
                continue
            if key.startswith("val_"):
                continue
            val_key = f"val_{key}"
            val_value = logs.get(val_key, "N/A")

            # Format numbers to 2 decimal places
            if isinstance(value, float):
                value = f"{value:.4f}"
            if isinstance(val_value, float):
                val_value = f"{val_value:.4f}"
            table.add_row([key, value, val_value])

        self.logger.info(f"\n{table}")

        # Check for NaNs and Infs
        has_inf_or_nan = False
        for key, value in logs.items():
            if math.isnan(value) or math.isinf(value):
                self.logger.warning(f"Found an invalid value: {key} = {value}")
                has_inf_or_nan = True
        if has_inf_or_nan:
            self.logger.warning("Terminating due to invalid values.")
            exit(1)
