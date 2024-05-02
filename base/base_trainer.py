import os
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from prettytable import PrettyTable


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
        self.save_period = config.SAVE_FREQ
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

        if config.TRAIN.USE_CHECKPOINT is not None:
            self._resume_checkpoint(config.TRAIN.USE_CHECKPOINT)

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

            if epoch % self.save_period == 0:
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
            model_name = "G" if key == "generator" else "D"
            model_type = "generator" if key == "generator" else "discriminator"
            state = {
                "name": model_name,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": self.optimizer[model_type].state_dict(),
                "monitor_best": self.mnt_best,
                "config": self.config,
            }
            filename = os.path.join(
                self.log_dir, f"checkpoint-epoch-{epoch}-{model_name}.pth"
            )
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

            if save_best:
                best_path = os.path.join(self.log_dir, f"model_best_{model_name}.pth")
                torch.save(state, best_path)
                self.logger.info(
                    f"Saving current best: model_best_{model_name}.pth ..."
                )

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        # TODO: Add resume from the best model
        return NotImplementedError
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["name"] != self.config.MODEL.NAME:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.models.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
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
