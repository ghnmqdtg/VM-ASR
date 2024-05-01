import torch
import numpy as np
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.output_logger import log_audio, log_waveform, log_spectrogram
from data_loader import preprocessing, postprocessing
from tqdm.contrib.logging import logging_redirect_tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader_train,
        data_loader_val=None,
        lr_scheduler=None,
        amp=False,
        gan=False,
        logger=None,
        len_epoch=None,
    ):
        super().__init__(model, metric_ftns, optimizer, config, logger)
        self.config = config
        self.device = device
        self.data_loader = data_loader_train
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader_train)
            self.len_epoch = len_epoch
        self.data_loader_val = data_loader_val
        self.epoch_log = {}  # Log for metrics each epoch (both training and validation)
        self.do_validation = self.data_loader_val is not None
        self.lr_scheduler = lr_scheduler
        self.amp = amp  # Automatic Mixed Precision
        self.gan = gan  # Generative Adversarial Network

    def init_gan(self):
        return NotImplementedError

    def init_metrics(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def update_discriminator(self):
        raise NotImplementedError

    def update_metrics(self, metrics_values, training=True):
        # Update the batch metrics
        for key, value in metrics_values.items():
            if training:
                self.train_metrics.update(key, value)
            else:
                self.valid_metrics.update(key, value)

    def log_outputs(self):
        return NotImplemented

    @staticmethod
    def update_progress_bar(tepoch, metrics_values):
        raise NotImplementedError

    def _progress(self, batch_idx, training=True):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            if training:
                total = self.data_loader.split_sample_num["train"]
            else:
                total = self.data_loader.split_sample_num["valid"]
        else:
            total = self.len_epoch

        if batch_idx == -1:
            current = total
        else:
            current = batch_idx * self.data_loader.batch_size

        return base.format(current, total, 100.0 * current / total)
