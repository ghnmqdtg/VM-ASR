import os
from abc import abstractmethod
from logger import TensorboardWriter
from prettytable import PrettyTable

from utils import load_from_path


class BaseTester:
    """
    Base class for all testers
    """

    def __init__(self, models, metric_ftns, config, logger):
        self.config = config
        self.logger = logger

        self.models = models
        self.metric_ftns = metric_ftns

        self.input_sr = int(config.TAG.split("_")[0])
        self.target_sr = int(config.TAG.split("_")[1])

        self.test_loader = None
        self.output_dir = self.config.OUTPUT

        # setup visualization writer instance
        self.writer = TensorboardWriter(
            self.output_dir, self.logger, config.TENSORBOARD.ENABLE
        )

        # Load the checkpoint
        self._resume_checkpoint()

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the models on the configured dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_batch(self, target, output, hf=None):
        """
        Evaluate the model for one batch of data.
        """
        raise NotImplementedError

    def _resume_checkpoint(self):
        """
        Load a model checkpoint from the provided path.
        """
        self.models, _, self.config, _ = load_from_path(
            self.models, None, self.config, self.logger
        )
        if self.target_sr != self.config.DATA.TARGET_SR:
            self.logger.error(
                f"Target sampling rate mismatch: {self.target_sr} vs {self.config.DATA.TARGET_SR}, please choose the correct checkpoint."
            )
            # Exit the program
            exit(1)
        if self.input_sr not in range(
            self.config.DATA.RANDOM_RESAMPLE[0], self.config.DATA.RANDOM_RESAMPLE[1] + 1
        ):
            self.logger.error(
                f"Input sampling rate mismatch: {self.input_sr} not in {self.config.DATA.RANDOM_RESAMPLE}, please choose the correct checkpoint."
            )
            # Exit the program
            exit(1)
        self.logger.info(
            f"Checkpoint loaded successfully from {self.config.MODEL.RESUME_PATH}"
        )

    def _log_results(self, results):
        """
        Log the summarized results using PrettyTable or any other formatting method.
        """
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for metric, value in results.items():
            table.add_row([metric, f"{value:.4f}"])
        self.logger.info(f"Summary of Evaluation:\n{table}")
