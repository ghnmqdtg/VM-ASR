import os
from abc import abstractmethod
from logger import TensorboardWriter

from utils import load_from_path


class BaseInference:
    """
    Base class for all inference operations
    """

    def __init__(self, models, config, logger):
        self.config = config
        self.logger = logger
        self.models = models

        # Get the target sample rate from config
        self.input_sr = int(config.TAG.split("_")[0])
        self.target_sr = int(config.TAG.split("_")[1])

        # Set the output directory
        self.output_dir = os.path.join(
            self.config.OUTPUT, f"{self.input_sr}_{self.target_sr}_inference"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup visualization writer instance if needed
        self.writer = TensorboardWriter(
            self.output_dir, self.logger, config.TENSORBOARD.ENABLE
        )

        # Load the checkpoint
        self._resume_checkpoint()

    @abstractmethod
    def infer_file(self, file_path, sr_input=None):
        """
        Run inference on a single audio file
        """
        raise NotImplementedError

    @abstractmethod
    def infer_directory(self, dir_path, sr_input=None):
        """
        Run inference on all audio files in a directory
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
