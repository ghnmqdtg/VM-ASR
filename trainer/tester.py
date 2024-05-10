import os
import csv
import time
from tqdm import tqdm
from base import BaseTester
from tqdm.contrib.logging import logging_redirect_tqdm
from logger.visualization import log_audio, log_waveform, log_spectrogram

from utils import MetricTracker
from utils.post_processing import unfold_audio, fold_audio

import torch
import torchaudio


class Tester(BaseTester):
    """
    Tester class
    """

    def __init__(
        self,
        models,
        metric_ftns,
        config,
        device,
        data_loader,
        logger=None,
    ):
        super().__init__(models, metric_ftns, config, logger)
        self.config = config
        self.device, self.device_ids = device
        self.test_loader = data_loader
        self.test_log = {}
        self.metrics = []
        self.test_metrics = MetricTracker(
            *self.metrics,
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )

        # Set the number of frames per segment
        self.num_frames_per_seg = int(
            int(self.config.DATA.SEGMENT * self.config.DATA.FLAC2WAV.SRC_SR)
            * self.target_sr
            / self.config.DATA.FLAC2WAV.SRC_SR
        )

        # Set models to device
        for key, model in self.models.items():
            self.models[key] = model.to(self.device)

        self.logger.info(f"Test metrics: {self.test_metrics.get_keys()}")

    def evaluate(self):
        """
        Evaluate the models on the configured dataset.
        """
        self.logger.info("Starting evaluation...")
        for model_key, model in self.models.items():
            model.eval()

        # Reset the train metrics
        self.test_metrics.reset()

        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
                with tqdm(
                    self.test_loader,
                    desc=f"[TEST] | {self.input_sr} to {self.target_sr} | {self._progress(0)}",
                    unit="batch",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    postfix={key: 0.0 for key in self.test_metrics.get_keys()},
                ) as tepoch:
                    for batch_idx, (
                        wave_input,
                        wave_target,
                        highcut,
                        filename,
                        pad_length,
                    ) in enumerate(tepoch):
                        # Reset the peak memory stats for the GPU
                        torch.cuda.reset_peak_memory_stats()
                        metrics_values = {}
                        wave_input = wave_input.to(self.device)
                        wave_target = wave_target.to(self.device)
                        if wave_input.size(2) <= self.num_frames_per_seg:
                            start_time = time.time()
                            # Forward pass
                            wave_out = self.models["generator"](wave_input, highcut)
                            # Compute metrics
                            run_time = time.time() - start_time
                            rtf = (
                                run_time
                                / (wave_input.size(2) - pad_length[0].item())
                                * self.config.DATA.TARGET_SR
                            )
                            metrics_values = self._evaluate_batch(
                                wave_out, wave_target, highcut
                            )
                            metrics_values["rtf"] = rtf
                            metrics_values["rtf_reciprocal"] = 1 / rtf
                        else:
                            run_time = time.time() - start_time
                            # Unfold the audio tensor into overlapping segments
                            segments = unfold_audio(
                                audio=wave_input,
                                segment_length=self.num_frames_per_seg,
                                overlap=self.config.TEST.OVERLAP,
                            )
                            processed_segments = torch.zeros_like(segments).to(
                                self.device
                            )
                            for i in range(segments.size(2)):
                                start_time = time.time()
                                # Forward pass
                                seg_out = self.models["generator"](
                                    segments[:, :, i], highcut
                                )
                                processed_segments[:, :, i] = seg_out
                            # Fold the processed segments back into the full audio
                            wave_out = fold_audio(
                                processed_segments,
                                total_length=wave_input.size(2),
                                segment_length=self.num_frames_per_seg,
                                overlap=self.config.TEST.OVERLAP,
                            )
                            run_time = time.time() - start_time
                            rtf = (
                                run_time
                                / (wave_input.size(2) - pad_length[0].item())
                                * self.config.DATA.TARGET_SR
                            )
                            metrics_values = self._evaluate_batch(
                                wave_out, wave_target, highcut
                            )
                            metrics_values["rtf"] = rtf
                            metrics_values["rtf_reciprocal"] = 1 / rtf

                        # Calculate the metrics
                        self.update_metrics(metrics_values)
                        # Update the progress bar
                        self.update_progress_bar(tepoch, metrics_values)

                        if self.config.TEST.SAVE_RESULT:
                            # Trim the output and target to the original length
                            if pad_length[0].item() != 0:
                                trim_length = wave_input.size(2) - pad_length[0].item()
                                wave_input = wave_input[:, :, :trim_length]
                                wave_out = wave_out[:, :, :trim_length]
                                wave_target = wave_target[:, :, :trim_length]

                            # Save audio in 16-bit PCM format using torchaudio
                            # TODO: Save to different output directory for each model
                            torchaudio.save(
                                f"{self.output_dir}/{filename[0].replace('.wav', '')}_up.wav",
                                wave_out[0].cpu().detach(),
                                self.config.DATA.TARGET_SR,
                                bits_per_sample=16,
                            )
                            torchaudio.save(
                                f"{self.output_dir}/{filename[0].replace('.wav', '')}_orig.wav",
                                wave_target[0].cpu().detach(),
                                self.config.DATA.TARGET_SR,
                                bits_per_sample=16,
                            )
                            torchaudio.save(
                                f"{self.output_dir}/{filename[0].replace('.wav', '')}_down.wav",
                                wave_input[0].cpu().detach(),
                                self.config.DATA.TARGET_SR,
                                bits_per_sample=16,
                            )

                # Save the log of the epoch into dict
                self.test_log = self.test_metrics.result()
                self.test_log.update({"sample_rate": self.input_sr})
                # Log the results
                self._log_results(self.test_log)
                result_filename = (
                    "results_16kHz.csv"
                    if self.target_sr == 16000
                    else "results_48kHz.csv"
                )
                # Save the results to a CSV file
                self.save_results_to_csv(self.test_log, filename=result_filename)

    def _evaluate_batch(self, wave_out, wave_target, highcut):
        metrics = {
            metric.__name__: metric(
                wave_out.squeeze(1), wave_target.squeeze(1), hf=highcut
            )
            for metric in self.metric_ftns
        }
        return metrics

    def update_metrics(self, metrics_values):
        # Update the batch metrics
        for key, value in metrics_values.items():
            self.test_metrics.update(key, value)

    @staticmethod
    def update_progress_bar(tepoch, metrics_values):
        progress_metrics = metrics_values
        # Add the memory usage to the progress bar
        progress_metrics["mem"] = (
            f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f} MB"
        )
        tepoch.set_postfix(progress_metrics)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        total = len(self.test_loader.dataset)
        if batch_idx == -1:
            current = total
        else:
            current = batch_idx * self.test_loader.batch_size

        return base.format(current, total, 100.0 * current / total)

    def save_results_to_csv(self, results, filename="results.csv"):
        # Check if file exists. If not, create it and write headers
        file_exists = os.path.isfile(filename)
        # Reorder the dict
        desired_order_list = [
            "sample_rate",
            "snr",
            "lsd",
            "lsd_hf",
            "lsd_lf",
            "rtf",
            "rtf_reciprocal",
        ]
        results = {k: results[k] for k in desired_order_list}

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([key.upper() for key in results.keys()])
            writer.writerow(results.values())
