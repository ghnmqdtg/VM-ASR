import os
import time
import glob
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path
from scipy.signal import resample_poly

import torch
import torchaudio

from base import BaseInference
from utils.post_processing import unfold_audio, fold_audio


class Inferencer(BaseInference):
    """
    Inference class for audio super resolution
    """

    def __init__(
        self,
        models,
        metric_ftns,
        config,
        device,
        logger=None,
    ):
        super().__init__(models, config, logger)
        self.config = config
        self.device, self.device_ids = device
        self.metric_ftns = metric_ftns

        # Set the number of frames per segment
        self.num_frames_per_seg = int(
            int(self.config.DATA.SEGMENT * self.config.DATA.FLAC2WAV.SRC_SR)
            * self.target_sr
            / self.config.DATA.FLAC2WAV.SRC_SR
        )

        # Move models to device
        for key, model in self.models.items():
            self.models[key] = model.to(self.device)

        # Create the dataset
        self.data = InferenceData(config)

    def infer_file(self, file_path, output_dir=None, iters=False):
        """
        Run inference on a single audio file

        Args:
            file_path: Path to the audio file
            output_dir: Optional output directory to save the results

        Returns:
            The processed audio waveform
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None

        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load the input audio file
        wave_input, highcut, pad_length = self.data.load_input(file_path)
        wave_input = wave_input.to(self.device)

        # Process the audio
        start_time = time.time()
        with torch.no_grad():
            if wave_input.size(2) <= self.num_frames_per_seg:
                # Forward pass
                wave_out = self.models["generator"](wave_input, highcut)
                # Compute metrics
                run_time = time.time() - start_time
                rtf = run_time / (
                    (wave_input.size(2) - pad_length[0].item())
                    / self.config.DATA.TARGET_SR
                )
            else:
                run_time = time.time() - start_time
                # Unfold the audio tensor into overlapping segments
                segments = unfold_audio(
                    audio=wave_input,
                    segment_length=self.num_frames_per_seg,
                    overlap=self.config.TEST.OVERLAP,
                )
                processed_segments = torch.zeros_like(segments).to(self.device)
                for i in range(segments.size(2)):
                    # Forward pass
                    seg_out = self.models["generator"](segments[:, :, i], highcut)
                    processed_segments[:, :, i] = seg_out
                # Fold the processed segments back into the full audio
                wave_out = fold_audio(
                    processed_segments,
                    total_length=wave_input.size(2),
                    segment_length=self.num_frames_per_seg,
                    overlap=self.config.TEST.OVERLAP,
                )

        run_time = time.time() - start_time
        if not iters:
            self.logger.info(f"Processing completed in {run_time:.2f}s")

        # Save the result
        output_file = os.path.join(output_dir, f"{Path(file_path).stem}_enhanced.wav")
        torchaudio.save(
            output_file,
            wave_out[0].cpu().detach(),
            self.target_sr,
            bits_per_sample=16,
        )
        if not iters:
            self.logger.info(f"Enhanced audio saved to {output_file}")

        return wave_out

    def infer_directory(
        self,
        dir_path,
        output_dir=None,
        file_types=(".wav", ".mp3", ".flac"),
    ):
        """
        Run inference on all audio files in a directory

        Args:
            dir_path: Path to the directory containing audio files
            output_dir: Optional output directory to save results
            file_types: Tuple of file extensions to process

        Returns:
            List of paths to processed files
        """
        if not os.path.exists(dir_path):
            self.logger.error(f"Directory not found: {dir_path}")
            return []

        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, os.path.basename(dir_path))
        os.makedirs(output_dir, exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in file_types:
            audio_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))

        if not audio_files:
            self.logger.warning(f"No audio files found in {dir_path}")
            return []

        self.logger.info(f"Found {len(audio_files)} audio files to process")

        # Process each file
        processed_files = []
        with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
            with tqdm(
                audio_files,
                desc=f"[INFERENCE] | {self.input_sr} to {self.target_sr} | Loading...",
                unit="file",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as t:
                for file_path in t:
                    t.set_description(
                        f"[INFERENCE] | {self.input_sr} to {self.target_sr} | {os.path.basename(file_path)}"
                    )
                    output = self.infer_file(file_path, output_dir, iters=True)
                if output is not None:
                    output_file = os.path.join(
                        output_dir, f"{Path(file_path).stem}_enhanced.wav"
                    )
                    processed_files.append(output_file)

        self.logger.info(f"Processed {len(processed_files)} files")
        return processed_files


class InferenceData:
    def __init__(self, config):
        self.config = config

    def load_input(self, file_path):
        """
        Load the input audio file
        """
        audio, sr = torchaudio.load(file_path)
        target_sr = int(self.config.TAG.split("_")[1])
        num_frames = int(self.config.DATA.SEGMENT * self.config.DATA.FLAC2WAV.SRC_SR)

        # Check if the sample rate of the original audio is the same as the target sample rate
        if sr != target_sr:
            # Resample the audio
            audio = self.resample_audio(
                audio, sr, target_sr, self.config.DATA.RESAMPLER
            )
            sr = target_sr
            # Update the num_frames based on the ratio of TARGET_SR and SRC_SR
            num_frames = int(num_frames * target_sr / self.config.DATA.FLAC2WAV.SRC_SR)

        # Check if the audio is stereo, convert it to mono
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # Pad the audio if the length is less than the specified number of frames
        pad_length = 0
        if audio.shape[-1] < num_frames:
            # Generate white noise
            pad_length = num_frames - audio.shape[-1]
            white_noise = (
                torch.randn((pad_length)) * self.config.DATA.PAD_WHITENOISE
            ).unsqueeze(0)
            # Pad the audio with white noise
            audio = torch.cat((audio, white_noise), dim=-1)
        # For the testing, some of audio would be longer than the number of frames
        # We pad them to match multiple of the number of frames
        elif audio.shape[-1] % num_frames != 0:
            pad_length = num_frames - (audio.shape[-1] % num_frames)
            white_noise = (
                torch.randn((pad_length)) * self.config.DATA.PAD_WHITENOISE
            ).unsqueeze(0)
            audio = torch.cat((audio, white_noise), dim=-1)

        # Highcut frequency = int((1 + n_fft // 2) * (sr_input // sr_target))
        highcut = int(
            (1 + self.config.DATA.STFT.N_FFT // 2) * (sr / self.config.DATA.TARGET_SR)
        )

        # Insert 1 channel dimension to mimic the batch dimension
        audio = audio.unsqueeze(0)
        highcut = torch.tensor(highcut).unsqueeze(0)
        pad_length = torch.tensor(pad_length).unsqueeze(0)

        return audio, highcut, pad_length

    def resample_audio(
        self, waveform: torch.Tensor, sr_org: int, sr_new: int, resampler: str
    ) -> torch.Tensor:
        """
        Resample the waveform to the new sample rate

        Args:
            waveform (torch.Tensor): The input waveform
            sr_org (int): The original sample rate
            sr_new (int): The new sample rate
            resampler (str): The resampler to use

        Returns:
            torch.Tensor: The resampled waveform
        """
        if resampler == "sox":
            # Convert the PyTorch tensor to a NumPy array
            waveform_np = waveform.squeeze().numpy()
            # Set the sample rate for the resampler
            self.sox_tfm.set_output_format(rate=sr_new)
            # Resample the audio
            waveform_resampled_np = self.sox_tfm.build_array(
                input_array=waveform_np, sample_rate_in=sr_org
            )
            waveform_resampled = torch.tensor(
                waveform_resampled_np, dtype=torch.float32
            ).unsqueeze(0)
            # Clear the effects
            self.sox_tfm.clear_effects()
        else:
            # Resample the audio using scipy
            waveform_resampled_np = resample_poly(
                waveform.numpy(), sr_new, sr_org, axis=-1
            )
            waveform_resampled = torch.tensor(
                waveform_resampled_np, dtype=torch.float32
            )

        return waveform_resampled
