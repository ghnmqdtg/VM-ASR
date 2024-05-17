from typing import Tuple
import os
import sox
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Resample and filter the audio
from scipy.signal import resample_poly
from scipy.signal import sosfiltfilt
from scipy.signal import butter, butter, cheby1, ellip, bessel

import torch
import torchaudio
import torchaudio.datasets as datasets
from torch.nn import functional as F
from torch.utils.data import random_split

from utils.stft import wav2spectro


def get_loader(config, logger):
    # Check if the dataset is VCTK_092
    if config.DATA.DATASET == "VCTK_092":
        if not config.EVAL_MODE:
            # Load the whole dataset for training and validation
            dataset = CustomVCTK_092(config, training=True, logger=logger)
            # Split the dataset into training and validation
            train_size = int(len(dataset) * (1 - config.DATA.VALID_SPLIT))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            # Create the data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=True,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=True,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True,
            )

            # # Get the number of samples in each split
            # split_sample_num = {
            #     "train": len(train_sampler) if train_sampler is not None else 0,
            #     "valid": len(val_sampler) if val_sampler is not None else 0,
            # }

            # print(
            #     f"Number of samples in each split: {split_sample_num['train']} training, {split_sample_num['valid']} validation"
            # )

            return train_loader, val_loader
        else:
            # Load the testing dataset
            test_dataset = CustomVCTK_092(config, training=False, logger=logger)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True,
            )
            return test_loader

    else:
        raise NotImplementedError(f"Dataset {config.DATA.DATASET} not implemented")


class CustomVCTK_092(datasets.VCTK_092):
    def __init__(
        self,
        config,
        audio_ext=".wav",
        training=True,
        logger=None,
        **kwargs,
    ):
        self.config = config
        self.logger = logger
        self.training = training
        # Check if the trimmed wav files exist
        if not os.path.isdir(
            os.path.join(self.config.DATA.DATA_PATH, self.config.DATA.FLAC2WAV.DST_PATH)
        ):
            # Print the message
            self.logger.info(
                "Trimmed wav files not found. Download and Convert flac to wav..."
            )
            # Convert flac to wav
            self._flac2wav()
            super().__init__(
                root=self.config.DATA.DATA_PATH, mic_id=self.config.DATA.MIC_ID
            )
        else:
            super().__init__(
                root=self.config.DATA.DATA_PATH, mic_id=self.config.DATA.MIC_ID
            )

        self._path = os.path.join(self.config.DATA.DATA_PATH, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(
            self.config.DATA.DATA_PATH, self.config.DATA.FLAC2WAV.DST_PATH
        )
        self._audio_ext = audio_ext
        self.training_test_split = self.config.DATA.TRAIN_SPLIT
        self.quantity = (
            (
                self.config.DATA.USE_QUANTITY
                if self.config.DATA.USE_QUANTITY > 0.0
                and self.config.DATA.USE_QUANTITY <= 1.0
                else ValueError("Quantity should be between 0 and 1")
            )
            if self.training
            else 1.0
        )
        if self.config.DATA.RESAMPLER == "sox":
            self.sox_tfm = sox.Transformer()

        # The number of time frames in a segment
        # If the length of the audio is more than the segment length, it will be trimmed
        # else, we will pad the audio with zeros
        # The number of frames is calculated as the segment length * original `.wav` sample rate
        # For 48kHz target: 1.705 * 48000 = 81840 samples
        # For 16kHz target: 2.555 * 48000 * (16000 / 48000) = 40880 samples, downsample is applied after loading the audio
        self.num_frames = int(
            self.config.DATA.SEGMENT * self.config.DATA.FLAC2WAV.SRC_SR
        )

        # Initialize the sample IDs
        self._sample_ids = []
        # Set the sample IDs file path
        self.sample_ids_file = os.path.join(
            self._path, f"sample_ids_{'train' if self.training else 'test'}.json"
        )
        # Load the sample IDs
        self._load_sample_ids()

    def _flac2wav(self):
        """
        As the dataset is downloaded in flac format, we need to convert it to wav.
        """
        # Make sure the destination folder exists
        os.makedirs(self.config.DATA.DATA_PATH, exist_ok=True)
        # Download the dataset
        dataset = datasets.VCTK_092(
            root=self.config.DATA.DATA_PATH,
            mic_id=self.config.DATA.MIC_ID,
            download=True,
        )

        # Check if the timestamps file exists
        if not os.path.isfile(self.config.DATA.FLAC2WAV.TIMESTAMPS):

            raise RuntimeError(
                "Timestamps file not found. Please run the `git submodule update --init --recursive` command to download the timestamps file."
            )

        # Print the message
        self.logger.info("Converting flac to wav...")
        trimmed_folder = os.path.join(
            self.config.DATA.DATA_PATH, self.config.DATA.FLAC2WAV.DST_PATH
        )
        # Read the timestamps txt as pandas dataframe
        timestamps = pd.read_csv(
            self.config.DATA.FLAC2WAV.TIMESTAMPS, sep=" ", header=None
        )
        # Set the column names
        timestamps.columns = ["filename", "start", "end"]
        # Convert seconds to samples with the sample rate
        timestamps["start"] = timestamps["start"] * self.config.DATA.FLAC2WAV.SRC_SR
        timestamps["end"] = timestamps["end"] * self.config.DATA.FLAC2WAV.SRC_SR
        # Change the type to int
        timestamps["start"] = timestamps["start"].astype(int)
        timestamps["end"] = timestamps["end"].astype(int)

        # Iterate over the dataset
        for waveform, sample_rate, transcript, speaker_id, utterance_id in tqdm(
            dataset, desc="Converting flac to wav"
        ):
            # Skip the speaker_id p280 and p315
            if speaker_id == "p280" or speaker_id == "p315":
                continue
            # Combine speaker_id and utterance_id, use it to match the timestamps
            flac_name = f"{speaker_id}_{utterance_id}"
            # Get the if from the timestamps dataframe
            flac_timestamps = timestamps[timestamps["filename"] == flac_name]
            # Set the destination file folder
            dest_folder = os.path.join(trimmed_folder, f"{speaker_id}")
            # Set the destination file path
            dest_path = os.path.join(dest_folder, f"{flac_name}.wav")
            # Make sure the output directory exists, if not, create it
            os.makedirs(dest_folder, exist_ok=True)
            # Check if the destination file already exists
            if os.path.exists(dest_path):
                continue
            # If the flac file is not in the timestamps, directly save the waveform
            if flac_timestamps.empty:
                torchaudio.save(dest_path, waveform, sample_rate)
                continue
            # Trim the waveform tensor based on the start and end of timestamps
            start = flac_timestamps["start"].values[0]
            end = flac_timestamps["end"].values[0]
            trimmed_waveform = waveform[:, start:end]
            # Save the trimmed waveform
            torchaudio.save(dest_path, trimmed_waveform, sample_rate)

        # Print the message
        self.logger.info(
            f"Finished converting flac to ({len(dataset)}) wav files to {trimmed_folder}."
        )
        # Delete the dataset
        del dataset

    def _load_sample_ids(self):
        """
        Load the sample IDs from the file if it exists.
        """
        if os.path.isfile(self.sample_ids_file):
            # Print the message
            self.logger.info(
                f"Loading {'training' if self.training else 'testing'} sample IDs..."
            )
            # Load the sample IDs from the file
            self._load_sample_ids_from_file()
        else:
            # Print the message
            self.logger.info(
                "Can't find sample IDs file. Parsing the folder structure..."
            )
            # Parse the folder structure and create the sample IDs
            self._parse_folder_and_create_sample_ids()
            # Load the sample IDs from the file
            self._load_sample_ids_from_file()

    def _parse_folder_and_create_sample_ids(self):
        """
        Parse the folder structure and create the sample IDs.
        """
        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._audio_dir))
        # Split the training and test speakers
        if self.training:
            self._speaker_ids = self._speaker_ids[: self.training_test_split[0]]
        else:
            self._speaker_ids = self._speaker_ids[self.training_test_split[0] :]

        self.logger.info(
            f"Number of speakers for {'training' if self.training else 'testing'}: {len(self._speaker_ids)}"
        )

        # Get _sample_ids
        for speaker_id in self._speaker_ids:
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(
                f for f in os.listdir(utterance_dir) if f.endswith(".txt")
            ):
                utterance_id = os.path.splitext(utterance_file)[0]
                # Check if the audio file exists, the txt might not have the corresponding audio file
                audio_path = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}{self._audio_ext}",
                )
                # Skip txt file that doesn't have the corresponding audio file
                if not os.path.isfile(audio_path):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

        self.logger.info(f"Number of samples: {len(self._sample_ids)}")

        # Save the generated sample IDs
        with open(self.sample_ids_file, "w") as f:
            json.dump(self._sample_ids, f)

    def _load_sample_ids_from_file(self):
        """
        Load the sample IDs from the file if it exists.
        """
        with open(self.sample_ids_file, "r") as f:
            self._sample_ids = json.load(f)
            # Load 100% of the sample IDs for testing, otherwise load the specified quantity
            quantity = self.quantity if self.training else 1.0
            num_smaples = len(self._sample_ids)
            loaded_samples = (
                int(num_smaples * quantity) if self.training else num_smaples
            )
            self.logger.info(
                f"Loading {quantity}% of the sample IDs ({loaded_samples} of {num_smaples})..."
            )

            # Randomly shuffle the sample IDs
            random.shuffle(self._sample_ids)
            # Load the specified quantity of the sample IDs
            self._sample_ids = self._sample_ids[:loaded_samples]

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

    def _load_audio(self, file_path, num_frames) -> Tuple[torch.Tensor | int]:
        if self.training:
            audio, sr = torchaudio.load(file_path, num_frames=num_frames)
            target_sr = self.config.DATA.TARGET_SR
        else:
            audio, sr = torchaudio.load(file_path)
            # Check if the sample rate of the original audio is the same as the target sample rate
            target_sr = int(self.config.TAG.split("_")[1])

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
        # For the training, the audio will be padded to the same length
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

        if self.config.DEBUG:
            # Save the spectrogram of the audio
            mag, phase = wav2spectro(
                audio,
                n_fft=self.config.DATA.STFT.N_FFT,
                hop_length=self.config.DATA.STFT.HOP_LENGTH,
                win_length=self.config.DATA.STFT.WIN_LENGTH,
                spectro_scale="log2",
            )
            # Save the spectrogram
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            img = axs[0].imshow(
                mag.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=-15,
            )
            plt.colorbar(img, ax=axs[0])
            img = axs[1].imshow(
                phase.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            plt.colorbar(img, ax=axs[1])
            plt.savefig("./debug/spectrogram_orig.png")
            plt.close()

        return audio, sr, pad_length

    def _get_io_pair(
        self, output: torch.Tensor, sr: int, pad_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and output pair for the model

        Args:
            output (torch.Tensor): The output waveform (target waveform)
            sr (int): The sample rate of the output waveform

        Returns:
            input (torch.Tensor): The waveform to be used as input
            output (torch.Tensor): The target waveform
            highcut_in_stft (int): The highcut frequency in the STFT for operating low-frequency replacement and calculating the LSD-HF and LSD-LF
        """
        if self.training:
            if self.config.DATA.WEIGHTED_SR.ENABLE:
                # Set the weight to randomly choose the SR
                # The lower SR has higher probability to be chosen
                sr_ranges = self.config.DATA.WEIGHTED_SR.RANGES
                sr_weights = self.config.DATA.WEIGHTED_SR.WEIGHTS
                range_idx = np.random.choice(len(sr_ranges), p=sr_weights)
                min_sr, max_sr = sr_ranges[range_idx]
                sr_input = random.randint(min_sr, max_sr)
            else:
                # Uniformly choose an integer from min to max in the list
                sr_input = random.randint(
                    self.config.DATA.RANDOM_RESAMPLE[0],
                    self.config.DATA.RANDOM_RESAMPLE[-1],
                )
        else:
            sr_input = int(self.config.TAG.split("_")[0])

        if sr_input != sr:
            if not self.training:
                # Random choose the low pass filter
                filter_ = random.choice(self.config.DATA.LPF.LPF_TRAIN)
                # Apply the low pass filter
                input = lowpass(
                    output, int(sr_input * 0.5), filter_, self.config.DATA.TARGET_SR
                )
            else:
                filter_ = self.config.DATA.LPF.LPF_TEST[0]
                # Apply the low pass filter
                input = lowpass(
                    output, int(sr_input * 0.5), filter_, self.config.DATA.TARGET_SR
                )
            # Downsample the audio
            input = self.resample_audio(
                output, sr, sr_input, self.config.DATA.RESAMPLER
            )
            # Upsample the audio
            input = self.resample_audio(input, sr_input, sr, self.config.DATA.RESAMPLER)
            # Align the waveform length
            input = align_waveform(input, output)
        else:
            input = output

        # Highcut frequency = int((1 + n_fft // 2) * (sr_input // sr_target))
        highcut_in_stft = int(
            (1 + self.config.DATA.STFT.N_FFT // 2)
            * (sr_input / self.config.DATA.TARGET_SR)
        )
        if self.config.DATA.RESAMPLER == "sox":
            # The noise is removed by the sox, we need to add it back
            # Otherwise, the phase will be empty
            if pad_length:
                white_noise = (
                    torch.randn((pad_length)) * self.config.DATA.PAD_WHITENOISE
                ).unsqueeze(0)
                # Replace the padded part with white noise
                input[:, -pad_length:] = white_noise

        if self.config.DEBUG:
            # Save the spectrogram of the audio
            mag, phase = wav2spectro(
                input,
                n_fft=self.config.DATA.STFT.N_FFT,
                hop_length=self.config.DATA.STFT.HOP_LENGTH,
                win_length=self.config.DATA.STFT.WIN_LENGTH,
                spectro_scale="log2",
            )
            # Save the spectrogram
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            img = axs[0].imshow(
                mag.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=-15,
            )
            plt.colorbar(img, ax=axs[0])
            img = axs[1].imshow(
                phase.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            plt.colorbar(img, ax=axs[1])
            plt.savefig("./debug/spectrogram_down.png")
            plt.close()

        return input, output, highcut_in_stft

    def _load_sample(
        self, speaker_id: str, utterance_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the audio and transcript for a given speaker and utterance ID.
        """
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}{self._audio_ext}",
        )
        # Read audio file
        audio, sr, pad_length = self._load_audio(
            audio_path, num_frames=self.num_frames or -1
        )
        input, output, highcut = self._get_io_pair(audio, sr, pad_length)

        return (
            input,
            output,
            highcut,
            f"{speaker_id}_{utterance_id}{self._audio_ext}",
            pad_length,
        )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id)

    def __len__(self) -> int:
        return len(self._sample_ids)


def align_waveform(
    waveform_resampled: torch.Tensor, waveform: torch.Tensor
) -> torch.Tensor:
    if waveform_resampled.shape[1] < waveform.shape[1]:
        # Pad the resampled waveform if it is shorter than the original waveform
        waveform_resampled = F.pad(
            waveform_resampled, (0, waveform.shape[1] - waveform_resampled.shape[1])
        )
    elif waveform_resampled.shape[1] > waveform.shape[1]:
        # Trim the resampled waveform if it is longer than the original waveform
        waveform_resampled = waveform_resampled[:, : waveform.shape[1]]

    return waveform_resampled


def lowpass(audio, highcut, filter_=("cheby1", 8), sr=48000):
    """
    Apply low pass filter to the audio.

    REF: On Filter Generalization for Music Bandwidth Extension Using Deep Neural Networks
    URL: https://github.com/serkansulun/deep-music-enhancer/blob/master/src/utils.py

    Args:
        audio (torch.Tensor): The input audio
        highcut (int): The cutoff frequency (it's half of the input sample rate)
        filter_ (tuple, optional): The filter type and order. Defaults to ("cheby1", 8).
        sr (int, optional): The target sample rate. Defaults to 48000.
    """
    nyq = sr / 2
    highcut /= nyq

    if filter_[0] == "butter":
        sos = butter(filter_[1], highcut, btype="lowpass", output="sos")
    elif filter_[0] == "cheby1":
        sos = cheby1(filter_[1], 0.05, highcut, btype="lowpass", output="sos")
    elif filter_[0] == "bessel":
        sos = bessel(filter_[1], highcut, norm="mag", btype="lowpass", output="sos")
    elif filter_[0] == "ellip":
        sos = ellip(filter_[1], 0.05, 20, highcut, btype="lowpass", output="sos")

    # Apply the filter
    input = sosfiltfilt(sos, audio)

    return torch.tensor(input.copy(), dtype=torch.float32)
