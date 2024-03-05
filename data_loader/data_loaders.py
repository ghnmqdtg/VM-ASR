import numpy as np
from typing import Tuple
import torch
import torchaudio
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
import prepocessing

try:
    from base import BaseDataLoader
except:
    import os
    import sys
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Now you can import BaseDataLoader
    from base.base_data_loader import BaseDataLoader

    # Load the `./config.json` as config
    import json
    with open('config.json') as f:
        config = json.load(f)


class VCTKDataLoader(BaseDataLoader):
    """
    VCTK_092 data loading
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, random_resample=[8000], length=32768):
        # Set up data directory
        self.data_dir = data_dir
        self.random_resample = random_resample
        self.length = length
        # Download VCTK_092 dataset
        # The data returns a tuple of the form: waveform, sample rate, transcript, speaker id and utterance id
        self.dataset = CustomVCTK_092(
            root=self.data_dir, random_resample=self.random_resample, length=self.length)
        # Print the total number of samples
        print(f"Total number of samples: {len(self.dataset)}")
        # Set up the data loader
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)


class CustomVCTK_092(torchaudio.datasets.VCTK_092):
    """
    Inherit the VCTK_092 dataset and make custom data processing pipeline.

    1. Assume you've run the `trim_flac2wav.py` script to trim the silence and save the trimmed wav files.
    2. The trimmed wav files are in the `./data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav` directory.
    3. Since we only use mic2, the file name is in the format of `p225_001.wav` without the mic_id.

    So, we modify the self._audio_dir and remove self._mic_id = mic_id.

    Args:
        root (str): Root directory of dataset where `VCTK-Corpus-0.92` folder exists.
    """

    def __init__(self, root, audio_ext=".wav", random_resample=[8000], length=32768, **kwargs):
        super().__init__(root, **kwargs)
        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed_wav")
        self._audio_ext = audio_ext
        self._random_resample = random_resample
        self._length = length
        # Check if the trimmed wav files exist
        if not os.path.isdir(self._audio_dir):
            raise RuntimeError(
                "Dataset not found. Please run data/trim_flac2wav.py to download and trim the silence first.")
        # Extracting speaker IDs from the folder structure
        # Note: Some of speakers has been removed by VCTK_092 class while running `trim_flac2wav.py`
        self._speaker_ids = sorted(os.listdir(self._audio_dir))
        self._sample_ids = []

        # Get _sample_ids
        for speaker_id in self._speaker_ids:
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(".txt")):
                utterance_id = os.path.splitext(utterance_file)[0]
                # Check if the audio file exists, the txt might not have the corresponding audio file
                audio_path = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}{self._audio_ext}",
                )
                if not os.path.isfile(audio_path):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_audio(self, file_path) -> Tuple[torch.Tensor | int]:
        return super()._load_audio(file_path)

    def _load_sample(self, speaker_id: str, utterance_id: str) -> Tuple[torch.Tensor, int, str, str]:
        transcript_path = os.path.join(
            self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}{self._audio_ext}",
        )
        # Read text
        transcript = self._load_text(transcript_path)
        # Read audio file
        waveform, sample_rate = self._load_audio(audio_path)
        # Equal length
        waveform = self.equal_length(waveform, sample_rate)
        # Process the waveform with the audio processing pipeline
        mel_lr, mel_hr = self._process_audio(waveform, sample_rate)

        # return (mel_lr, mel_hr, sample_rate, transcript, speaker_id, utterance_id)
        return (mel_lr, mel_hr)
    
    def equal_length(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Pad or crop the waveform to a fixed length as required.

        Args:
            waveform (torch.Tensor): Waveform
            sample_rate (int): Sample rate

        Returns:
            torch.Tensor: Padded or cropped waveform
        """
        # print(f'Shape of waveform: {waveform.shape}, target length: {self._length}')
        # If the waveform is shorter than the required length, pad it
        if waveform.shape[1] < self._length:
            pad_length = self._length - waveform.shape[1]
            # If the phase is training or validation, pad the waveform randomly
            # If the phase is testing, pad the waveform from the beginning
            r = random.randint(0, pad_length)
            # Pad the waveform with zeros to the left and right
            # Left: random length between 0 and pad_length, Right: pad_length - r
            waveform = F.pad(waveform, (r, pad_length - r), mode='constant', value=0)
            # print(f"Pad length: {pad_length}, Random length: {r}")
        else:
            # If the waveform is longer than the required length, crop it randomly from the beginning
            start = random.randint(0, waveform.shape[1] - self._length)
            # Crop the waveform from start to start + length (fixed length)
            waveform = waveform[:, start:start + self._length]
            # print(f"Crop to length: {self._length}, Start: {start}")
        
        # print(f'New shape of padded or cropped waveform: {waveform.shape}')
        return waveform

    def _process_audio(self, waveform_hr: torch.Tensor, source_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert high resolution waveform to low resolution mel spectrogram and high resolution mel spectrogram.

        Args:
            waveform_hr (torch.Tensor): High resolution waveform
            source_sr (int): Sample rate of the source waveform (e.g. 48000 Hz)
        """
        # Randomly select a sample rate for the low resolution waveform
        target_sr = random.choice(self._random_resample)
        # Resample the waveform
        waveform_lr = torchaudio.transforms.Resample(
            source_sr, target_sr)(waveform_hr)
        # print(f'Chosen target sample rate: {target_sr}',
        #       f'Original sample rate: {source_sr}')
        # Convert the waveform to mel spectrogram
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 80
        mel_lr = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )(waveform_lr)
        mel_hr = torchaudio.transforms.MelSpectrogram(
            sample_rate=source_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )(waveform_hr)
        # Print the shape of the mel spectrogram
        # Reshape the mel spectrogram
        mel_lr = mel_lr.squeeze(0)
        mel_hr = mel_hr.squeeze(0)
        # Print the shape of the mel spectrogram
        # print(
        #     f"Low resolution mel shape: {mel_lr.shape}, High resolution mel shape: {mel_hr.shape}")

        # Plot the mel spectrogram
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.title("Low resolution mel spectrogram")
        # plt.imshow(mel_lr.log2().detach().numpy()[0], aspect="auto", origin="lower")
        # plt.subplot(2, 1, 2)
        # plt.title("High resolution mel spectrogram")
        # plt.imshow(mel_hr.log2().detach().numpy()[0], aspect="auto", origin="lower")
        # # Save the plot to a file
        # plt.savefig("mel_spectrogram.png")

        # Return the mel spectrogram
        return mel_lr, mel_hr

    def __getitem__(self, n: int) -> Tuple[torch.Tensor | int | str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Mel spectrogram (low resolution)
            Tensor:
                Mel spectrogram (high resolution)
            int:
                Sample rate
            str:
                Transcript
            str:
                Speaker ID
            str:
                Utterance ID
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id)

    def __len__(self) -> int:
        return len(self._sample_ids)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# Debugging
if __name__ == '__main__':
    # Set up the data loader
    data_loader = VCTKDataLoader(
        data_dir=config['data_loader']['args']['data_dir'], batch_size=128, num_workers=2, validation_split=0.1)
    # data_loader = MnistDataLoader(data_dir=config['data_loader']['args']['data_dir'], batch_size=16, num_workers=2, validation_split=0.1)

    # Iterate over the data loader
    for batch_idx, data in enumerate(data_loader):
        # Print the batch index and the data
        print(f'data: {data}')
        break