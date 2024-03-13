from typing import Tuple
import torch
import torchaudio.datasets as datasets
import random
import json
import os
import sys
from tqdm import tqdm

try:
    from base import BaseDataLoader
    import data_loader.preprocessing as preprocessing
except:
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    from utils import ensure_dir
    from base.base_data_loader import BaseDataLoader
    import data_loader.preprocessing as preprocessing
    import data_loader.postprocessing as postprocessing


class VCTKDataLoader(BaseDataLoader):
    """
    VCTK_092 data loading
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        random_resample=[8000],
        **kwargs,
    ):
        # Set up data directory
        self.data_dir = data_dir
        self.random_resample = random_resample
        # Download VCTK_092 dataset
        # The data returns a tuple of the form: waveform, sample rate, transcript, speaker id and utterance id
        self.dataset = CustomVCTK_092(
            root=self.data_dir, random_resample=self.random_resample, **kwargs
        )
        # Print the total number of samples
        print(f"Total number of samples: {len(self.dataset)}")
        # Set up the data loader
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class CustomVCTK_092(datasets.VCTK_092):
    """
    Inherit the VCTK_092 dataset and make custom data processing pipeline.

    1. Assume you've run the `trim_flac2wav.py` script to trim the silence and save the trimmed wav files.
    2. The trimmed wav files are in the `./data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav` directory.
    3. Since we only use mic2, the file name is in the format of `p225_001.wav` without the mic_id.

    So, we modify the self._audio_dir and remove self._mic_id = mic_id.

    Args:
        root (str): Root directory of dataset where `VCTK-Corpus-0.92` folder exists.
        audio_ext (str, optional): The extension of the audio files. Defaults to ".wav".
        random_resample (list, optional): List of target sample rates to choose from. Defaults to [8000].
        **kwargs: Additional arguments for the VCTK_092 class.
    """

    def __init__(self, root, audio_ext=".wav", random_resample=[8000], **kwargs):
        super().__init__(root)
        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed_wav")
        self._audio_ext = audio_ext
        self._random_resample = random_resample
        self._sample_ids = []
        self.chunking_params = kwargs.get("chunking_params", None)
        # Check if the trimmed wav files exist
        if not os.path.isdir(self._audio_dir):
            raise RuntimeError(
                "Dataset not found. Please run data/trim_flac2wav.py to download and trim the silence first."
            )

        self.sample_ids_file = os.path.join(self._path, "sample_ids.json")

        if os.path.isfile(self.sample_ids_file):
            # Print the message
            print("Loading sample IDs from file...")
            # Load the sample IDs from the file
            self._load_sample_ids_from_file()
        else:
            # Print the message
            print("Can't find sample IDs file. Parsing the folder structure...")
            # Parse the folder structure and create the sample IDs
            self._parse_folder_and_create_sample_ids()

    def _parse_folder_and_create_sample_ids(self):
        """
        Parse the folder structure and create the sample IDs.
        """
        # Extracting speaker IDs from the folder structure
        # Note: Some of speakers has been removed by VCTK_092 class while running `trim_flac2wav.py`
        self._speaker_ids = sorted(os.listdir(self._audio_dir))

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
                if not os.path.isfile(audio_path):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

        # Save the generated sample IDs
        with open(self.sample_ids_file, "w") as f:
            json.dump(self._sample_ids, f)

    def _load_sample_ids_from_file(self):
        """
        Load the sample IDs from the file if it exists.
        """
        with open(self.sample_ids_file, "r") as f:
            self._sample_ids = json.load(f)
            # TODO: Set percentage of the dataset in the config file
            # Only load the 10% of the dataset for debugging
            self._sample_ids = self._sample_ids[: int(len(self._sample_ids) * 0.10)]

    def _load_audio(self, file_path) -> Tuple[torch.Tensor | int]:
        return super()._load_audio(file_path)

    def _load_sample(
        self, speaker_id: str, utterance_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the audio and transcript for a given speaker and utterance ID.
        """
        # Get the transcript and audio file path
        # transcript_path = os.path.join(
        #     self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}{self._audio_ext}",
        )
        # Read text (Optional)
        # transcript = self._load_text(transcript_path)
        # Read audio file
        waveform, sr = self._load_audio(audio_path)
        # TODO: Return the padding length for post-processing
        # Process the waveform with the audio processing pipeline
        mag_phase_pair_x, mag_phase_pair_y = self.get_io_pairs(waveform, sr)

        return mag_phase_pair_x, mag_phase_pair_y

    def get_io_pairs(self, waveform: torch.Tensor, sr_org: int) -> torch.Tensor:
        """
        Get the input-output pairs for the audio processing pipeline.
        1. Crop or pad the waveform to a fixed length (consistent shapes within batches for efficient computation)
        2. Get magnitude and phase of the original audio
        3. Normalize the audio (Optional)
        4. Apply low pass filter to avoid aliasing
        5. Downsample the audio to a lower sample rate (simulating a low resolution audio)
        6. Upsample the audio to a higher sample rate (unifying the input shape)
        7. Apply low pass filter to remove the artifacts from the resampling
        8. Chunk the audio into smaller fixed-length segments (to fit the model input shape)
        9. Get magnitude and phase of the chunked low resolution audio
        10. Return the x-y pair of magnitude and phase

        Args:
            waveform (Tensor): The input audio waveform
            sr (int): The sample rate of the audio waveform

        Returns:
            Tensor: The input-output pair of magnitude and phase
        """
        # List of target sample rates to choose from
        target_sample_rates = self._random_resample
        # Normalize the audio
        # waveform = waveform / waveform.abs().max() * 0.95 * 32768
        # Apply the audio preprocessing pipeline
        sr_new = random.choice(target_sample_rates)
        # Crop or pad the waveform to a fixed length
        waveform = preprocessing.crop_or_pad_waveform(waveform)
        # Get magnitude and phase of the original audio
        mag_phase_pair_y = self._get_mag_phase(waveform, chunk_wave=True)

        # Preprocess the audio
        # Apply low pass filter to avoid aliasing
        waveform = preprocessing.low_pass_filter(waveform, sr_org, sr_new)
        # Downsample the audio to a lower sample rate
        waveform = preprocessing.resample_audio(waveform, sr_org, sr_new)
        # Upsample the audio to a higher sample rate
        waveform = preprocessing.resample_audio(waveform, sr_new, sr_org)
        # Remove the artifacts from the resampling
        waveform = preprocessing.low_pass_filter(waveform, sr_org, sr_new)
        # Get magnitude and phase of the preprocessed audio
        mag_phase_pair_x = self._get_mag_phase(waveform, chunk_wave=True)

        return mag_phase_pair_x, mag_phase_pair_y

    def _get_mag_phase(
        self, waveform: torch.Tensor, chunk_wave: bool = True
    ) -> torch.Tensor:
        """
        Compute the magnitude and phase of the audio in the time-frequency domain.

        Args:
            waveform (Tensor): The input audio waveform
            chunk_wave (bool): Whether to chunk the audio into smaller fixed-length segments

        Returns:
            Tensor: The magnitude and phase of the audio in the time-frequency domain
        """
        if chunk_wave:
            # Size of each audio chunk
            chunk_size = self.chunking_params["chunk_size"]
            # Overlap size between chunks
            overlap = int(chunk_size * self.chunking_params["overlap"])
            # Chunk the audio into smaller fixed-length segments
            # chunks is torch.stack() with shape (num_chunks, chunk_size)
            chunks, padding_length = preprocessing.cut2chunks(
                waveform=waveform,
                chunk_size=chunk_size,
                overlap=overlap,
                return_padding_length=True,
            )
            # Compute STFT for each segment and convert to magnitude and phase
            mag = []
            phase = []
            for chunk_y in chunks:
                mag_chunk, phase_chunk = preprocessing.get_mag_phase(
                    chunk_y, chunk_wave=True
                )
                mag.append(mag_chunk)
                phase.append(phase_chunk)

            # Make mag and phase to tensors, shape (num_chunks, 1, num_frequencies, num_frames)
            mag = torch.stack(mag)
            phase = torch.stack(phase)
            # Remove the dimension of 1
            mag = mag.squeeze(1)
            phase = phase.squeeze(1)
            # Make mag and phase to a pair, shape (2, num_chunks, num_frequencies, num_frames)
            mag_phase_pair = torch.stack((mag, phase))
        else:
            # Compute STFT for the waveform and convert to magnitude and phase
            mag, phase = preprocessing.get_mag_phase(waveform, chunk_wave=False)
            # Make mag and phase to a pair, shape (2, num_frequencies, num_frames)
            mag_phase_pair = torch.stack((mag, phase))

        # Return the magnitude and phase in tensors
        return mag_phase_pair

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Magnitude of the audio in the time-frequency domain
            Tensor:
                Phase of the audio in the time-frequency domain
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id)

    def __len__(self) -> int:
        return len(self._sample_ids)


# Debugging
if __name__ == "__main__":
    # Load packages that only used for debugging
    import time
    import torchaudio

    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open("./config.json") as f:
        config = json.load(f)

    # Ensure the output directory exists
    output_dir = "./output/dev/data_loader"
    ensure_dir(output_dir)

    print("Loading the VCTK_092 dataset...")

    # Set up the data loader
    data_loader = VCTKDataLoader(
        data_dir=config["data_loader"]["args"]["data_dir"],
        batch_size=128,
        num_workers=4,
        validation_split=0.1,
        chunking_params={"chunk_size": 10160, "overlap": 0.5},
    )

    # Iterate over the data loader with tqdm
    for batch_idx, data in enumerate(tqdm(data_loader)):
        # Print the batch index and the data
        print(
            f"batch_idx: {batch_idx}, len(data): {len(data)}, data[0].shape (x): {data[0].shape}, data[1].shape (y): {data[1].shape}"
        )
        # Check the data
        # The shape of x is torch.Size([128 (batch_size), 2 (mag and phase), 15 (chunks), 513 (frequency bins), 101 (frames)])
        # The shape of chunked y is torch.Size([128 (batch_size), 2 (mag and phase), 15 (chunks), 513 (frequency bins), 101 (frames)])
        # The shape of not chunked y is torch.Size([128 (batch_size), 2 (mag and phase), 513 (frequency bins), 256 (frames)])
        x, y = data
        x_mag, x_phase = x[0]
        y_mag, y_phase = y[0]
        print(f"x_mag.shape: {x_mag.shape}, x_phase.shape: {x_phase.shape}")
        print(f"y_mag.shape: {y_mag.shape}, y_phase.shape: {y_phase.shape}")

        # Post-processing
        # Reconstruct the chunked waveform of magnitude and phase spectrograms
        reconstructed_waveform_x = postprocessing.reconstruct_from_stft_chunks(
            mag=x_mag.unsqueeze(0), phase=x_phase.unsqueeze(0), crop=True
        )
        # Save the reconstructed waveform
        torchaudio.save(
            f"{output_dir}/reconstructed_waveform_x_{timestr}.wav",
            reconstructed_waveform_x,
            48000,
        )
        # Print the reconstructed waveform shape
        print(f"Reconstructed waveform x shape: {reconstructed_waveform_x.shape}")

        # Reconstruct the full waveform of magnitude and phase spectrograms
        # Test for not chunked original waveform
        # reconstructed_waveform_y = postprocessing.reconstruct_from_stft(
        #     mag=y_mag, phase=y_phase)
        # Test for chunked original waveform
        reconstructed_waveform_y = postprocessing.reconstruct_from_stft_chunks(
            mag=y_mag.unsqueeze(0), phase=y_phase.unsqueeze(0), crop=True
        )
        # Save the reconstructed waveform
        torchaudio.save(
            f"{output_dir}/reconstructed_waveform_y_{timestr}.wav",
            reconstructed_waveform_y,
            48000,
        )
        # Print the reconstructed waveform shape
        print(f"Reconstructed waveform y shape: {reconstructed_waveform_y.shape}")
        break
