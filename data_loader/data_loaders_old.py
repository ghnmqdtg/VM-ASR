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
        num_workers=1,
        quantity=1.0,
        training_test_split=[100, 8],
        validation_split=0.0,
        training=True,
        **kwargs,
    ):
        # Set up data directory
        quantity = (
            quantity
            if quantity > 0.0 and quantity <= 1.0
            else sys.exit("quantity of loading data should be in the range of (0, 1].")
        )
        # Download VCTK_092 dataset
        # The data returns a tuple of the form: waveform, sample rate, transcript, speaker id and utterance id
        self.dataset = CustomVCTK_092(
            root=data_dir,
            training_test_split=training_test_split,
            training=training,
            quantity=quantity,
            **kwargs,
        )

        # Data shape is used for model summary
        shape = [int(x) for x in self.dataset[0][0].shape]
        # Get only one chunk
        shape[1] = 1
        # Set the data shape
        self.data_shape = tuple(shape)

        # Print the total number of samples
        print(f"Total number of samples: {len(self.dataset)}")
        # Set up the data loader
        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function to deal with the audio data with different lengths.

        Args:
            batch (list): List of tensors of in the shape of (batch_size, 2 (mag or phase), num_chunks, num_frequencies, num_frames)
        """
        # Get the batch size
        batch_size = len(batch)
        # Split the batches into x and y
        batch_split = list(zip(*batch))
        # Initialize tensor to store the new batches for x and y
        new_batch_x = []
        new_batch_y = []
        # Get the max length of chunks
        max_chunk_length = max([x.size(1) for x in batch_split[0]])
        # Loop over the batch and pad the chunks
        for i in range(batch_size):
            # Get the shape of the chunk (x and y are the same shape)
            data_shape = batch_split[0][i].shape

            if data_shape[1] < max_chunk_length:
                # Chunk of zeros for padding
                padding_chunk = torch.zeros(
                    2,
                    max_chunk_length - data_shape[1],
                    data_shape[2],
                    data_shape[3],
                )
                # Concatenate the padding chunk to new batch for x and y
                new_batch_x.append(torch.cat((batch_split[0][i], padding_chunk), dim=1))
                new_batch_y.append(torch.cat((batch_split[1][i], padding_chunk), dim=1))
            else:
                new_batch_x.append(batch_split[0][i])
                new_batch_y.append(batch_split[1][i])

        # Stack the new batch for x and y
        new_batch_x = torch.stack(new_batch_x)
        new_batch_y = torch.stack(new_batch_y)

        return (
            new_batch_x,
            new_batch_y,
            torch.tensor(batch_split[2]),
            torch.tensor(batch_split[3]),
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

    def __init__(
        self,
        root,
        audio_ext=".wav",
        training_test_split=[100, 8],
        training=True,
        quantity=1.0,
        **kwargs,
    ):
        super().__init__(root)
        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed_wav")
        self._audio_ext = audio_ext
        self.training_test_split = training_test_split
        self.training = training
        self.quantity = quantity
        self._sample_ids = []
        self.random_resample = kwargs.get("random_resample", [8000])
        self.length = kwargs.get("length", 121890)
        self.white_noise = kwargs.get("white_noise", 0)
        self.stft_enabled = kwargs.get("stft_enabled", True)
        self.chunking_enabled = kwargs.get("chunking_enabled", True)
        self.random_lpf = kwargs.get("random_lpf", False)
        self.scale = kwargs.get("scale", "log")
        self.chunking_params = kwargs.get("chunking_params", None)
        self.stft_params = kwargs.get("stft_params", None)
        # Check if the trimmed wav files exist
        if not os.path.isdir(self._audio_dir):
            raise RuntimeError(
                "Dataset not found. Please run data/trim_flac2wav.py to download and trim the silence first."
            )

        self.sample_ids_file = os.path.join(
            self._path, f"sample_ids_{'train' if self.training else 'test'}.json"
        )
        # Load the sample IDs
        self._load_sample_ids()

    def _load_sample_ids(self):
        """
        Load the sample IDs from the file if it exists.
        """
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
            # Load the sample IDs from the file
            self._load_sample_ids_from_file()

    def _parse_folder_and_create_sample_ids(self):
        """
        Parse the folder structure and create the sample IDs.
        """
        # Extracting speaker IDs from the folder structure
        # Note: Some of speakers has been removed by VCTK_092 class while running `trim_flac2wav.py`
        self._speaker_ids = sorted(os.listdir(self._audio_dir))
        # Split the training and test speakers
        if self.training:
            self._speaker_ids = self._speaker_ids[: self.training_test_split[0]]
            print(f"Number of speakers: {len(self._speaker_ids)}")
        else:
            self._speaker_ids = self._speaker_ids[self.training_test_split[0] :]
            print(f"Number of speakers: {len(self._speaker_ids)}")

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
            num_smaples = len(self._sample_ids)
            loaded_samples = int(num_smaples * self.quantity)
            print(
                f"Loading {self.quantity}% of the sample IDs ({loaded_samples} of {num_smaples})..."
            )
            # Set the seed on data loading for reproducibility
            random.seed(9527)
            # Randomly shuffle the sample IDs
            random.shuffle(self._sample_ids)
            # Load the specified quantity of the sample IDs
            self._sample_ids = self._sample_ids[:loaded_samples]

    def get_io_pairs(self, waveform_org: torch.Tensor, sr_org: int) -> torch.Tensor:
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
            waveform_org (Tensor): The input audio waveform
            sr (int): The sample rate of the audio waveform

        Returns:
            x (Tensor): The magnitude and phase of the input in the time-frequency domain
            y (Tensor): The magnitude and phase of the original audio in the time-frequency domain
            padding_length (int): The length of the padding for the last chunk
            hf (int): The highcut frequency for the low pass filter
        """
        # List of target sample rates to choose from
        # sr_new = random.choice(self.random_resample)
        # * TEST: Uniformly choose an integer from min to max in the list
        sr_new = random.randint(self.random_resample[0], self.random_resample[-1])
        # Highcut frequency = int((1 + n_fft // 2) * (sr_new // sr_org))
        # We only use it for computing the LSD
        hf = int(1 + (self.stft_params["full"]["n_fft"] // 2) * (sr_new / sr_org))
        # Crop or pad the waveform to a fixed length
        waveform_org = preprocessing.crop_or_pad_waveform(
            waveform_org, {"length": self.length, "white_noise": self.white_noise}
        )
        if sr_new != sr_org:
            # Preprocess the audio
            if self.random_lpf:
                order = random.randint(1, 11)
                ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
            else:
                order = 6
                ripple = 1e-3

            # Apply low pass filter to avoid aliasing
            waveform = preprocessing.low_pass_filter(
                waveform_org, sr_org, sr_new, order=order, ripple=ripple
            )
            # Downsample the audio to a lower sample rate
            waveform = preprocessing.resample_audio(waveform, sr_org, sr_new)
            # Upsample the audio to a higher sample rate
            waveform = preprocessing.resample_audio(waveform, sr_new, sr_org)
            # Remove the artifacts from the resampling
            waveform = preprocessing.low_pass_filter(
                waveform, sr_org, sr_new, order=order, ripple=ripple
            )
            # Align the length of the waveform
            waveform = preprocessing.align_waveform(waveform, waveform_org)
        else:
            waveform = waveform_org

        # Get data and target pairs
        if self.stft_enabled:
            # Return the magnitude and phase of the audio in the time-frequency domain
            # Shape: (2 (mag and phase), num_chunks, num_frequencies, num_frames)
            x, padding_length = self._get_mag_phase(
                waveform,
                chunk_wave=self.chunking_enabled,
                chunk_buffer=self.chunking_params["chunk_buffer"],
                scale=self.scale,
            )
            y, padding_length = self._get_mag_phase(
                waveform_org,
                chunk_wave=self.chunking_enabled,
                chunk_buffer=self.chunking_params["chunk_buffer"],
                scale=self.scale,
            )
        else:
            if self.chunking_enabled:
                # Return the chunked waveform
                # Shape: (num_chunks, channel (mono: 1), chunk_size + 2 * chunk_buffer)
                x = preprocessing.cut2chunks(
                    waveform=waveform,
                    chunk_size=self.chunking_params["chunk_size"],
                    overlap=self.chunking_params["overlap"],
                    chunk_buffer=self.chunking_params["chunk_buffer"],
                )
                y = preprocessing.cut2chunks(
                    waveform=waveform_org,
                    chunk_size=self.chunking_params["chunk_size"],
                    overlap=self.chunking_params["overlap"],
                    chunk_buffer=self.chunking_params["chunk_buffer"],
                )
                # print(x.shape, y.shape)
            else:
                # Return the waveform
                # Shape: (1, self.length)
                x = waveform
                y = waveform_org

        return x, y, padding_length, hf

    def _get_mag_phase(
        self,
        waveform: torch.Tensor,
        chunk_wave: bool = True,
        chunk_buffer: int = 0,
        scale: str = "log",
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
                chunk_buffer=chunk_buffer,
                return_padding_length=True,
            )
            # Compute STFT for each segment and convert to magnitude and phase
            mag = []
            phase = []
            for chunk_y in chunks:
                mag_chunk, phase_chunk = preprocessing.get_mag_phase(
                    chunk_y,
                    chunk_wave=True,
                    chunk_buffer=chunk_buffer,
                    scale=scale,
                    stft_params=self.stft_params,
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
            # Compute STFT for the whole waveform and convert to magnitude and phase
            mag, phase = preprocessing.get_mag_phase(
                waveform,
                chunk_wave=False,
                scale=scale,
                stft_params=self.stft_params,
            )
            # Make mag and phase to a pair, shape (2, num_frequencies, num_frames)
            mag_phase_pair = torch.stack((mag, phase))

        # Return the magnitude and phase in tensors
        return mag_phase_pair, padding_length

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
        x, y, padding_length, hf = self.get_io_pairs(waveform, sr)

        return x, y, padding_length, hf

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

    with open("./config/config.json") as f:
        config = json.load(f)

    # Ensure the output directory exists
    output_dir = "./output/dev/data_loader"
    ensure_dir(output_dir)

    print("Loading the VCTK_092 dataset...")

    # Set up the data loader
    data_loader = VCTKDataLoader(
        data_dir=config["data_loader"]["args"]["data_dir"],
        quantity=0.001,
        batch_size=4,
        num_workers=0,
        validation_split=0.1,
        chunking_params={"chunk_size": 10160, "overlap": 0, "chunk_buffer": 240},
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
