from tqdm import tqdm
from glob import glob
import pandas as pd
import torchaudio
import os
import sys

# Used for debugging data_loader
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import utils
from utils import ensure_dir

# Load the `./config.json` as config
import json

with open("./config/config.json") as f:
    config = json.load(f)


if __name__ == "__main__":
    # Download VCTK_092 dataset, default mic is "mic1"
    dataset = torchaudio.datasets.VCTK_092(
        root=config["data_loader"]["args"]["data_dir"], mic_id="mic1", download=True
    )
    # Check the total number of samples, and print the first sample
    print(f"Total number of samples: {len(dataset)}")
    print(dataset[0])

    # Set the destination path
    trimmed_folder = os.path.join(
        config["data_loader"]["args"]["data_dir"],
        "VCTK-Corpus-0.92/wav48_silence_trimmed_wav",
    )
    # Make sure the output directory exists, if not, create it
    ensure_dir(trimmed_folder)

    # Read the timestamps txt as pandas dataframe
    timestamps = pd.read_csv(config["flac2wav"]["timestamps"], sep=" ", header=None)
    # Set the column names
    timestamps.columns = ["filename", "start", "end"]
    # Convert seconds to samples with the sample rate
    timestamps["start"] = timestamps["start"] * config["flac2wav"]["source_sr"]
    timestamps["end"] = timestamps["end"] * config["flac2wav"]["source_sr"]
    # Change the type to int
    timestamps["start"] = timestamps["start"].astype(int)
    timestamps["end"] = timestamps["end"].astype(int)
    # Print the timestamps
    print(timestamps.head())

    # Iterate over the dataset
    for waveform, sample_rate, transcript, speaker_id, utterance_id in tqdm(dataset):
        # Combine speaker_id and utterance_id, use it to match the timestamps
        flac_name = f"{speaker_id}_{utterance_id}"
        # Get the if from the timestamps dataframe
        flac_timestamps = timestamps[timestamps["filename"] == flac_name]
        # Set the destination file folder
        dest_folder = os.path.join(trimmed_folder, f"{speaker_id}")
        # Set the destination file path
        dest_path = os.path.join(dest_folder, f"{flac_name}.wav")
        # Make sure the output directory exists, if not, create it
        ensure_dir(dest_folder)
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
