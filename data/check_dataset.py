# This is a script to check the distribution of lengths of files in the dataset.
import matplotlib.pyplot as plt
import pandas as pd
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
    # Set the source and out path
    source_path = os.path.join(
        config["data_loader"]["args"]["data_dir"],
        "VCTK-Corpus-0.92/wav48_silence_trimmed_wav",
    )
    output_path = "output/dev/data_preprocessing"
    # Ensure the output directory exists
    ensure_dir(output_path)
    # Read the timestamps txt as pandas dataframe
    timestamps = pd.read_csv(config["flac2wav"]["timestamps"], sep=" ", header=None)
    # Set the column names
    timestamps.columns = ["filename", "start", "end"]
    # The start and end are in seconds, subtract the start from the end to get the length
    timestamps["length"] = timestamps["end"] - timestamps["start"]

    # Print the statistics
    print(timestamps["length"].describe())
    # Print the first 5 rows
    print(timestamps.head())

    # Plot the distribution of lengths
    # Save the plot
    timestamps["length"].plot(kind="hist", bins=100)
    # Set title
    plt.title("Length Distribution")
    # Set x label
    plt.xlabel("Length")
    # Set y label
    plt.ylabel("Count")
    # Plot the 4 quantiles
    plt.axvline(timestamps["length"].quantile(0.25), color="r", linestyle="--")
    plt.axvline(timestamps["length"].quantile(0.5), color="r", linestyle="--")
    plt.axvline(timestamps["length"].quantile(0.75), color="r", linestyle="--")
    # Save the plot
    plt.savefig(os.path.join(output_path, "audio_length_distribution.png"))
