# This is a script to check the distribution of lengths of files in the dataset.
import os
import sys
import glob
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Used for debugging data_loader
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == "__main__":
    # Set the source and out path
    source_path = "data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav"
    output_path = "debug/dev/data_preprocessing"
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    # Get the list of files
    files = glob.glob(os.path.join(source_path, "**/*.wav"), recursive=True)
    # Create a DataFrame to store the timestamps
    timestamps = pd.DataFrame(columns=["file", "length"])
    # Iterate over the files
    for idx, file in enumerate(tqdm(files)):
        # Get the length of the file
        length = len(sf.read(file)[0]) / 48000
        # Append the length to the list
        timestamps = pd.concat(
            [timestamps, pd.DataFrame({"file": [file], "length": [length]})],
            ignore_index=True,
        )

    # Print the statistics
    print(timestamps["length"].describe())
    # Print the quantiles
    print(timestamps["length"].quantile([0.25, 0.5, 0.75]))

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
