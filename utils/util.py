import json
import torch
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from itertools import repeat
import matplotlib.pyplot as plt
from collections import OrderedDict

try:
    with open("./config.json") as f:
        config = json.load(f)
except:
    import os
    import sys

    # Used for debugging data_loader
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    with open("./config.json") as f:
        config = json.load(f)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        # Set columns to float
        self._data = self._data.astype("float")
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def fig2np(fig):
    """
    Convert a Matplotlib figure to a numpy array with RGBA channels
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_waveform(names, waveforms, title="Waveform", xlim=None, ylim=None):
    """
    Plots the waveform using matplotlib

    Args:
        names (list): List of names for the waveforms
        waveforms (Tensor): Waveforms to plot
        sample_rate (int): Sample rate of audio signal
        title (str): Title of the plot
        xlim (list): Limits for the x-axis
        ylim (list): Limits for the y-axis

    Returns:
        np.ndarray: Waveform plot
    """
    # Set the number of subplots
    n_plots = len(waveforms)
    # Create a subplots figure with n_plots rows
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 14))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(names, waveforms)):
        waveform = waveform.t().detach().cpu().numpy()
        # Set the title of the plot
        axs[i].set_title(name)
        # Plot the waveform
        axs[i].plot(waveform)
        # Set the x-axis label
        axs[i].set_xlabel("Samples")
        # Set the y-axis label
        axs[i].set_ylabel("Amplitude")
        # Set the x-axis limits
        if xlim:
            axs[i].set_xlim(xlim)
        # Set the y-axis limits
        if ylim:
            axs[i].set_ylim(ylim)
    # Set the title of the plot
    plt.suptitle(title)
    # Set layout to tight
    plt.tight_layout()
    # Convert fig to numpy array
    fig.canvas.draw()
    plot = fig2np(fig)
    # Close the figure to free memory
    plt.close()
    # Return the plot
    return plot


def plot_spectrogram_from_wave(names, waveforms, title="Spectrogram", stft=False):
    """
    Plots the spectrogram using matplotlib

    Args:
        names (list): List of names for the waveforms
        waveforms (Tensor): Waveforms to plot
        sample_rate (int): Sample rate of audio signal
        title (str): Title of the plot

    Returns:
        np.ndarray: Spectrogram plot
    """
    # Set STFT parameters
    n_fft = 1022
    hop_length = 478
    win_length = 956
    window = torch.hann_window(win_length).to(waveforms[0].device)
    # Set the number of subplots
    n_plots = len(waveforms)
    # Create a subplots figure with n_plots rows
    if stft:
        fig, axs = plt.subplots(n_plots, 2, figsize=(16, 12))
    else:
        fig, axs = plt.subplots(n_plots, 1, figsize=(16, 12))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(names, waveforms)):
        if stft:
            # Compute the STFT
            spec = torch.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            # Plot the magnitude of the STFT
            img = axs[i, 0].pcolormesh(
                torch.log2(torch.abs(spec) + 1e-8).squeeze(0).detach().cpu().numpy(),
                vmin=-15,
                cmap="viridis",
                shading="auto",
            )
            # Add colorbar
            plt.colorbar(img, ax=axs[i, 0])
            # Plot the phase of the STFT
            img = axs[i, 1].pcolormesh(
                torch.angle(spec).squeeze(0).detach().cpu().numpy(),
                cmap="viridis",
                shading="auto",
            )
            # Add colorbar
            plt.colorbar(img, ax=axs[i, 1])
            # Set the title
            axs[i, 0].set_title(name)
            axs[i, 1].set_title(name)
            # Set the x-axis label
            axs[i, 0].set_xlabel("Time")
            axs[i, 1].set_xlabel("Time")
            # Set the y-axis label
            axs[i, 0].set_ylabel("Frequency")
            axs[i, 1].set_ylabel("Frequency")
        else:
            # Set the title
            # Plot the waveform (Spectrogram)
            frequencies, times, spectrogram = signal.spectrogram(
                waveform.squeeze().detach().cpu().numpy(), fs=config["source_sr"]
            )
            img = axs[i].pcolormesh(
                times,
                frequencies,
                10 * np.log10(spectrogram + 1e-8),
                cmap="viridis",
                shading="auto",
            )
            # Add colorbar
            plt.colorbar(img, ax=axs[i])
            axs[i].set_title(name)
            # Set the x-axis label
            axs[i].set_xlabel("Time")
            # Set the y-axis label
            axs[i].set_ylabel("Frequency")
    # Set the title of the plot
    plt.suptitle(title)
    # Set layout to tight
    plt.tight_layout()
    # Convert fig to numpy array
    fig.canvas.draw()
    plot = fig2np(fig)
    # Close the figure to free memory
    plt.close()
    # Return the plot
    return plot


def plot_spectrogram_from_chunks(names, chunk_list, title="Spectrogram (Chunks)"):
    """
    Plots the spectrogram of chunks using matplotlib.

    The plot has three rows and the number of columns is equal to the number of chunks.

    Args:
        names (list): List of names for the waveforms
        chunk_list (Tensor): List of chunks to plot
        sample_rate (int): Sample rate of audio signal
        title (str): Title of the plot

    Returns:
        np.ndarray: Spectrogram plot
    """
    # Initialize the figure (3 rows, number of chunks columns)
    fig, axs = plt.subplots(
        3, chunk_list[0].size(1), figsize=(21, 7), sharex=True, sharey=True
    )
    # Set the title of the plot
    plt.suptitle(title)
    # Remove axis labels
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    # Add axis labels
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # Iterate over each names (row)
    for i, name in enumerate(names):
        # Set the number of chunks to be the first dimension
        chunks = chunk_list[i].permute(1, 0, 2, 3)
        # Iterate over each chunk (column)
        for j, chunk in enumerate(chunks):
            # Set the title of the plot
            axs[i, j].set_title(f"{j+1}")
            # Plot the chunk
            axs[i, j].imshow(
                chunk.squeeze().detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
            )
    # Set layout to tight
    plt.tight_layout()
    # Convert fig to numpy array
    fig.canvas.draw()
    plot = fig2np(fig)
    # Close the figure to free memory
    plt.close()
    # Return the plot
    return plot


def log_audio(writer, name_list, waveforms, sample_rate=config["source_sr"]):
    for name, waveform in zip(name_list, waveforms):
        writer.add_audio(name, waveform, sample_rate=sample_rate)


def log_waveform(writer, name_list, waveforms):
    """
    Plot and log waveforms to tensorboard in a figure

    Args:
        writer (TensorboardWriter): Tensorboard writer object
        name_list (list): List of names for the waveforms
        waveforms (list): List of waveforms to plot
        sample_rate (int): Sample rate of audio signal

    Returns:
        None
    """
    # Call plot_waveform for each waveform
    wave_plot = plot_waveform(names=name_list, waveforms=waveforms)
    # Log the waveform plot
    writer.add_image("Waveform", wave_plot, dataformats="HWC")


def log_spectrogram(writer, name_list, specs, stft=False, chunks=False):
    """
    Plot and log spectrograms to tensorboard in a figure

    Args:
        writer (TensorboardWriter): Tensorboard writer object
        name_list (list): List of names for the waveforms
        waveforms (list): List of waveforms to plot
        sample_rate (int): Sample rate of audio signal

    Returns:
        None
    """
    if not chunks:
        # Call plot_spectrogram_from_wave for each waveform
        spec_plot = plot_spectrogram_from_wave(
            names=name_list, waveforms=specs, stft=stft
        )
        filename = "Spectrogram (STFT)" if stft else "Spectrogram"
        # Log the spectrogram plot
        writer.add_image(filename, spec_plot, dataformats="HWC")
    if chunks:
        # chunks has magnitude and phase [[tensor_1, tensor_2], [tensor_1, tensor_2], [tensor_1, tensor_2]]
        # We want to get [[tensor_1, tensor_1, tensor_1], [tensor_2, tensor_2, tensor_2]]
        for i in range(2):
            chunk_list = [chunk[i] for chunk in specs]
            chunk_plot = plot_spectrogram_from_chunks(
                names=name_list, chunk_list=chunk_list
            )
            filename = "Chunks (Magnitude)" if i == 0 else "Chunks (Phase)"
            writer.add_image(filename, chunk_plot, dataformats="HWC")
