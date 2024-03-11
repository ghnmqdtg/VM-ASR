import json
import torch
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from itertools import repeat
import matplotlib.pyplot as plt
from collections import OrderedDict


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


def plot_spectrogram(names, waveforms, title="Spectrogram", stft=False):
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
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 14))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(names, waveforms)):
        # Set the title of the plot
        axs[i].set_title(name)
        if stft:
            # Plot the waveform (STFT)
            axs[i].imshow(
                torch.log2(
                    torch.abs(
                        torch.stft(
                            waveform,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length,
                            window=window,
                            return_complex=True,
                        )
                    )
                    + 1e-8
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy(),
                aspect="auto",
                origin="lower",
            )
        else:
            # Plot the waveform (Spectrogram)
            # TODO: Set sample rate in config
            frequencies, times, spectrogram = signal.spectrogram(
                waveform.squeeze().detach().cpu().numpy(), fs=48000
            )
            axs[i].pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-8))
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


# TODO: Set sample rate in config
def log_audio(writer, name_list, waveforms, sample_rate=48000):
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


def log_spectrogram(writer, name_list, specs, stft=False):
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
    # Call plot_spectrogram for each waveform
    spec_plot = plot_spectrogram(names=name_list, waveforms=specs, stft=stft)
    filename = "Spectrogram (STFT)" if stft else "Spectrogram"
    # Log the spectrogram plot
    writer.add_image(filename, spec_plot, dataformats="HWC")
