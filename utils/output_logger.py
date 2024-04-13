import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from data_loader import preprocessing


def fig2np(fig):
    """
    Convert a Matplotlib figure to a numpy array with RGBA channels
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


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


def log_spectrogram(
    writer, name_list, specs, stft=False, chunks=False, sample_rate=48000
):
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
            names=name_list, waveforms=specs, stft=stft, sample_rate=sample_rate
        )
        filename = "Spectrogram (STFT)" if stft else "Spectrogram"
        # Log the spectrogram plot
        writer.add_image(filename, spec_plot, dataformats="HWC")
    if chunks:
        # chunks has magnitude and phase [[tensor_1, tensor_2], [tensor_1, tensor_2], [tensor_1, tensor_2]]
        # We want to get [[tensor_1, tensor_1, tensor_1], [tensor_2, tensor_2, tensor_2]]
        for i in range(2):
            chunk_list = [chunk[i] for chunk in specs]
            filename = "Chunks (Magnitude)" if i == 0 else "Chunks (Phase)"
            chunk_plot = plot_spectrogram_from_chunks(
                names=name_list, chunk_list=chunk_list, title=filename
            )
            writer.add_image(filename, chunk_plot, dataformats="HWC")


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


def plot_spectrogram_from_wave(
    names, waveforms, title="Spectrogram", stft=False, sample_rate=48000
):
    """
    Plots the spectrogram using matplotlib

    Args:
        names (list): List of names for the waveforms
        waveforms (Tensor): Waveforms to plot
        title (str): Title of the plot
        stft (bool): Whether to use STFT or Spectrogram
        sample_rate (int): Sample rate of audio signal

    Returns:
        np.ndarray: Spectrogram plot
    """
    # Set the number of subplots
    n_plots = len(waveforms)
    # Create a subplots figure with n_plots rows
    if stft:
        fig, axs = plt.subplots(n_plots, 3, figsize=(16, 12))
    else:
        fig, axs = plt.subplots(n_plots, 2, figsize=(10, 12))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(names, waveforms)):
        if stft:
            # Get mag, mag_dB, phase
            mag, phase = preprocessing.get_mag_phase(
                waveform=waveform, chunk_wave=False, scale="log"
            )
            mag_dB, _ = preprocessing.get_mag_phase(
                waveform=waveform, chunk_wave=False, scale="dB"
            )
            # Plot the magnitude of the STFT
            img = axs[i, 0].imshow(
                mag.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=-15,
            )
            plt.colorbar(img, ax=axs[i, 0])
            # Plot the magnitude of the STFT
            img = axs[i, 1].imshow(
                mag_dB.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=-40,
            )
            plt.colorbar(img, ax=axs[i, 1])
            # Plot the phase of the STFT
            img = axs[i, 2].imshow(
                phase.squeeze(0).detach().cpu().numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            plt.colorbar(img, ax=axs[i, 2])
            # Set the title
            axs[i, 0].set_title(f"{name} (log)")
            axs[i, 1].set_title(f"{name} (dB)")
            axs[i, 2].set_title(name)
            # Set the x-axis label
            axs[i, 0].set_xlabel("Time")
            axs[i, 1].set_xlabel("Time")
            axs[i, 2].set_xlabel("Time")
            # Set the y-axis label
            axs[i, 0].set_ylabel("Frequency")
            axs[i, 1].set_ylabel("Frequency")
            axs[i, 2].set_ylabel("Frequency")
        else:
            # Plot the waveform (Spectrogram)
            frequencies, times, spectrogram = signal.spectrogram(
                waveform.squeeze().detach().cpu().numpy(),
                fs=sample_rate,
                scaling="spectrum",
            )
            img = axs[i, 0].pcolormesh(
                times,
                frequencies,
                10 * np.log10(spectrogram + 1e-18),
                cmap="viridis",
                shading="auto",
                vmin=-150,
            )
            plt.colorbar(img, ax=axs[i, 0])
            img = axs[i, 1].pcolormesh(
                times,
                frequencies,
                10 * np.log10(np.abs(spectrogram) ** 2 + 1e-18),
                cmap="viridis",
                shading="auto",
            )
            plt.colorbar(img, ax=axs[i, 1])
            # Set the title
            axs[i, 0].set_title(f"{name} (log)")
            axs[i, 1].set_title(f"{name} (dB)")
            # Set the x-axis label
            axs[i, 0].set_xlabel("Time")
            axs[i, 1].set_xlabel("Time")
            # Set the y-axis label
            axs[i, 0].set_ylabel("Frequency (kHz)")
            axs[i, 1].set_ylabel("Frequency (kHz)")
            # Set y-axis from Hz to kHz
            axs[i, 0].yaxis.set_major_locator(
                plt.FixedLocator([0, 5000, 10000, 15000, 20000])
            )
            axs[i, 1].yaxis.set_major_locator(
                plt.FixedLocator([0, 5000, 10000, 15000, 20000])
            )
            axs[i, 0].set_yticklabels(
                [f"{int(f/1000)}" for f in axs[i, 0].get_yticks()]
            )
            axs[i, 1].set_yticklabels(
                [f"{int(f/1000)}" for f in axs[i, 1].get_yticks()]
            )

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


def plot_spectrogram_from_chunks(names, chunk_list, title="Chunks (Magnitude)"):
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
    # Set the titles for each row at the left side of the figure
    axs[0, 0].set_ylabel("Input")
    axs[1, 0].set_ylabel("Output")
    axs[2, 0].set_ylabel("Target")
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
            axs[0, j].set_title(f"{j+1}")
            # Plot the chunk
            img = axs[i, j].pcolormesh(
                # Clip the values
                chunk.squeeze().detach().cpu().numpy(),
                # vmax=7 if title == "Chunks (Magnitude)" else None,
                # vmin=-15 if title == "Chunks (Magnitude)" else None,
                cmap="viridis",
                shading="auto",
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
