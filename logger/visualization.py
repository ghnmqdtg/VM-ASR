import importlib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from utils.stft import wav2spectro


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, but currently not installed on "
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to "
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                )
                logger.warning(message)

        self.step = 0
        self.mode = ""

        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name
                    )
                )
            return attr


def log_audio(writer, wave_in, wave_out, wave_target, config):
    name_list = ["Input", "Output", "Target"]
    wave_list = [wave_in, wave_out, wave_target]
    for name, wave in zip(name_list, wave_list):
        writer.add_audio(name, wave, sample_rate=config.DATA.TARGET_SR)


def log_waveform(writer, wave_in, wave_out, wave_target, config):
    name_list = ["Input", "Output", "Target"]
    wave_list = [wave_in, wave_out, wave_target]
    # Call plot_waveform for each waveform
    wave_plot = plot_waveform(
        name_list=name_list,
        wave_list=wave_list,
        config=config,
    )
    # Log the waveform plot
    writer.add_image("Waveform", wave_plot, dataformats="HWC")


def log_spectrogram(writer, wave_in, wave_out, wave_target, config):
    name_list = ["Input", "Output", "Target"]
    wave_list = [wave_in, wave_out, wave_target]
    # Call plot_spectrogram for each waveform
    spectro_plot = plot_spectrogram(
        name_list=name_list,
        wave_list=wave_list,
        config=config,
    )
    # Log the spectrogram plot
    writer.add_image("Spectrogram", spectro_plot, dataformats="HWC")


def fig2np(fig):
    """
    Convert a Matplotlib figure to a numpy array with RGBA channels
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_waveform(name_list, wave_list, config, title="Waveform", xlim=None, ylim=None):
    """
    Plots the waveform using matplotlib

    Args:
        name_list (list): List of names for the waveforms
        wave_list (Tensor): Waveforms to plot
        sample_rate (int): Sample rate of audio signal
        title (str): Title of the plot
        xlim (list): Limits for the x-axis
        ylim (list): Limits for the y-axis

    Returns:
        np.ndarray: Waveform plot
    """
    # Set the number of subplots
    n_plots = len(wave_list)
    # Create a subplots figure with n_plots rows
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 14))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(name_list, wave_list)):
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


def plot_spectrogram(name_list, wave_list, config, title="Spectrogram"):
    # Set the number of subplots
    n_plots = len(wave_list)
    # Create a subplots figure with n_plots rows
    fig, axs = plt.subplots(n_plots, 3, figsize=(16, 12))
    # Iterate over each waveform
    for i, (name, waveform) in enumerate(zip(name_list, wave_list)):
        mag, phase = wav2spectro(
            waveform,
            n_fft=config.DATA.STFT.N_FFT,
            hop_length=config.DATA.STFT.HOP_LENGTH,
            win_length=config.DATA.STFT.WIN_LENGTH,
            spectro_scale="log2",
        )
        mag_dB, _ = wav2spectro(
            waveform,
            n_fft=config.DATA.STFT.N_FFT,
            hop_length=config.DATA.STFT.HOP_LENGTH,
            win_length=config.DATA.STFT.WIN_LENGTH,
            spectro_scale="dB",
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
        axs[i, 0].set_title(f"{name} (log2)")
        axs[i, 1].set_title(f"{name} (dB)")
        axs[i, 2].set_title(f"{name} (phase)")
        # Set the x-axis label
        axs[i, 0].set_xlabel("Time")
        axs[i, 1].set_xlabel("Time")
        axs[i, 2].set_xlabel("Time")
        # Set the y-axis label
        axs[i, 0].set_ylabel("Frequency")
        axs[i, 1].set_ylabel("Frequency")
        axs[i, 2].set_ylabel("Frequency")

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
