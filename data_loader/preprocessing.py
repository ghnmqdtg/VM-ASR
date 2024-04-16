import json
import math
import time
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import random
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.signal import cheby1, butter
from scipy.signal import sosfiltfilt
from scipy.signal import resample_poly

try:
    from utils import ensure_dir
    from data_loader.postprocessing import (
        concatenate_wave_chunks,
        reconstruct_from_stft,
        reconstruct_from_stft_chunks,
    )
except:
    import os
    import sys

    # Used for debugging data_loader
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from utils import ensure_dir
    from data_loader.postprocessing import (
        concatenate_wave_chunks,
        reconstruct_from_stft,
        reconstruct_from_stft_chunks,
    )


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    config_dataloader: dict = {"length": 121890, "white_noise": 1e-06},
) -> torch.Tensor:
    device = waveform.device
    length = config_dataloader["length"]
    white_noise = config_dataloader["white_noise"]
    # If the waveform is shorter than the required length, pad it
    if waveform.shape[1] < length:
        pad_length = length - waveform.shape[1]
        # Randomly select the length to pad
        pad_begin = random.randint(0, pad_length)
        pad_end = pad_length - pad_begin
        if not white_noise:
            # Pad the waveform with zeros to the left and right
            # Left: random length between 0 and pad_length, Right: pad_length - r
            waveform = F.pad(waveform, (pad_begin, pad_end), mode="constant", value=0)
        else:
            # Pad white noise to the waveform
            noise_front = (torch.randn(pad_begin).to(device) * white_noise).unsqueeze(0)
            noise_back = (torch.randn(pad_end).to(device) * white_noise).unsqueeze(0)
            waveform = torch.cat((noise_front, waveform, noise_back), dim=-1)
            # Apply fade in and fade out to the padded waveform
            waveform = T.Fade(
                fade_in_len=pad_begin, fade_out_len=pad_end, fade_shape="exponential"
            )(waveform)
        # print(f"Pad length: {pad_length}, Random length: {r}")
    else:
        # If the waveform is longer than the required length, crop it randomly from the beginning
        start = random.randint(0, waveform.shape[1] - length)
        # Crop the waveform from start to start + length (fixed length)
        waveform = waveform[:, start : start + length]
        # print(f"Crop to length: {length}, Start: {start}")

    # print(f'New shape of padded or cropped waveform: {waveform.shape}')
    return waveform


def butter_lowpass_filter(
    normalized_cutoff: float, waveform: torch.Tensor
) -> torch.Tensor:
    """
    Apply low pass filter to the waveform to remove the high frequency components.
    """
    order = random.randint(1, 11)
    # Create the filter coefficients
    sos = butter(order, normalized_cutoff, btype="low", output="sos")
    # Apply the filter
    waveform = sosfiltfilt(sos, waveform)
    # Convert the waveform to tensor
    waveform_tensor = torch.tensor(waveform.copy(), dtype=torch.float32)
    # Return the waveform
    return waveform_tensor


def cheby1_lowpass_filter(
    normalized_cutoff: float, waveform: torch.Tensor, order: int, ripple: float
) -> torch.Tensor:
    """
    Apply low pass filter to the waveform to remove the high frequency components.
    """
    # Create the filter coefficients
    sos = cheby1(order, ripple, normalized_cutoff, btype="lowpass", output="sos")
    # Apply the filter
    waveform = sosfiltfilt(sos, waveform)
    # Convert the waveform to tensor
    waveform_tensor = torch.tensor(waveform.copy(), dtype=torch.float32)
    # Return the waveform
    return waveform_tensor


def low_pass_filter(
    waveform: torch.Tensor, sr_org: int, sr_new: int, **kwargs
) -> torch.Tensor:
    """
    Apply low pass filter to the waveform to remove the high frequency components.
    This can avoid aliasing when downsampling the waveform.

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int): The new sample rate
        randomize (bool): Whether to randomize the filter parameters

    Returns:
        torch.Tensor: The filtered waveform
    """
    # Define the cutoff frequency and the ratio to the Nyquist frequency
    nyquist = sr_org / 2
    highcut = sr_new // 2
    normalized_cutoff = highcut / nyquist
    order = kwargs.get("order", 6)
    ripple = kwargs.get("ripple", 1e-3)
    # Apply the low pass filter
    waveform = cheby1_lowpass_filter(normalized_cutoff, waveform, order=order, ripple=ripple)

    # Return the waveform
    return waveform


def resample_audio(waveform: torch.Tensor, sr_org: int, sr_new: int) -> torch.Tensor:
    """
    Resample the waveform to the new sample rate

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int): The new sample rate

    Returns:
        torch.Tensor: The resampled waveform
    """
    # waveform_resampled = T.Resample(sr_org, sr_new)(waveform)
    waveform_resampled = resample_poly(waveform, sr_new, sr_org, axis=-1)
    return torch.tensor(waveform_resampled, dtype=torch.float32)


def align_waveform(
    waveform_resampled: torch.Tensor, waveform: torch.Tensor
) -> torch.Tensor:
    # Make sure the waveform has the same length
    if waveform_resampled.shape[1] < waveform.shape[1]:
        waveform_resampled = F.pad(
            waveform_resampled,
            (0, waveform.shape[1] - waveform_resampled.shape[1]),
            mode="constant",
            value=0,
        )
    elif waveform_resampled.shape[1] > waveform.shape[1]:
        waveform_resampled = waveform_resampled[: waveform.shape[1]]

    return waveform_resampled


def low_sr_simulation(waveform: torch.Tensor, sr_org: int, sr_new: int) -> torch.Tensor:
    """
    Simulate the low sample rate

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int): The new sample rate

    Returns:
        torch.Tensor: The low sample rate waveform
    """
    # Apply low pass filter to the waveform
    waveform_lr = low_pass_filter(waveform, sr_org, sr_new)
    # Downsample the waveform
    waveform_lr = resample_audio(waveform_lr, sr_org, sr_new)
    # Return the low sample rate waveform
    return waveform_lr


def cut2chunks(
    waveform: torch.Tensor,
    chunk_size: int,
    overlap: int,
    chunk_buffer: int = 0,
    return_padding_length: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """
    Cut the waveform into chunks with the specified size and overlap.

    Args:
        waveform (torch.Tensor): The input waveform
        chunk_size (int): The size of each chunk
        overlap (int): The overlap between chunks
        return_padding_length (bool, optional): Whether to return the padding length. Defaults to False.

    Returns:
        torch.Tensor: A stack of chunks
        padding_length (int, optional): The length of the padding. Defaults to None.
    """
    # Calculate total chunk size including buffer
    total_chunk_size = chunk_size + 2 * chunk_buffer

    # Get the length of the waveform
    length = waveform.shape[-1]

    # Calculate the number of chunks by dividing the length by the step size
    step_size = chunk_size - overlap
    num_chunks = max(1, math.ceil((length - chunk_buffer) / step_size))

    # Prepare padding for the last chunk
    padding_needed = total_chunk_size * num_chunks - length
    padding_length = padding_needed + chunk_buffer

    # Pad waveform at the end to accommodate even division into chunks
    waveform = F.pad(waveform, (chunk_buffer, padding_length), "constant", 0)

    # Create chunks with overlap and buffer
    chunks = []
    for i in range(num_chunks):
        start = i * step_size
        chunks.append(waveform[..., start : start + total_chunk_size])

    # Stack all chunks along a new dimension
    stacked_chunks = torch.stack(chunks)

    if return_padding_length:
        return stacked_chunks, padding_length
    else:
        return stacked_chunks, 0


def get_mag_phase(
    waveform: torch.Tensor,
    chunk_wave: bool = True,
    chunk_buffer: int = 0,
    batch_input: bool = False,
    stft_params: dict = {
        "chunks": {"n_fft": 1022, "hop_length": 80, "win_length": 320},
        "full": {"n_fft": 1022, "hop_length": 478, "win_length": 956},
    },
    scale: str = "log",  # "log" or "dB"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply short time Fourier transform to the waveform and return the magnitude and phase

    Args:
        waveform (torch.Tensor): The waveform
        chunk_wave (bool, optional): Whether to chunk the waveform. Defaults to True.
        batch_input (bool, optional): Whether the input is a batch. Defaults to False.

    Returns:
        torch.Tensor: The magnitude
        torch.Tensor: The phase
    """
    if chunk_wave:
        n_fft = stft_params["chunks"]["n_fft"]
        hop_length = stft_params["chunks"]["hop_length"]
        win_length = stft_params["chunks"]["win_length"]
        window = torch.hann_window(win_length).to(waveform.device)
        if chunk_buffer:
            # Calculate how much to crop on each side of the spectrogram
            crop_side = chunk_buffer // hop_length
    else:
        n_fft = stft_params["full"]["n_fft"]
        hop_length = stft_params["full"]["hop_length"]
        win_length = stft_params["full"]["win_length"]
        window = torch.hann_window(win_length).to(waveform.device)

    if not batch_input:
        # Apply short time Fourier transform to the waveform
        spec = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        if chunk_buffer:
            # Crop the spectrogram
            spec = spec[..., crop_side:-crop_side]

        if scale == "dB":
            # The AmplitudeToDB default input is in power |x|^2
            mag = T.AmplitudeToDB(stype="power", top_db=80)(torch.abs(spec).pow(2))
            phase = torch.angle(spec)
        else:
            # Magnitude is calculated as the absolute value, and log2 is applied to compress the dynamic range
            mag = torch.log2(torch.abs(spec) + 1e-8)
            phase = torch.angle(spec)

        phase = torch.angle(spec)

        # Return the magnitude and phase
        return mag, phase
    else:
        # The input would be of shape (batch_size, 1 (mono), waveform_length)
        # Loop over the batch and apply STFT to each waveform
        mags = []
        phases = []
        for i in range(waveform.size(0)):
            # Apply short time Fourier transform to the waveform
            spec = torch.stft(
                waveform[i],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            if chunk_buffer:
                # Crop the spectrogram
                spec = spec[..., crop_side:-crop_side]

            if scale == "dB":
                # The AmplitudeToDB default input is in power |x|^2
                mag = T.AmplitudeToDB(stype="power", top_db=80)(torch.abs(spec).pow(2))
                phase = torch.angle(spec)
            else:
                # Magnitude is calculated as the absolute value, and log2 is applied to compress the dynamic range
                mag = torch.log2(torch.abs(spec) + 1e-8)
                phase = torch.angle(spec)

            mags.append(mag)
            phases.append(phase)
        # Stack the magnitude and phase
        mags = torch.stack(mags, dim=0)
        phases = torch.stack(phases, dim=0)
        # Print the shapes of the magnitude and phase
        # print(f"Shape of mag: {mags.shape}, Shape of phase: {phases.shape}")
        # Return the magnitude and phase
        return mags, phases


def plot_all(waveform: torch.Tensor, sample_rate: int, filename: str) -> None:
    """
    Plot the waveform, magnitude and phase in a row at the same figure and save it to the output folder

    Args:
        waveform (torch.Tensor): The waveform
        sample_rate (int): The sample rate
        filename (str): The filename of the output figure

    Returns:
        None
    """
    # Get the magnitude and phase
    mag_log, phase = get_mag_phase(waveform, chunk_wave=False, scale="log")
    mag_dB, _ = get_mag_phase(waveform, chunk_wave=False, scale="dB")
    # Plot the waveform, magnitude and phase
    plt.figure(figsize=(16, 4))
    # Add title to the figure
    plt.suptitle(f"Waveform, Magnitude and Phase at {sample_rate} Hz")
    # Waveform
    plt.subplot(1, 4, 1)
    plt.plot(waveform.t().numpy())
    plt.title("Waveform")
    # Magnitude (log)
    plt.subplot(1, 4, 2)
    plt.pcolormesh(mag_log.numpy().squeeze(0), vmin=-15, cmap="viridis", shading="auto")
    plt.title("Magnitude (log)")
    plt.colorbar()
    # Magnitude (dB)
    plt.subplot(1, 4, 3)
    plt.imshow(
        mag_dB.numpy().squeeze(0),
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="viridis",
    )
    plt.title("Magnitude (dB)")
    plt.colorbar()
    # Phase
    plt.subplot(1, 4, 4)
    plt.imshow(phase.numpy().squeeze(0), aspect="auto", origin="lower")
    plt.colorbar()
    # Add space between subplots
    plt.tight_layout()
    plt.title("Phase")
    plt.savefig(filename)
    plt.close()


def plot_chunks(sr_new, chunks, mag_chunks, mag_chunks_dB, phase_chunks):
    fig, axs = plt.subplots(4, mag_chunks.size(0), figsize=(25, 7))
    # Set the title of the plot
    plt.suptitle(f"Chunks: Magnitude and Phase at {sr_new} Hz")
    # Set the titles for each row at the left side of the figure
    axs[0, 0].set_ylabel("Magnitude (log)")
    axs[1, 0].set_ylabel("Magnitude (dB)")
    axs[2, 0].set_ylabel("Phase")
    axs[3, 0].set_ylabel("Waveform")
    # Set y limit for the waveform plot
    axs[3, 0].set_ylim(-1, 1)
    for i in range(mag_chunks.size(0)):
        axs[0, i].imshow(mag_chunks[i].numpy(), aspect="auto", origin="lower")
        axs[1, i].imshow(mag_chunks_dB[i].numpy(), aspect="auto", origin="lower")
        axs[2, i].imshow(phase_chunks[i].numpy(), aspect="auto", origin="lower")
        axs[3, i].plot(chunks[i].squeeze().numpy())
    plt.tight_layout()
    # plt.savefig(f"./output/dev/data_preprocessing/{timestr}_chunks_{sr_new}.png")
    plt.savefig(f"./output/dev/data_preprocessing/chunks_{sr_new}.png")


if __name__ == "__main__":
    # Choose an audio file to test
    filepath = "./data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p225/p225_001.wav"
    # Make sure the output folder exists
    ensure_dir("./output/dev/data_preprocessing")
    # Load the configuration file
    with open("./config/config.json") as f:
        config = json.load(f)
    config_dataloader = config["data_loader"]["args"]

    # Choose the test to run
    TEST_FILTER = True
    TEST_FULL_WAVEFORM = True
    TEST_CONCATENATE_CHUNKS = True
    TEST_RECONSTRUCT_FROM_CHUNKS = True

    # Set the parameters
    # List of target sample rates to choose from
    target_sample_rates = [16000]
    # Size of each audio chunk
    chunk_size = config_dataloader["chunking_params"]["chunk_size"]
    chunk_buffer = config_dataloader["chunking_params"]["chunk_buffer"]
    # Overlap size between chunks
    overlap = int(chunk_size * config_dataloader["chunking_params"]["overlap"])
    # Apply the audio preprocessing pipeline
    sr_new = random.choice(target_sample_rates)
    print(f"Randomly selected new sample rate: {sr_new} Hz")
    # Apply the audio preprocessing pipeline
    waveform, sr_org = torchaudio.load(filepath)
    # Crop or pad the waveform to the required length
    waveform = crop_or_pad_waveform(waveform, config_dataloader)
    waveform_filtered = low_pass_filter(waveform, sr_org, sr_new)
    waveform_downsampled = resample_audio(waveform_filtered, sr_org, sr_new)
    # Apply upsampling to get a unified sample rate as input
    waveform_upsampled = resample_audio(waveform_downsampled, sr_new, sr_org)
    # Align the waveform to the same length
    waveform_upsampled = align_waveform(waveform_upsampled, waveform)

    # Cut the waveform into chunks
    chunks, padding_length = cut2chunks(
        waveform_upsampled,
        chunk_size,
        overlap,
        chunk_buffer,
        return_padding_length=True,
    )
    print(f"chunk size: {chunk_size}, chunk_buffer: {chunk_buffer}, overlap: {overlap}")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Test the low sample rate simulation pipeline
    if TEST_FILTER:
        # Plot the waveform, magnitude and phase
        plot_all(
            waveform,
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_step_0_{sr_new}.png",
        )
        plot_all(
            waveform_filtered,
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_step_1_{sr_new}.png",
        )
        plot_all(
            waveform_downsampled,
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_step_2_{sr_new}.png",
        )
        plot_all(
            low_pass_filter(waveform_upsampled, sr_org, sr_new),
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_step_3_{sr_new}.png",
        )

    if TEST_FULL_WAVEFORM:
        # Get the magnitude and phase of the full waveform
        mag, phase = get_mag_phase(
            waveform,
            chunk_wave=False,
        )
        # Print the shapes of the magnitude and phase
        print(f"Shape of mag: {mag.shape}, Shape of phase: {phase.shape}")
        # Reconstruct the waveform from the magnitude and phase
        waveform_reconstructed_stft = reconstruct_from_stft(
            mag, phase, config_dataloader=config_dataloader
        )
        # Plot the waveform, magnitude and phase
        plot_all(
            waveform_reconstructed_stft,
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_full_{sr_new}.png",
        )
        # Save the reconstructed waveform
        torchaudio.save(
            f"./output/dev/data_preprocessing/{timestr}_full_{sr_new}.wav",
            waveform_reconstructed_stft,
            sr_org,
        )

    if TEST_CONCATENATE_CHUNKS:
        # Reconstruct the waveform from the chunks
        waveform_reconstructed = concatenate_wave_chunks(
            chunks, chunk_size, overlap, padding_length
        )

        # Save the reconstructed waveform
        torchaudio.save(
            f"./output/dev/data_preprocessing/wave_chunks_{sr_new}.wav",
            waveform_reconstructed,
            sr_org,
        )

    if TEST_RECONSTRUCT_FROM_CHUNKS:
        # Get the magnitude and phase of the chunks
        mag_chunks, phase_chunks = get_mag_phase(
            chunks,
            chunk_wave=True,
            chunk_buffer=chunk_buffer,
            batch_input=True,
            stft_params=config_dataloader["stft_params"],
        )
        mag_chunks_dB, _ = get_mag_phase(
            chunks,
            chunk_wave=True,
            chunk_buffer=chunk_buffer,
            batch_input=True,
            stft_params=config_dataloader["stft_params"],
            scale="dB",
        )
        print(
            f"Shape of mag_chunks: {mag_chunks.shape}, Shape of phase_chunks: {phase_chunks.shape}"
        )
        # Plot the magnitude, phase and wave of chunks
        # The figure is in the shape of (3, chunk_num)
        # Drop the channel dimension
        plot_chunks(
            sr_new,
            chunks,
            mag_chunks.squeeze(1),
            mag_chunks_dB.squeeze(1),
            phase_chunks.squeeze(1),
        )

        # Stack the magnitude and phase, shapes are [1, chunk_num, freq, time]
        mag_chunks = mag_chunks.permute(1, 0, 2, 3)
        mag_chunks_dB = mag_chunks_dB.permute(1, 0, 2, 3)
        phase_chunks = phase_chunks.permute(1, 0, 2, 3)
        print(f"Shape of mag: {mag_chunks.shape}, Shape of phase: {phase_chunks.shape}")

        # Reconstruct the waveform from the magnitude and phase
        waveform_reconstructed_stft = reconstruct_from_stft_chunks(
            mag_chunks,
            phase_chunks,
            padding_length,
            batch_input=False,
            config_dataloader=config_dataloader,
        )
        # Plot the waveform, magnitude and phase
        plot_all(
            waveform_reconstructed_stft,
            sr_new,
            f"./output/dev/data_preprocessing/{timestr}_chunks_{sr_new}.png",
        )
        # # Save the reconstructed waveform
        torchaudio.save(
            f"./output/dev/data_preprocessing/{timestr}_chunks_{sr_new}.wav",
            waveform_reconstructed_stft,
            sr_org,
        )
