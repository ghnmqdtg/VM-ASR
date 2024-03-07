import time
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import random
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.signal import cheby1
from scipy.signal import sosfiltfilt
from postpocessing import concatenate_wave_chunks, reconstruct_from_stft, reconstruct_from_stft_chunks

try:
    from utils import ensure_dir
except:
    import os
    import sys
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from utils import ensure_dir


def crop_or_pad_waveform(waveform: torch.Tensor) -> torch.Tensor:
        # TODO: Set the length from the config file
        length = 121890
        # If the waveform is shorter than the required length, pad it
        if waveform.shape[1] < length:
            pad_length = length - waveform.shape[1]
            # If the phase is training or validation, pad the waveform randomly
            # If the phase is testing, pad the waveform from the beginning
            r = random.randint(0, pad_length)
            # Pad the waveform with zeros to the left and right
            # Left: random length between 0 and pad_length, Right: pad_length - r
            waveform = F.pad(waveform, (r, pad_length - r),
                             mode='constant', value=0)
            # print(f"Pad length: {pad_length}, Random length: {r}")
        else:
            # If the waveform is longer than the required length, crop it randomly from the beginning
            start = random.randint(0, waveform.shape[1] - length)
            # Crop the waveform from start to start + length (fixed length)
            waveform = waveform[:, start:start + length]
            # print(f"Crop to length: {length}, Start: {start}")

        # print(f'New shape of padded or cropped waveform: {waveform.shape}')
        return waveform


def low_pass_filter(waveform: torch.Tensor, sr_org: int, sr_new: int) -> torch.Tensor:
    """
    Apply low pass filter to the waveform to remove the high frequency components.
    This can avoid aliasing when downsampling the waveform.

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int): The new sample rate

    Returns:
        torch.Tensor: The filtered waveform
    """
    # Define the cutoff frequency and the ratio to the Nyquist frequency
    nyquist = sr_org / 2
    highcut = sr_new // 2
    hi = highcut / nyquist
    # TODO: Set filter funtion as a parameter
    # Setup the low pass filter: chebyshev type 1
    sos = cheby1(8, 0.1, hi, btype='lowpass', output='sos')
    # Apply the filter
    waveform = sosfiltfilt(sos, waveform)
    # Convert the waveform to tensor
    waveform_tensor = torch.tensor(waveform.copy(), dtype=torch.float32)
    # Return the waveform
    return waveform_tensor


def resample_audio(waveform: torch.Tensor, sr_org: int, sr_new: int) -> torch.Tensor:
    """
    Downsample the waveform to the new sample rate

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int): The new sample rate

    Returns:
        torch.Tensor: The downsampled waveform
    """
    waveform_downsampled = T.Resample(
        sr_org, sr_new)(waveform)
    return waveform_downsampled


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


def cut2chunks(waveform: torch.Tensor, chunk_size: int, overlap: int, return_padding_length: bool = False) -> torch.Tensor | tuple[torch.Tensor, int]:
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
    # Initialize the list of chunks
    chunks = []
    # Get the length of the waveform
    length = waveform.size(-1)
    # Step size
    step_size = chunk_size - overlap

    start = 0
    padding_length = 0
    # Cut the waveform into chunks, finish when the last chunk is smaller than chunk_size
    while start + chunk_size <= length:
        chunks.append(waveform[..., start:start + chunk_size])
        start += step_size

    # Pad the last chunk with zeros
    if start < length:
        last_chunk = waveform[..., start:]
        # If padding is required, pad the last chunk to the chunk_size
        padding_length = chunk_size - last_chunk.size(-1)
        last_chunk = torch.nn.functional.pad(
            last_chunk, (0, padding_length), 'constant', 0)
        chunks.append(last_chunk)

    # print(
    #     f"Length: {length}, Chunk number: {len(chunks)}, Padding length: {padding_length}")

    if return_padding_length:
        return torch.stack(chunks), padding_length
    else:
        return torch.stack(chunks)


def low_sr_simulation_pipeline(waveform: torch.Tensor, sr_org: int, sr_new: int | list[int], chunk_size: int, overlap: int) -> list[torch.Tensor]:
    """
    Apply the low sample rate simulation pipeline to the waveform

    Args:
        waveform (torch.Tensor): The input waveform
        sr_org (int): The original sample rate
        sr_new (int | list[int]): The new sample rate
        chunk_size (int): The size of each chunk
        overlap (int): The overlap between chunks

    Returns:
        List[torch.Tensor]: A list of chunks
    """
    # Check if sr_new is a list
    if isinstance(sr_new, list):
        # Randomly select a new sample rate
        sr_new = random.choice(sr_new)
    # Apply the low sample rate simulation
    waveform_lr = low_sr_simulation(waveform, sr_org, sr_new)
    # Cut the waveform into chunks
    chunks = cut2chunks(waveform_lr, chunk_size, overlap)
    # Return the list of chunks
    return chunks


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
    mag, phase = get_mag_phase(waveform, chunk_wave=False)
    # Plot the waveform, magnitude and phase
    plt.figure(figsize=(12, 4))
    # Add title to the figure
    plt.suptitle(f"Waveform, Magnitude and Phase at {sample_rate} Hz")
    plt.subplot(1, 3, 1)
    plt.plot(waveform.t().numpy())
    plt.title("Waveform")
    plt.subplot(1, 3, 2)
    plt.imshow(mag.log2().numpy().squeeze(0), aspect='auto', origin='lower')
    plt.title("Magnitude")
    plt.subplot(1, 3, 3)
    plt.imshow(phase.numpy().squeeze(0), aspect='auto', origin='lower')
    plt.title("Phase")
    plt.savefig(filename)
    plt.close()


def get_mag_phase(waveform: torch.Tensor, chunk_wave: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply short time Fourier transform to the waveform and return the magnitude and phase

    Args:
        waveform (torch.Tensor): The waveform

    Returns:
        torch.Tensor: The magnitude
        torch.Tensor: The phase
    """
    # TODO: Set the parameters from the config file
    if chunk_wave:
        n_fft = 1022
        hop_length = 80
        win_length = 320
        window = torch.hann_window(win_length)
    else:
        n_fft = 1022
        hop_length = 478
        win_length = 956
        window = torch.hann_window(win_length)
    # Apply short time Fourier transform to the waveform
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=True)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    # Return the magnitude and phase
    return mag, phase


if __name__ == '__main__':
    # Test the toy_load function
    filepath = "./data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p225/p225_001.wav"
    # Make sure the output folder exists
    ensure_dir("./output/dev/data_preprocessing")
    # Set the parameters
    # List of target sample rates to choose from
    target_sample_rates = [8000, 16000, 24000]
    # Size of each audio chunk
    chunk_size = 10160
    # Overlap size between chunks
    overlap = 0
    # Apply the audio preprocessing pipeline
    sr_new = random.choice(target_sample_rates)
    print(f"Randomly selected new sample rate: {sr_new} Hz")
    # Apply the audio preprocessing pipeline
    waveform, sr_org = torchaudio.load(filepath)
    # Crop or pad the waveform to the required length
    waveform = crop_or_pad_waveform(waveform)
    waveform_filtered = low_pass_filter(waveform, sr_org, sr_new)
    waveform_downsampled = resample_audio(waveform_filtered, sr_org, sr_new)
    # Apply upsampling to get a unified sample rate as input
    waveform_upsampled = resample_audio(waveform_downsampled, sr_new, sr_org)
    # Cut the waveform into chunks
    chunks, padding_length = cut2chunks(
        waveform_upsampled, chunk_size, overlap, return_padding_length=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Plot the waveform, magnitude and phase
    plot_all(waveform_upsampled, sr_new,
             f"./output/dev/data_preprocessing/lr_simulation_{sr_new}_{timestr}.png")
    # # Save the chunks to the output folder
    # for i, chunk in enumerate(chunks):
    #     torchaudio.save(
    #         f"./output/dev/data_preprocessing/chunk_{timestr}_{i}_{sr_new}.wav", chunk, sr_org)
    print(
        f"Processed {len(chunks)} chunks from {filepath} at {sr_new} Hz sample rate")

    # Reconstruct the waveform from the chunks
    waveform_reconstructed = concatenate_wave_chunks(
        chunks, chunk_size, overlap, padding_length)

    # Save the reconstructed waveform
    torchaudio.save(
        f"./output/dev/data_preprocessing/reconstructed_wave_chunks_{timestr}_{sr_new}.wav", waveform_reconstructed, sr_org)


    # Get the magnitude and phase of the full waveform
    mag, phase = get_mag_phase(waveform, chunk_wave=False)
    # Print the shapes of the magnitude and phase
    print(f"Shape of mag: {mag.shape}, Shape of phase: {phase.shape}")
    # Reconstruct the waveform from the magnitude and phase
    waveform_reconstructed_stft = reconstruct_from_stft(mag, phase)
    # Save the reconstructed waveform
    torchaudio.save(
        f"./output/dev/data_preprocessing/reconstructed_stft_full_{timestr}_{sr_new}.wav", waveform_reconstructed_stft, sr_org)


    # FIX: Chunking -> STFT -> Concatenate -> iSTFT will cause high frequency peak artifacts
    # Get the magnitude and phase of the chunks
    mag = []
    phase = []
    for chunk in chunks:
        mag_chunk, phase_chunk = get_mag_phase(chunk, chunk_wave=True)
        mag.append(mag_chunk)
        phase.append(phase_chunk)

    # Shapes are [1, freq, time] for both mag and phase
    print(f"Shape of mag: {mag[0].shape}, Shape of phase: {phase[0].shape}")
    # Stack the magnitude and phase, shapes are [1, chunk_num, freq, time]
    mag = torch.stack(mag, dim=1)
    phase = torch.stack(phase, dim=1)
    print(f"Shape of mag: {mag.shape}, Shape of phase: {phase.shape}")

    # Reconstruct the waveform from the magnitude and phase
    waveform_reconstructed_stft = reconstruct_from_stft_chunks(
        mag, phase, padding_length)
    # Save the reconstructed waveform
    torchaudio.save(
        f"./output/dev/data_preprocessing/reconstructed_stft_chunks_{timestr}_{sr_new}.wav", waveform_reconstructed_stft, sr_org)
