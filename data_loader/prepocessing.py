import time
import torch
import torchaudio
import torchaudio.transforms as T
import random
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.signal import cheby1
from scipy.signal import sosfiltfilt

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


def low_pass_filter(waveform, sr_org, sr_new):
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


def resample_audio(waveform, sr_org, sr_new):
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


def low_sr_simulation(waveform, sr_org, sr_new):
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


def concatenate_chunks(chunks: torch.Tensor, chunk_size: int, overlap: int, padding_length) -> torch.Tensor:
    """
    Concatenate the chunks into a single waveform by averaging the overlap regions and removing the padding of the last chunk.

    Args:
        chunks (torch.Tensor): A list of chunks
        chunk_size (int): The size of each chunk
        overlap (int): The overlap between chunks
        padding_length (int): The length of the padding for the last chunk

    Returns:
        torch.Tensor: The concatenated waveform
    """
    # Return the concatenated waveform if it's empty
    if len(chunks) == 0:
        return torch.tensor([])
    # Adjust total_length calculation
    if len(chunks) > 1:
        total_length = chunk_size + \
            (len(chunks) - 1) * (chunk_size - overlap) - padding_length
    else:
        total_length = chunk_size - padding_length

    # Initialize the concatenated waveform with the first chunk
    concatenated = chunks[0].clone()
    # Iterate through the rest of the chunks and concatenate them
    for i in range(1, len(chunks)):
        curr_chunk = chunks[i]
        # Handle overlap by averaging
        if overlap > 0:
            averaged_overlap = (
                concatenated[:, -overlap:] + curr_chunk[:, :overlap]) / 2
            concatenated = torch.cat(
                [concatenated[:, :-overlap], averaged_overlap, curr_chunk[:, overlap:]], dim=-1)
        else:
            concatenated = torch.cat([concatenated, curr_chunk], dim=-1)

    # print(
    #     f"Total length: {total_length}, Concatenated length: {concatenated.size(-1)}, Padding length: {padding_length}")

    # Remove the padding from the last chunk if any
    concatenated = concatenated[..., :total_length]

    # Return the concatenated waveform
    return concatenated


def plot_all(waveform, sample_rate, filename):
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
    mag, phase = get_mag_phase(waveform)
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


def get_mag_phase(waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply short time Fourier transform to the waveform and return the magnitude and phase

    Args:
        waveform (torch.Tensor): The waveform

    Returns:
        torch.Tensor: The magnitude
        torch.Tensor: The phase
    """
    # TODO: Set the parameters from the config file
    n_fft = 1024
    hop_length = 80
    win_length = 320
    window = torch.hann_window(win_length)
    # Apply short time Fourier transform to the waveform
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=True)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    # Return the magnitude and phase
    return mag, phase


# TODO: This function should be moved to postprocessing.py
def reconstruct_waveform_stft(mag: torch.Tensor, phase: torch.Tensor, padding_length: int = 0) -> torch.Tensor:
    # TODO: Set the parameters from the config file
    # Size of each audio chunk
    chunk_size = 8000
    # Overlap size between chunks
    overlap = 0
    n_fft = 1024
    hop_length = 80
    win_length = 320
    window = torch.hann_window(win_length)
    # Combine magnitude and phase to get the complex STFT
    complex_stft = mag * torch.exp(1j * phase)
    # torch.Size([1, 9, 513, 101])
    # Swap the channel 1 and channel 0 and get torch.Size([9, 1, 513, 101])
    complex_stft = complex_stft.permute(1, 0, 2, 3)

    # Prepare an empty list to store each iSTFT chunk
    waveform_chunks = []

    # Iterate over chunks and apply iSTFT
    for i in range(complex_stft.size(0)):
        # Extract the i-th chunk
        chunk = complex_stft[i]
        # Apply iSTFT
        waveform_chunk = torch.istft(
            chunk,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )

        # Append the waveform chunk to the list
        waveform_chunks.append(waveform_chunk)

    # Concatenate chunks
    waveform = concatenate_chunks(
        waveform_chunks, chunk_size, overlap, padding_length)

    return waveform


if __name__ == '__main__':
    # Test the toy_load function
    filepath = "./data/VCTK-Corpus-0.92/wav48_silence_trimmed_wav/p225/p225_001.wav"
    # Make sure the output folder exists
    ensure_dir("./output/dev/data_preprocessing")
    # Set the parameters
    # List of target sample rates to choose from
    target_sample_rates = [8000, 16000, 24000]
    # Size of each audio chunk
    chunk_size = 8000
    # Overlap size between chunks
    overlap = 0
    # Apply the audio preprocessing pipeline
    sr_new = random.choice(target_sample_rates)
    print(f"Randomly selected new sample rate: {sr_new} Hz")
    # Apply the audio preprocessing pipeline
    waveform, sr_org = torchaudio.load(filepath)
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
    waveform_reconstructed = concatenate_chunks(
        chunks, chunk_size, overlap, padding_length)

    # Save the reconstructed waveform
    torchaudio.save(
        f"./output/dev/data_preprocessing/reconstructed_{timestr}_{sr_new}.wav", waveform_reconstructed, sr_org)

    # FIX: Chunking -> STFT -> Concatenate -> iSTFT will cause high frequency peak artifacts
    # Get the magnitude and phase of the chunks
    mag = []
    phase = []
    for chunk in chunks:
        mag_chunk, phase_chunk = get_mag_phase(chunk)
        mag.append(mag_chunk)
        phase.append(phase_chunk)

    # Shapes are [1, freq, time] for both mag and phase
    print(f"Shape of mag: {mag[0].shape}, Shape of phase: {phase[0].shape}")
    # Stack the magnitude and phase, shapes are [1, chunk_num, freq, time]
    mag = torch.stack(mag, dim=1)
    phase = torch.stack(phase, dim=1)
    print(f"Shape of mag: {mag.shape}, Shape of phase: {phase.shape}")

    # Reconstruct the waveform from the magnitude and phase
    waveform_reconstructed_stft = reconstruct_waveform_stft(
        mag, phase, padding_length)
    # Save the reconstructed waveform
    torchaudio.save(
        f"./output/dev/data_preprocessing/reconstructed_stft_{timestr}_{sr_new}.wav", waveform_reconstructed_stft, sr_org)
