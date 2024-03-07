import torch


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


def reconstruct_waveform_stft(mag: torch.Tensor, phase: torch.Tensor, padding_length: int = 0) -> torch.Tensor:
    # TODO: Set the parameters from the config file
    # Size of each audio chunk
    chunk_size = 10160
    # Overlap size between chunks
    overlap = 0
    n_fft = 1022
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