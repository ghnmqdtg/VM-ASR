import torch


def unfold_audio(audio, segment_length, overlap):
    # Unfold the audio tensor into overlapping segments
    segments = audio.unfold(
        dimension=-1, size=segment_length, step=segment_length - overlap
    )
    return segments


def fold_audio(segments, total_length, segment_length, overlap):
    # Calculate step size
    step = segment_length - overlap

    # Get the batch size and number of segments
    batch_size, channels, num_segments, _ = segments.size()

    # Initialize the reconstructed audio tensor
    reconstructed = torch.zeros(batch_size, channels, total_length).to(segments.device)
    count = torch.zeros(batch_size, channels, total_length).to(segments.device)

    # Accumulate outputs from the segments
    for i in range(num_segments):
        start = i * step
        end = start + segment_length
        reconstructed[:, :, start:end] += segments[:, :, i]
        count[:, :, start:end] += 1

    # Avoid division by zero
    count[count == 0] = 1
    reconstructed /= count
    return reconstructed


if __name__ == "__main__":
    # Example Usage
    batch_size, channels, length = 1, 1, 81760
    audio = torch.randn(batch_size, channels, length)
    print(f"Original Audio Shape: {audio.shape}")  # torch.Size([1, 1, 81760])

    segment_length = 40880
    overlap = 2000
    segments = unfold_audio(audio, segment_length, overlap)
    print(f"Segmented Audio Shape: {segments.shape}")  # torch.Size([1, 1, 2, 40880])

    # Process segments
    processed_segments = torch.zeros_like(segments)
    for i in range(segments.size(2)):
        processed_segments[:, :, i] = segments[:, :, i]
        print(f"Processed Segment {i} Shape: {segments[:, :, i].shape}")

    # Fold the processed segments back into the full audio
    reconstructed_audio = fold_audio(
        processed_segments, length, segment_length, overlap
    )
    print(
        f"Reconstructed Audio Shape: {reconstructed_audio.shape}"
    )  # torch.Size([1, 1, 81760])
