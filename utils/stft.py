from typing import Tuple

import torch
import torchaudio
import torchaudio.transforms as T


class DBToAmplitude(T.AmplitudeToDB):
    """
    Convert the input from dB to amplitude. It's not implemented in torchaudio.transforms.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Set the power to 1 to convert from dB to power (multiplier is set to 10.0)
        # In our project, we convert the magnitude to dB with "power" in AmplitudeToDB
        return torchaudio.functional.DB_to_amplitude(x, ref=1, power=1)


def wav2spectro(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    spectro_scale: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply short time Fourier transform to the waveform and return the magnitude and phase
    """
    # Define the STFT parameters
    n_fft = n_fft
    hop_length = hop_length
    win_length = win_length
    window = torch.hann_window(win_length).to(waveform.device)

    *other, length = waveform.shape
    # Remove the channel dimension
    waveform = waveform.reshape(-1, length)

    # Apply the short time Fourier transform
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        return_complex=True,
    )

    _, freqs, frame = spec.shape

    # Get the magnitude and phase
    if spectro_scale == "dB":
        # The AmplitudeToDB default input is in power |x|^2
        mag = T.AmplitudeToDB(stype="power", top_db=80)(torch.abs(spec).pow(2))
        phase = torch.angle(spec)
    else:
        # Magnitude is calculated as the absolute value, and log2 is applied to compress the dynamic range
        mag = torch.log2(torch.abs(spec) + 1e-8)
        phase = torch.angle(spec)

    return mag.view(*other, freqs, frame), phase.view(*other, freqs, frame)


def spectro2wav(
    mag: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    spectro_scale: str,
) -> torch.Tensor:
    """
    Apply inverse short time Fourier transform to the magnitude and phase and return the waveform
    """
    # Define the STFT parameters
    # n_fft = n_fft
    hop_length = hop_length
    win_length = win_length
    window = torch.hann_window(win_length).to(mag.device)

    *other, freqs, frames = mag.shape
    n_fft = 2 * freqs - 2

    mag = mag.view(-1, freqs, frames)
    phase = phase.view(-1, freqs, frames)

    # Apply the inverse short time Fourier transform
    if spectro_scale == "dB":
        # The DBToAmplitude default input is in power |x|^2
        spec = DBToAmplitude()(mag).pow(0.5) * torch.exp(1j * phase)
    else:
        # Inverse log2 to get the magnitude
        spec = torch.exp2(mag) * torch.exp(1j * phase)

    # Apply the inverse short time Fourier transform
    waveform = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
    )

    _, length = waveform.shape

    return waveform.view(*other, length)
