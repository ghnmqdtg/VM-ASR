import torch
import torch.nn.functional as F


def mae_loss(output, target):
    return F.l1_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


# ==================== MultiResolutionSTFTLoss ==================== #
# REF: Parallel WaveGAN
# URL: https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/losses/stft_loss.py
# ================================================================= #
def stft(x, fft_size, hop_size, win_length, window, emphasize_high_freq=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
        emphasize_high_freq (bool): Whether to emphasize high frequency.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window=window, return_complex=True
    )
    # View as real
    x_stft = torch.view_as_real(x_stft)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    magnitude = torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)

    if emphasize_high_freq:
        # Create a linear weight scale that increases from 1 to 2 across the frequency axis
        freq_weights = torch.linspace(1.0, 2.0, magnitude.size(1), device=x.device)
        freq_weights = freq_weights.view(1, -1, 1)
        magnitude = magnitude * freq_weights

    return magnitude


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        emphasize_high_freq=False,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.emphasize_high_freq = emphasize_high_freq
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        self.window = self.window.to(x.device)
        x_mag = stft(
            x,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            self.emphasize_high_freq,
        )
        y_mag = stft(
            y,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            self.emphasize_high_freq,
        )
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        factor_sc=0.1,
        factor_mag=0.1,
        emphasize_high_freq=False,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, emphasize_high_freq)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


# ==================== GAN Losses ==================== #
class HiFiGANLoss:
    """
    HiFi-GAN Loss module.

    URL: https://github.com/jik876/hifi-gan/blob/master/models.py
    """

    def __init__(self):
        super(HiFiGANLoss, self).__init__()

    def discriminator_loss(self, real_data, generated_data):
        loss = 0
        for dr, dg in zip(real_data, generated_data):
            r_loss = torch.mean((dr - 1) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss

        return loss

    def generator_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            loss += l

        return loss

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2
