import torch
import torch.nn.functional as F


def mae_loss(output, target):
    return F.l1_loss(output, target)


def mse_loss(output, target, config):
    # Create a Hann window
    window = torch.hann_window(config.DATA.STFT.WIN_LENGTH).to(output.device)
    # Flatten the output and target waves
    output = output.view(-1, output.size(-1))
    target = target.view(-1, target.size(-1))
    # Convert the output and target waves to spectrograms
    output_spec = torch.stft(
        output,
        n_fft=config.DATA.STFT.N_FFT,
        hop_length=config.DATA.STFT.HOP_LENGTH,
        win_length=config.DATA.STFT.WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    target_spec = torch.stft(
        target,
        n_fft=config.DATA.STFT.N_FFT,
        hop_length=config.DATA.STFT.HOP_LENGTH,
        win_length=config.DATA.STFT.WIN_LENGTH,
        window=window,
        return_complex=True,
    )

    output_power = torch.log(torch.abs(output_spec).pow(2) + 1e-8)
    target_power = torch.log(torch.abs(target_spec).pow(2) + 1e-8)

    return F.mse_loss(output_power, target_power)


# ==================== MultiResolutionSTFTLoss ==================== #
# REF: Parallel WaveGAN
# URL: https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/losses/stft_loss.py
# ================================================================= #
def stft(waveform, fft_size, hop_size, win_length, window, return_phase=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        waveform (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
        return_phase (bool): Whether to return phase tensor or not.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    spec = torch.stft(
        waveform,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    # View as real
    spec_real = torch.view_as_real(spec)

    magnitude = torch.sqrt(
        torch.clamp(spec_real[..., 0] ** 2 + spec_real[..., 1] ** 2, min=1e-7)
    ).transpose(2, 1)
    phase = torch.angle(spec) if return_phase else None

    return magnitude, phase


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


class STFTPhaseLoss(torch.nn.Module):
    """STFT phase loss module."""

    def __init__(self):
        """Initilize STFT phase loss module."""
        super(STFTPhaseLoss, self).__init__()

    def forward(self, x_phase, y_phase):
        """Calculate forward propagation.
        Args:
            x_phase (Tensor): Phase spectrogram of predicted signal (B, #frames, #freq_bins).
            y_phase (Tensor): Phase spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: STFT phase loss value.
        """
        return F.l1_loss(y_phase, x_phase)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        return_phase=False,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.return_phase = return_phase
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        if return_phase:
            self.phase_loss = STFTPhaseLoss()

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
        x_mag, x_phase = stft(
            x,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            self.return_phase,
        )
        y_mag, y_phase = stft(
            y,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            self.return_phase,
        )
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        if self.return_phase:
            phase_loss = self.phase_loss(x_phase, y_phase)
            return sc_loss, mag_loss, phase_loss
        else:
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
        factor_phase=0.0,
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
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase
        self.return_phase = factor_phase > 0.0
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, self.return_phase)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        if self.return_phase:
            sc_loss = 0.0
            mag_loss = 0.0
            phase_loss = 0.0
            for f in self.stft_losses:
                sc_l, mag_l, ph_l = f(x, y)
                sc_loss += sc_l
                mag_loss += mag_l
                phase_loss += ph_l
            sc_loss /= len(self.stft_losses)
            mag_loss /= len(self.stft_losses)
            phase_loss /= len(self.stft_losses)

            return (
                self.factor_sc * sc_loss,
                self.factor_mag * mag_loss,
                self.factor_phase * phase_loss,
            )
        else:
            sc_loss = 0.0
            mag_loss = 0.0
            for f in self.stft_losses:
                sc_l, mag_l = f(x, y)
                sc_loss += sc_l
                mag_loss += mag_l
            sc_loss /= len(self.stft_losses)
            mag_loss /= len(self.stft_losses)

            return self.factor_sc * sc_loss, self.factor_mag * mag_loss, None


# ==================== GAN Losses ==================== #
class HiFiGANLoss:
    """
    HiFi-GAN Loss module.

    URL: https://github.com/jik876/hifi-gan/blob/master/models.py
    """

    def __init__(self, gan_loss_type):
        super(HiFiGANLoss, self).__init__()
        self.gan_loss_type = gan_loss_type

    def discriminator_loss(self, real_data, generated_data):
        loss = 0
        if self.gan_loss_type == "lsgan":
            for dr, dg in zip(real_data, generated_data):
                r_loss = torch.mean((dr - 1) ** 2)
                g_loss = torch.mean(dg**2)
                loss += r_loss + g_loss
        elif self.gan_loss_type == "wgan" or self.gan_loss_type == "wgan-gp":
            for dr, dg in zip(real_data, generated_data):
                r_loss = -torch.mean(dr)
                g_loss = torch.mean(dg)
                loss += r_loss + g_loss
        elif self.gan_loss_type == "hinge":
            for dr, dg in zip(real_data, generated_data):
                r_loss = torch.nn.ReLU()(1.0 - dr).mean()
                g_loss = torch.nn.ReLU()(1.0 + dg).mean()
                loss += r_loss + g_loss

        return loss

    def generator_loss(self, disc_outputs):
        loss = 0
        if self.gan_loss_type == "lsgan":
            for dg in disc_outputs:
                l = torch.mean((1 - dg) ** 2)
                loss += l
        elif self.gan_loss_type == "wgan" or self.gan_loss_type == "wgan-gp":
            for dg in disc_outputs:
                loss += -torch.mean(dg)
        elif self.gan_loss_type == "hinge":
            for dg in disc_outputs:
                loss += -torch.mean(dg)

        return loss

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        total_n_layers = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                total_n_layers += 1
                loss += torch.mean(torch.abs(rl - gl))

        return loss / total_n_layers
