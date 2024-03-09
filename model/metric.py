import torch
import torch.nn.functional as F


def psnr(output, target, max_val=32767):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between the predicted and target waveforms.

    Args:
        output (torch.Tensor): The predicted waveform.
        target (torch.Tensor): The target waveform.
        max_val (float): The maximum value of the input waveforms (e.g., 32767 for 16-bit audio)

    Returns:
        torch.Tensor: The PSNR value for each example in the batch.
    """
    mse = F.mse_loss(output, target)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    # Compute the mean PSNR for the whole batch and convert to float
    psnr = psnr.item()
    return psnr


def log_spectral_distance(output_mag, target_mag):
    """
    Compute the Log-Spectral Distance (LSD) between the predicted and target waveforms.

    Args:
        output_mag (torch.Tensor): The magnitude of the predicted waveform.
        target_mag (torch.Tensor): The magnitude of the target waveform.

    Returns:
        torch.Tensor: The LSD value for each example in the batch.
    """
    # Add a small epsilon to avoid log of zero
    epsilon = 1e-8
    output_mag = torch.clamp(output_mag, min=epsilon)
    target_mag = torch.clamp(target_mag, min=epsilon)

    # Compute the log-spectral distance
    log_diff = torch.log10(output_mag) - torch.log10(target_mag)
    # Mean over frequency bins and then mean over time frames
    lsd = torch.mean(torch.sqrt(torch.mean(log_diff**2, dim=-1)), dim=-1)
    # Compute the mean LSD for the whole batch and convert to float
    lsd = torch.mean(lsd).item()

    return lsd
