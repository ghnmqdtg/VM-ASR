import torch
import torch.nn.functional as F


def stft(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(
        audio, n_fft, hop_length, window=hann_window, return_complex=True
    )
    # Compute the power of the complex STFT
    spec = torch.sqrt(stft_spec.real.pow(2) + stft_spec.imag.pow(2))
    return spec


def snr(output, target, **kwargs):
    snr = (
        20
        * torch.log10(
            torch.norm(target, dim=-1)
            / torch.norm(output - target, dim=-1).clamp(min=1e-8)
        )
    ).mean()
    return snr.item()


def lsd(output, target, **kwargs):
    sp = torch.log10(stft(output).square().clamp(1e-8))
    st = torch.log10(stft(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean().item()


def lsd_hf(output, target, hf):
    sp = torch.log10(stft(output).square().clamp(1e-8))
    st = torch.log10(stft(target).square().clamp(1e-8))
    val = []
    for i in range(output.size(0)):
        hf_i = hf[i].item()
        val.append(
            (
                (sp[i, hf_i:, :] - st[i, hf_i:, :])
                .square()
                .mean(dim=0)
                .sqrt()
                .mean()
                .item()
            )
        )
    return torch.tensor(val).mean().item()


def lsd_lf(output, target, hf):
    sp = torch.log10(stft(output).square().clamp(1e-8))
    st = torch.log10(stft(target).square().clamp(1e-8))
    val = []
    for i in range(output.size(0)):
        hf_i = hf[i].item()
        val.append(
            (
                (sp[i, :hf_i, :] - st[i, :hf_i, :])
                .square()
                .mean(dim=0)
                .sqrt()
                .mean()
                .item()
            )
        )
    return torch.tensor(val).mean().item()
