import torch
import torch.nn.functional as F


def mae_loss(output, target):
    return F.l1_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    total_n_layers = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            total_n_layers += 1
            loss += torch.mean(torch.abs(rl - gl))

    return loss / total_n_layers


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss

    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l

    return loss
