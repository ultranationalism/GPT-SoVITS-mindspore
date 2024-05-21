import math

#import torch
#from torch.nn import functional as F
from mindspore import ops

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += ops.mean(ops.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = ops.mean((1 - dr) ** 2)
        g_loss = ops.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = ops.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * ops.exp(-2.0 * logs_p)
    kl = ops.sum(kl * z_mask)
    l = kl / ops.sum(z_mask)
    return l


def mle_loss(z, m, logs, logdet, mask):
    l = ops.sum(logs) + 0.5 * ops.sum(
        ops.exp(-2 * logs) * ((z - m) ** 2)
    )  # neg normal likelihood w/o the constant term
    l = l - ops.sum(logdet)  # log jacobian determinant
    l = l / ops.sum(
        ops.ones_like(z) * mask
    )  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l
