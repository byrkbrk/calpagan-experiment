from utils import torch, nn
"""
Note these loss functions implement GAN + L1 (not cGAN + L1)"""

adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 400
lambda_non_zero = 0


def get_disc_loss(disc, real, fake):
    # Send relu-masked fake to disc
    relu = nn.ReLU()
    output_fake = disc(relu(fake.detach()))
    output_real = disc(real)

    loss_fake = adv_criterion(output_fake, torch.zeros_like(output_fake))
    loss_real = adv_criterion(output_real, torch.ones_like(output_real))
    return (loss_fake + loss_real)/2


def get_gen_loss(disc, real, fake):
    # Send relu-masked fake to discriminator
    relu = nn.ReLU()
    output_fake = disc(relu(fake))
    adv_fake = adv_criterion(output_fake, torch.ones_like(output_fake))
    non_zero_locations = real > 0
    
    non_zero_loss = lambda_non_zero*adv_criterion(real[non_zero_locations], fake[non_zero_locations])
    recon_loss = lambda_recon*recon_criterion(real, fake)
    return recon_loss + adv_fake + non_zero_loss, recon_loss, adv_fake
