import torch
import torch.nn.functional as F
import torch.nn as nn

# z założenia wszystkie straty są normalizowane do [0, 1]

def loss_function(recon_x, x):
    """
    oblicza stratę dla wariacyjnego autoenkodera (VAE).
    zwraca MSE i KLD (dla enkoderów determ. KLD=0)
    """
    batch_size = x.size(0)
    num_pixels = x.size(1) * x.size(2) * x.size(3)

    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (batch_size * num_pixels)
    # recon_loss = F.l1_loss(recon_x, x, reduction='sum') / (batch_size * num_pixels)

    return recon_loss

def ssim_loss(recon_x, x, kernel_size=15):
    batch_size = x.size(0)
    num_pixels = x.size(1) * x.size(2) * x.size(3)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # średnie przez kernel z podanym rozmiarem i odpowiednim paddingiem
    padding = kernel_size // 2
    mu1 = F.avg_pool2d(recon_x, kernel_size, 1, padding)
    mu2 = F.avg_pool2d(x, kernel_size, 1, padding)

    # odchylenia standardowe
    sigma1 = F.avg_pool2d(recon_x ** 2, kernel_size, 1, padding) - mu1 ** 2
    sigma2 = F.avg_pool2d(x ** 2, kernel_size, 1, padding) - mu2 ** 2
    sigma12 = F.avg_pool2d(recon_x * x, kernel_size, 1, padding) - mu1 * mu2

    # SSIM numerator i denominator
    ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    # obliczanie SSIM
    ssim = ssim_n / ssim_d
    
    # strata rekonstrukcji
    recon_loss = torch.clamp((1 - ssim) / 2, 0, 1).mean()
    
    # test - strata łączona - SSIM + MSR
    # recon_loss += 1 * F.l1_loss(recon_x, x, reduction='sum') / (batch_size * num_pixels)
    recon_loss += 1 * F.mse_loss(recon_x, x, reduction='sum') / (batch_size * num_pixels)
    # recon_loss = recon_loss / 2
    # koniec testu 

    return recon_loss

from torchmetrics.functional import structural_similarity_index_measure as ssim

def ssim_loss1(recon_x, x, kernel_size=11):
    # funkcja nie działa prawidłowo
    ssim_value = ssim(recon_x, x, data_range=1.0, kernel_size=kernel_size)
    recon_loss = (1 - ssim_value) / 2
    return recon_loss
