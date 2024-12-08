import torch
import torch.nn as nn
from training.losses import loss_function, ssim_loss
from utils.visualization import show_reconstructions
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def calculate_conv_output_shape(input_shape, kernel_size, stride, padding, dilation=1):
    '''
    oblicza rozmiar po wartswie dla dynamicznego rozmiaru bottlenecka
    '''
    height, width = input_shape
    height = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    width = ((width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return height, width

def calculate_deconv_output_shape(input_shape, kernel_size, stride, padding, output_padding=0, dilation=1):
    height, width = input_shape
    height = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    width = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return height, width

def calculate_pool_output_shape(input_shape, kernel_size, stride, padding):
    h_in, w_in = input_shape
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    return (h_out, w_out)

class BaseAE(nn.Module):
    def __init__(self, device):
        super(BaseAE, self).__init__()
        self.device = device

    def encode(self, input):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def train_step(self, x, optimizer, scaler, ssim_kernel=11, grad_clip_value=None):
        optimizer.zero_grad()

        # forward pass
        with torch.amp.autocast(device_type=self.device.type):
            recon = self.forward(x)
            # losses = loss_function(recon, x)
            losses = ssim_loss(recon, x, ssim_kernel)
            
        # _______________________________________________________ L1 regularyzacja TEST
        l1_reg = 0
        for param in self.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param)) 
        losses += l1_reg * .000001     # współczynnik regularyzacji
        # _______________________________________________________ TEST END
        
        # backward pass ze skalowaniem
        scaler.scale(losses).backward()
        
        scaler.unscale_(optimizer)

        # clipping gradientów
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_value)

        # krok optymalizatora i aktualizacja skalera
        scaler.step(optimizer)
        scaler.update()

        # obliczenie normy gradientów
        grad_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt().item()

        return losses, grad_norm
    
    def evaluate(self, data_loader, ssim_kernel):
        total_loss = 0
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                with torch.amp.autocast(device_type=self.device.type):
                    recon = self.forward(batch_data)
                    losses = ssim_loss(recon, batch_data, kernel_size=ssim_kernel)
                total_loss += losses.item()
        avg_loss = total_loss / len(data_loader)
        return avg_loss


    def log_and_visualize(self, fig, axs, epoch, losses, data, color_mode):
        
        # wizualizacja rekonstrukcji
        show_reconstructions(fig, axs, self, data[:1], epoch, color_mode)
                