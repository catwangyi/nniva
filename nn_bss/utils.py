import torch
import numpy as np


def mag(x, eps=1e-9):
    return torch.sqrt(x.real**2+x.imag**2+eps)


def stft(input, N_fft = 512, hop_length = 256):
    if input.dim() == 2:
        input = input.unsqueeze(1)
    assert input.dim() == 3
    device = input.device
    window = torch.hann_window(N_fft).to(device)
    B,C,T = input.shape
    x = input.view(B*C, T) # [B, C, T] -> [B*C, T]
    output = torch.stft(x, 
                        n_fft=N_fft, 
                        hop_length=hop_length, 
                        window = window, 
                        return_complex=True
                        ) # [B*C, T] -> [B*C, frequecies, frames]
    _, fre, time = output.shape
    output = output.reshape(B, C, fre, time) # [B*C, T] -> [B, C, frequecies, frames]
    return output

def istft(input, N_fft = 512, hop_length = 256):
    # input :torch.complex64 shape:[B, C, fre, time]
    if input.dim() == 3:
        input.unsqueeze(1)
    B, C, fre, time = input.shape
    input = input.reshape(B*C, fre, time)
    device = input.device
    window = torch.hann_window(N_fft).to(device)
    _, fre, time = input.shape
    output = torch.istft(input, 
                        n_fft = N_fft, 
                        hop_length = hop_length, 
                        window = window
                        ) # [B*C ,frequecies, frames] -> [B*C, T]
    output = output.reshape(B, C, -1)
    return output