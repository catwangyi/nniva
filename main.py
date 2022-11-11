import torch
import torchaudio


complex_tensor = torch.tensor(3.2749e+10+0.j, dtype=torch.complex64)
root = torch.sqrt(complex_tensor)
print(complex_tensor)
print(root)