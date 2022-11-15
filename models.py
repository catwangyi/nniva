from numpy import intp
import torch
import torch.nn as nn
import torch.nn.functional as functional
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.float64
class my_model(nn.Module):
    def __init__(self, in_channels=513, out_channels=1):
        super().__init__()
        self.rnn = nn.LSTM(in_channels, 512)
        self.linear = nn.Linear(512, out_channels)
        self.out = torch.sigmoid
    def forward(self, x):
        a = self.rnn(x)[0]
        out = self.linear(a)
        return self.out(out)

if __name__ == "__main__":
    model = my_model(513, 513)
    input = torch.rand(2, 45, 513)
    W = torch.ones_like(input)
    temp_w = torch.ones_like(W)
    import time
    label = torch.ones_like(input)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from tqdm import tqdm
    loss = torch.tensor(0.)
    bar =  tqdm(range(10), ascii=True)
    for i in bar:
        optimizer.zero_grad()
        out = model(input)
        temp = W[0] * out[0]
        temp_w[0] = temp
        time.sleep(0.5)
        loss = functional.mse_loss(temp_w, label)
        loss.backward(retain_graph=True)
        bar.set_postfix({'loss':f'{loss.item():.5f}'})
        optimizer.step()
