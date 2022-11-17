import torch
import torch.nn as nn
import torch.nn.functional as functional
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.float64
class my_model(nn.Module):
    def __init__(self, in_channels=513, out_channels=1):
        super().__init__()
        # self.cov = nn.Conv1d(in_channels=513, out_channels=1, kernel_size=1, bias=False)
        self.rnn = nn.LSTM(in_channels, 512, 1)
        self.linear = nn.Linear(512, out_channels)
        self.out = nn.Sigmoid()
    def forward(self, x):
        a = torch.relu(self.linear(self.rnn(x)[0]))
        return a

if __name__ == "__main__":
    model = my_model()
    input = torch.rand(1, 513, 2)
    
    import time
    label = torch.ones(1, 1, 2)
    W = torch.ones_like(label)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from tqdm import tqdm
    loss = torch.tensor(0.)
    # bar =  tqdm(range(10), ascii=True)
    with torch.autograd.set_detect_anomaly(True):
        for i in range(10):
            optimizer.zero_grad()
            out = model(input) # [1, 1, 2]
            a = (out * W)
            # time.sleep(0.5)
            loss = functional.mse_loss(a, label)
            W = a.detach().clone()
            loss.backward(retain_graph=True)
            print(loss.item())
            # bar.set_postfix({'loss':f'{loss.item():.5f}'})
            optimizer.step()
