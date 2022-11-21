import torch
from torch import nn
import torch.nn.functional as functional
complex_type = torch.complex128
real_type = torch.float64
from linalg import divide, mag_sq
torch.set_default_tensor_type(torch.DoubleTensor)

class gate_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
        ) 
    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))

class gate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
        ) 
    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))

class crn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate1 = gate_conv(in_channels=2, out_channels=16, kernel_size=3, stride=2) # 256
        self.gate2 = gate_conv(in_channels=16, out_channels=32, kernel_size=3, stride=2) # 127
        self.gate3 = gate_conv(in_channels=32, out_channels=64, kernel_size=3, stride=2) # 63
        self.lstm = nn.LSTM(64, 128, 1, bidirectional=True)
        self.dnn = nn.Linear(256, 64)
        self.degate3 = gate_deconv(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.degate2 = gate_deconv(in_channels=32, out_channels=16, kernel_size=3, stride=2)
        self.degate1 = gate_deconv(in_channels=16, out_channels=1, kernel_size=3, stride=2)
        

    def forward(self, x):
        power = torch.mean((x.real)**2 + (x.imag)**2 + 1e-12, dim=[-1], keepdim=True)
        inpt_real = torch.log10(torch.abs(x.real) / torch.sqrt(power) + 1e-12)
        inpt_imag = torch.log10(torch.abs(x.imag) / torch.sqrt(power) + 1e-12)
        inpt = torch.cat([inpt_real, inpt_imag], dim=-2)
        enc1 = self.gate1(inpt) # 256
        enc2 = self.gate2(functional.pad(enc1, pad=[1, 0])) # 128
        enc3 = self.gate3(functional.pad(enc2, pad=[1, 0])) # 64
        enh = self.dnn(self.lstm(enc3)[0])
        dec3 = self.degate3(enh)
        dec2 = self.degate2(dec3)[..., :-2]
        dec1 = self.degate1(dec2)[..., :-2]
        return dec1


class my_model(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.dnn = nn.LSTM(513, 256, 4, bidirectional=True).type(real_type)
        # self.dnn2 = nn.Linear(512, 513)
        self.dnn1 = nn.Linear(in_channels, 256).type(real_type)
        self.norm1 = nn.LayerNorm(256).type(real_type)
        self.dnn2 = nn.Linear(256, 128).type(real_type)
        self.norm2 = nn.LayerNorm(128).type(real_type)
        self.dnn3 = nn.Linear(128, 1).type(real_type)
        # self.norm3 = nn.LayerNorm(1).type(real_type)

        nn.init.xavier_uniform_(self.dnn1.weight)
        nn.init.xavier_uniform_(self.dnn2.weight)
        nn.init.xavier_uniform_(self.dnn3.weight)

    def forward(self, x):
        power = torch.mean((x.real)**2 + (x.imag)**2 + 1e-12, dim=[-1], keepdim=True)
        inpt_real = torch.log10(torch.abs(x.real) / torch.sqrt(power) + 1e-12)
        inpt_imag = torch.log10(torch.abs(x.imag) / torch.sqrt(power) + 1e-12)
        inpt = torch.cat([inpt_real, inpt_imag], dim=-2)
        x1 = torch.relu(self.norm1(self.dnn1(inpt)))
        x2 = torch.relu(self.norm2(self.dnn2(x1)))
        x3 = torch.sigmoid(self.dnn3(x2))
        return x3

class GLULayer(nn.Module):
    def __init__(
        self, n_input, n_output, n_sublayers=3, kernel_size=3, pool_size=2, eps=1e-5
    ):
        super().__init__()

        self.args = (n_input, n_output)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "kernel_size": kernel_size,
            "pool_size": pool_size,
        }

        lin_bn_layers = []
        lin_layers = []
        gate_bn_layers = []
        gate_layers = []
        pool_layers = []

        conv_type = nn.Conv1d if n_input >= n_output else nn.ConvTranspose1d
        # conv_type = nn.Conv1d

        for n in range(n_sublayers):
            n_out = n_output if n == n_sublayers - 1 else n_input

            lin_layers.append(
                conv_type(
                    in_channels=n_input,
                    out_channels=pool_size * n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            lin_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))

            gate_layers.append(
                conv_type(
                    in_channels=n_input,
                    out_channels=pool_size * n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            gate_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))

            pool_layers.append(
                nn.MaxPool1d(
                    kernel_size=pool_size,
                )
            )

            self.lin_layers = nn.ModuleList(lin_layers)
            self.lin_bn_layers = nn.ModuleList(lin_bn_layers)
            self.gate_layers = nn.ModuleList(gate_layers)
            self.gate_bn_layers = nn.ModuleList(gate_bn_layers)
            self.pool_layers = nn.ModuleList(pool_layers)

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints(self):
        pass

    def forward(self, X):

        for lin, lin_bn, gate, gate_bn, pool in zip(
            self.lin_layers,
            self.lin_bn_layers,
            self.gate_layers,
            self.gate_bn_layers,
            self.pool_layers,
        ):
            G = gate(X)
            G = gate_bn(G)
            G = torch.sigmoid(G)
            X = lin(X)
            X = lin_bn(X)
            X = G * X
            X = pool(X.transpose(1, 2)).transpose(1, 2)

        return X


class GLUMask(nn.Module):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        mag_spec=True,
        log_spec=True,
        n_sublayers=1,
    ):
        super().__init__()

        self.mag_spec = mag_spec
        self.log_spec = log_spec

        if mag_spec:
            n_inputs = n_freq
        else:
            n_inputs = 2 * n_freq

        self.layers = nn.ModuleList(
            [
                GLULayer(n_inputs, n_bottleneck, n_sublayers=1, pool_size=pool_size),
                GLULayer(
                    n_bottleneck,
                    n_bottleneck,
                    n_sublayers=n_sublayers,
                    pool_size=pool_size,
                ),
                nn.Dropout(p=dropout_p),
                GLULayer(
                    n_bottleneck,
                    n_bottleneck,
                    n_sublayers=n_sublayers,
                    pool_size=pool_size,
                ),
                nn.ConvTranspose1d(
                    in_channels=n_bottleneck,
                    out_channels=n_freq,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
            ]
        )

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X = X.reshape((-1, n_freq, n_frames))
        X_pwr = mag_sq(X)

        # we want to normalize the scale of the input signal
        g = torch.clamp(torch.mean(X_pwr, dim=(-2, -1), keepdim=True), min=1e-5)

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        if self.mag_spec:
            X = divide(X_pwr, g)
        else:
            X = divide(X, torch.sqrt(g))
            X = torch.view_as_real(X)
            X = torch.cat((X[..., 0], X[..., 1]), dim=-2)

        # work with something less prone to explode
        if self.log_spec:
            X = torch.abs(X)
            weights = torch.log10(X + 1e-7)
        else:
            weights = X

        # apply all the layers
        for idx, layer in enumerate(self.layers):
            weights = layer(weights)

        # transform to weight by applying the sigmoid
        weights = torch.sigmoid(weights)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))

if __name__ == "__main__":
    input = torch.rand(1, 513, 1)
    model = GLUMask(513, 64)
    out = model(input)
    print(out.shape)
