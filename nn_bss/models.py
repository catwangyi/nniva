import torch
from torch import nn
import torch.nn.functional as functional
from loss import loss_func
from tqdm import tqdm
from utils import stft, istft
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def normlization(x):
    """ 
        Works for any input size = 4 D

        Args:
            input (:class:`torch.Tensor`): Shape `[batch, chan, dim1, dim2]
        eg  input:[batch, chan, fre, time]
            len(x.shape)=4
        Returns:
            value: 0 ~ 1
            :class:`torch.Tensor`: input `[batch, chan, fre, time]`

    """
    B, C, _, _ = x.shape
    x1 = x.reshape(B, C,-1)
    x_max, _ = torch.max(torch.abs(x1), 2)
    x_max = x_max.reshape(B, C, 1, 1)
    x_out = x / (x_max + 1e-8)
    return x_out

def global_norm(x):
    """ 
        Works for any input size = 4 D

        Args:
            input (:class:`torch.Tensor`): Shape `[batch, chan, dim1, dim2]
        eg  input:[batch, chan, fre, time]
            len(x.shape)=4
        Returns:
            :class:`torch.Tensor`: input `[batch, chan, fre, time]`

    """
    B, C, _, _ = x.shape
    x1 = x.reshape(B, C, -1)
    mean = x1.mean(dim=2).reshape(B, C, 1, 1)
    std = x1.std(dim=2, unbiased=False).view(B, C, 1, 1)
    x = (x - mean) / (std + 1e-6)
    # x_out = normlization(x)
    return x

def inverse_2x2_matrix(mat):
    assert mat.shape[-1] == mat.shape[-2] and mat.shape[-2]==2
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]
    num = a*d-b*c
    if(torch.any(torch.abs(num)<1e-9)):
        num = torch.abs(num).clamp_min(1e-9)
    new_mat = torch.zeros_like(mat)
    new_mat[..., 0, 0] = d / num
    new_mat[..., 0, 1] = -b / num
    new_mat[..., 1, 0] = -c / num
    new_mat[..., 1, 1] = a / num
    return new_mat
    
class gate_deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
        ) 
    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))

class gate_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
        ) 
    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))

class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is True.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=True):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1


        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                            bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
    

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        self.rnn.flatten_parameters() #解决RNN的权值不是单一连续的
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

class SepratinKernel(nn.Module):
    def __init__(self, input_dim=513, model_dim=256, lstm_layers=3, nspk=2, dropout=0):
        super(SepratinKernel, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim # model_dim 相当于隐层
        self.LSTM_hidden = model_dim*2
        self.lstm_layer_number = lstm_layers
        self.nspk = nspk
        self.lstm_layer = nn.ModuleList([])
        self.lstm_norm = nn.ModuleList([])

        for i in range(self.lstm_layer_number):
            self.lstm_layer.append(SingleRNN("LSTM", self.model_dim, self.model_dim, dropout=dropout, bidirectional=False))
            self.lstm_norm.append(nn.LayerNorm(self.model_dim))
        self.linear_in = nn.Sequential(nn.Linear(nspk*self.input_dim, self.model_dim),
                                        nn.ReLU())
        self.linear_out =  nn.Sequential(nn.Linear(self.model_dim, self.input_dim),
                                        nn.ReLU())
        self.linear_mask = nn.Linear(self.input_dim, self.input_dim * nspk)
        self.sm = nn.Sigmoid()
        #self.sm = nn.Tanh()


    def forward(self, inputs):
        mag = (torch.sqrt(inputs.real**2+inputs.imag**2+1e-8))
        phase = torch.atan2(inputs.imag, inputs.real+1e-8)
        assert inputs.dim() == 4
        #  Input: mixture_stft [batch, channel, frequency, Time] 
        B, C, fre, time = inputs.shape
        linear_in = (mag).reshape(B, C*fre, time) #  [B, C, fre, time] -> [B, C*fre, time]
        linear_in = linear_in.permute(0, 2, 1).contiguous() #  [B, C*fre, time] -> [B, time, C*fre]
        LSTM_in = self.linear_in(linear_in) #  [B, time, C*fre] -> [B, time, D]

        #  Separated Net
        for i in range(self.lstm_layer_number):
            LSTM_out = self.lstm_layer[i](LSTM_in) #  [B, time, D]
            LSTM_out = self.lstm_norm[i](LSTM_out) #  [B, time, D]
            LSTM_in = LSTM_in + LSTM_out #  [B, time, D]

        #  Make masks
        output = self.linear_out(LSTM_in) #  [B, time, D] -> [B, time, fre]
        mask = self.linear_mask(output) # [B, time, fre] -> [B, time, 2*fre]
        mask = self.sm(mask) *(1-1e-6) + 1e-6
        mask = mask.permute(0, 2, 1).reshape(B, C, fre, time) # [B, time, 2*fre] -> [B, 2*fre, time] - >[B, 2, F, T]
        vn1 = torch.mean(((mask * mag)[:, [0]])**2, dim=[-1, -2, -3], keepdim=True)
        vn2 = torch.mean(((mask * mag)[:, [1]])**2, dim=[-1, -2, -3], keepdim=True)
        return vn1, vn2
        # return torch.complex(mask * mag * torch.cos(phase), mask*mag*torch.sin(phase))

class crn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate1 = gate_conv(in_channels=2, out_channels=16, kernel_size=(3,1), stride=(2,1)) # 256
        self.gate2 = gate_conv(in_channels=16, out_channels=32, kernel_size=(3,1), stride=(2,1)) # 127
        self.gate3 = gate_conv(in_channels=32, out_channels=64, kernel_size=(3,1), stride=(2,1)) # 63
        self.gate4 = gate_deconv(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(2, 1))
        self.lstm = nn.LSTM(128, 256, 1, bidirectional=False)
        self.dnn = nn.Linear(256, 128)
        self.degate4 = gate_conv(in_channels=2*128, out_channels=64, kernel_size=(3, 1), stride=(2, 1))
        self.degate3 = gate_deconv(in_channels=2*64, out_channels=32, kernel_size=(3,1), stride=(2,1))
        self.degate2 = gate_deconv(in_channels=2*32, out_channels=16, kernel_size=(3,1), stride=(2,1))
        self.degate1 = gate_deconv(in_channels=2*16, out_channels=2, kernel_size=(3,1), stride=(2,1))
        
    def forward(self, x):
        # b, c, f, t
        mag = torch.sqrt(x.real**2+x.imag**2+1e-8)
        phase = torch.atan2(x.imag, x.real+1e-8)
        enc1 = self.gate1((mag)) # B, 16, 256, T
        enc2 = self.gate2(functional.pad(enc1, pad=[0, 0, 1, 0])) # B, 32, 128, T
        enc3 = self.gate3(functional.pad(enc2, pad=[0, 0, 1, 0])) # B, 64, 64, T
        enc4 = self.gate4(functional.pad(enc3, pad=[0, 0, 1, 0]))
        B, C, F, T = enc4.shape
        dec4 = self.dnn(self.lstm(enc4.permute(0, 3, 1, 2).reshape(F, B*T, C))[0]).reshape(F, B, T, C).permute(1, 3, 0, 2) # B, 64, 64, T
        dec3 = self.degate4(torch.cat([dec4, enc4], dim=1))[..., 1:,:]
        dec2 = self.degate3(torch.cat([dec3, enc3], dim=1))[..., 1:,:]
        dec1 = self.degate2(torch.cat([dec2, enc2], dim=1))[..., 1:,:]
        out = self.degate1(torch.cat([dec1, enc1], dim=1))
        out = torch.sigmoid(out)*(1-1e-6) + 1e-6
        return torch.complex(out * mag * torch.cos(phase), out*mag*torch.sin(phase))

class cal_vn_model(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.epsi = 1e-9
        self.nfft = 1024 if args is None else args.N_fft
        self.hoplen = self.nfft // 4
        # in_channels = self.nfft//2 +1
        self.lstm_list = nn.ModuleList([])
        self.lstm_norm = nn.ModuleList([])

        for i in range(3):
            self.lstm_list.append(nn.LSTM(256, 256))
            self.lstm_norm.append(nn.LayerNorm(256))
        self.linear_in = nn.Sequential(
            nn.Linear(2*513, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(256, 513),
            nn.LayerNorm(513),
            nn.ReLU(),
            nn.Linear(513, 513*2),
            nn.Sigmoid(),
        )

    def forward(self, inpt, hx):
        [B, C, F, T] = inpt.shape
        mag = torch.sqrt(inpt.real**2 + inpt.imag**2 + 1e-8)
        # phase = torch.atan2(inpt.imag, inpt.real+1e-8)
        lstm_in = (mag).permute(3, 0, 1, 2).reshape(T, B, C*F)
        # max = lstm_in.max()
        # min = lstm_in.min()
        # lstm_in = (lstm_in - min) / (max-min + 1e-8)
        lstm_in = self.linear_in((lstm_in))
        for i in range(len(self.lstm_list)):
            lstm_in, hx = self.lstm_list[i](lstm_in)
            lstm_in = self.lstm_norm[i](lstm_in)
        x = self.out(lstm_in)*(1-1e-6) + 1e-6 # -> [T, B, 2*513]
        x = x.permute(1, 2, 0).reshape(B, C, F, -1) #-> [B, C, F, T]
        vn1 = torch.mean((x[:, [0]]*mag)**2, dim=[-1, -2, -3], keepdim=True)
        vn2 = torch.mean((x[:, [1]]*mag)**2, dim=[-1, -2, -3], keepdim=True)
        return vn1, vn2, hx

class GRUBlock_frame(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, covFlag=False, bidirectional=False):
        super(GRUBlock_frame, self).__init__()
        self.covFlag = covFlag
        self.GRU = nn.GRUCell(in_channels, hidden_size)
        if self.covFlag == True:
            self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True))

    def forward(self, x, hx):
        # self.GRU.flatten_parameters()
        hx = self.GRU(x, hx) # [B, C_in] -> [B, D]
        if self.covFlag == 1:
            output = hx.unsqueeze(1) # [B, D] -> [B, 1, D]
            output = output.transpose(1,2) # [B, 1, D] -> [B, D, 1]
            output = self.conv(output) # [B, D, 1] -> [B, C_out, 1]
            output = torch.squeeze(output,2)
            # output = output.transpose(1,2) # [B, C_out, 1] -> [B, 1, C_out]
        else:
            output = hx
        return output, hx

class RNNBlock_frame(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, covFlag=False, bidirectional=False):
        super(RNNBlock_frame, self).__init__()
        self.covFlag = covFlag
        self.RNN = nn.RNNCell(in_channels, hidden_size)
        if self.covFlag is True:
            self.conv = nn.Sequential(nn.Conv1d(hidden_size * (2 if bidirectional==True else 1), out_channels, kernel_size = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace = True))

    def forward(self, x, hx):
        # self.GRU.flatten_parameters()
        hx = self.RNN(x, hx)  # [B, C_in] -> [B, D]
        if self.covFlag == 1:
            output = hx.unsqueeze(1)  # [B, D] -> [B, 1, D]
            output = output.transpose(1, 2)  # [B, 1, D] -> [B, D, 1]
            output = self.conv(output)  # [B, D, 1] -> [B, C_out, 1]
            output = torch.squeeze(output, 2)
            # output = output.transpose(1,2) # [B, C_out, 1] -> [B, 1, C_out]
        else:
            output = hx
        return output, hx

class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=True):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 2))
        if squeeze:
            x = torch.squeeze(x,-1)
        return x

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        xdim = x.dim()
        if xdim != 3:
            if xdim == 2:  
                x = torch.unsqueeze(x, 2)
            else:
                raise RuntimeError("{} accept 3D tensor as input".format(
                    self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        if xdim == 2:
            x = torch.squeeze(x, 2)
        return x

class nlmsParamsModel(nn.Module):
    def __init__(self, input_dim=513, hidden_size=256, rnn_method=1):
        super(nlmsParamsModel, self).__init__()
        self.rnnMethod = rnn_method  # 0 for RNNcell, 1 for GRUcell
        self.infeatureSize = 2*input_dim
        self.nb_neuron = hidden_size  # 257
        self.hidden_neuron = hidden_size
        self.out_neuron = hidden_size
        self.out_dim = input_dim
        self.cov1d_flag = 0
        if self.cov1d_flag:
            self.FC = nn.Sequential(Conv1D(self.infeatureSize, self.nb_neuron, 1), nn.LayerNorm(self.nb_neuron), nn.ReLU())
        else:
            self.FC = nn.Sequential(nn.Linear(self.infeatureSize, self.nb_neuron), nn.LayerNorm(self.nb_neuron), nn.ReLU()) # nn.Linear(input_dim, self.nb_neuron)

        self.layernorm = ChannelWiseLayerNorm(self.out_neuron)
        if self.rnnMethod == 1:
            self.RNN = GRUBlock_frame(self.nb_neuron, self.hidden_neuron, self.out_neuron, covFlag=False)
        else:
            self.RNN = RNNBlock_frame(self.nb_neuron, self.hidden_neuron, self.out_neuron, covFlag=False)

        self.v1Dense = nn.Sequential(nn.Linear(self.out_neuron, self.out_dim), nn.LayerNorm(self.out_dim), nn.Sigmoid())
        self.v2Dense = nn.Sequential(nn.Linear(self.out_neuron, self.out_dim), nn.LayerNorm(self.out_dim), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1/input_dim)

    def forward(self, inpt, hx): 
        B, _, F, _ = inpt.shape
        x = torch.abs(normlization(inpt)).reshape(B, -1) # B, 2*F
        l1Out = self.FC(x) # B, 2*F
        l2Out, hx = self.RNN(l1Out, hx) # B, 2*F
        l2Out = self.layernorm(l2Out) # B, 2*F
        maskv1 = self.v1Dense(l2Out+l1Out)*(1-1e-6)+1e-6 # B, F
        maskv2 = self.v2Dense(l2Out+l1Out)*(1-1e-6)+1e-6 # B, F
        vn1 = torch.mean((maskv1 * torch.abs(inpt[:, 0].squeeze()))**2, dim=[-1], keepdim=True).reshape(B, 1, 1, -1)
        vn2 = torch.mean((maskv2 * torch.abs(inpt[:, 1].squeeze()))**2, dim=[-1], keepdim=True).reshape(B, 1, 1, -1)
        return vn1, vn2, hx
        # vn1 = (maskv1.unsqueeze(-1) * torch.abs(inpt[:, 0]))**2
        # vn2 = (maskv2.unsqueeze(-1) * torch.abs(inpt[:, 1]))**2
        # return vn1.unsqueeze(-1), vn2.unsqueeze(-1), hx

class my_model(nn.Module):
    def __init__(self, ppflag, cal_vn, joint_wpe, args=None, ):
        super().__init__()
        self.epsi = 1e-6
        self.cal_vn = cal_vn
        self.joint_wpe = joint_wpe
        self.nfft = 1024 if args is None else args.N_fft
        self.hoplen = self.nfft // 4
        self.vn_model = nlmsParamsModel()
        self.ppflag = ppflag
        if ppflag:
            self.pp_model = SepratinKernel()
    
    # def update_wpe_dnn(self, x_D, invR_WPE1, invR_WPE2, vn1, vn2, y1, y2, G_wpe1, G_wpe2, wpe_beta):
    #     eps = torch.tensor(1e-8, device=x_D.device, dtype=x_D.real.dtype)
    #     temp = x_D.conj().transpose(-1, -2) # [513, 1, 20]
    #     deno = (wpe_beta * vn1 + temp @ invR_WPE1 @ x_D)
    #     new_deno_real = torch.where(torch.abs(deno.real) < eps, eps, deno.real)
    #     new_deno_imag = torch.where(torch.abs(deno.imag) < eps, eps, deno.imag)
    #     K_wpe = (invR_WPE1 @ x_D) / torch.complex(new_deno_real, new_deno_imag) # [513, 20, 1]
    #     new_invR_WPE1 = (invR_WPE1 -K_wpe @ temp @ invR_WPE1) / wpe_beta # [513, 20, 20]
    #     new_G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()
    #     deno = (wpe_beta * vn2 + temp @ invR_WPE2 @ x_D)
    #     new_deno_real = torch.where(torch.abs(deno.real) < eps, eps, deno.real)
    #     new_deno_imag = torch.where(torch.abs(deno.imag) < eps, eps, deno.imag)
    #     K_wpe = (invR_WPE2 @ x_D) / torch.complex(new_deno_real, new_deno_imag) # [513, 20, 1]
    #     new_invR_WPE2 = (invR_WPE2 -K_wpe @ temp @ invR_WPE2) / wpe_beta # [513, 20, 20]
    #     new_G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
    #     return new_invR_WPE1, new_invR_WPE2, new_G_wpe1, new_G_wpe2

    def update_auxiva_dnn(self, V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye):
        eps = torch.tensor(1e-8, device=y1.device, dtype=y1.real.dtype)
        yyh1 = y1.unsqueeze(-1) @ y1.unsqueeze(-2).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
        V1 = alpha_iva * V1 + (1-alpha_iva) * yyh1 / (vn1 + eps)# [513, 2, 2]
        tempv1 = V1 + 1e-6 * temp_eye *(V1[..., 0, 0] + V1[..., 1, 1]).reshape(V1.shape[0], V1.shape[1], 1, 1)  # [513, 2, 2]
        w1 = inverse_2x2_matrix(W @ tempv1) @ temp_eye[..., 0].unsqueeze(-1)
        # w1_max, _ = torch.max(torch.abs(w1), dim=-2, keepdim=True)
        # w1 = w1 / w1_max.clamp_min(eps)
        deno = w1.conj().transpose(-1, -2) @ V1 @ w1
        # deno_real = torch.where(torch.abs(deno.real)<eps, eps, deno.real)
        # deno_imag = torch.where(torch.abs(deno.imag)<eps, eps, deno.imag)
        # new_w1 = w1 / torch.sqrt(torch.complex(deno_real, deno_imag))
        new_w1 = w1 / torch.sqrt(deno + eps)
        new_w = torch.cat([new_w1.conj().transpose(-1, -2), W[..., [1], :]], dim=-2)
        yyh2 = y2.unsqueeze(-1) @ y2.unsqueeze(-2).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
        V2 = alpha_iva * V2 + (1-alpha_iva) * yyh2 / (vn2 + eps)
        tempv2 = V2 + 1e-6 * temp_eye * (V2[..., 0, 0] + V2[..., 1, 1]).reshape(V2.shape[0], V2.shape[1], 1, 1)
        w2 = inverse_2x2_matrix(new_w @ tempv2) @ temp_eye[..., 1].unsqueeze(-1) 
        # w2_max, _ = torch.max(torch.abs(w2), dim=-2, keepdim=True)
        # w2 = w2 / w2_max.clamp_min(eps)
        deno = w2.conj().transpose(-1, -2) @ V2 @ w2
        # deno_real = torch.where(torch.abs(deno.real)<eps, eps, deno.real)
        # deno_imag = torch.where(torch.abs(deno.imag)<eps, eps, deno.imag)
        # new_w2 = w2 / torch.sqrt(torch.complex(deno_real, deno_imag))
        new_w2 = w2 / torch.sqrt(deno + eps)
        new_w = torch.cat([new_w[...,[0], :], new_w2.conj().transpose(-1, -2)], dim=-2)
        # new_W = torch.cat([new_w1.conj().transpose(-1, -2), new_w2.conj().transpose(-1, -2)], dim=-2)
        Wbp = (inverse_2x2_matrix(new_w) * temp_eye) @ new_w # [513, 2, 2] * [513, 2, 2]
        return V1, V2, new_w, Wbp

    def update_wpe(self, x_D, invR_WPE1, invR_WPE2, vn1, vn2, y1, y2, G_wpe1, G_wpe2, wpe_beta):
        temp = x_D.conj().transpose(-1, -2) # [513, 1, 20]
        K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + temp @ invR_WPE1 @ x_D) # [513, 20, 1]
        invR_WPE1 = (invR_WPE1 -K_wpe @ (temp @ invR_WPE1)) / wpe_beta # [513, 20, 20]
        G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()

        K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + temp @ invR_WPE2 @ x_D) # [513, 20, 1]
        invR_WPE2 = (invR_WPE2 -K_wpe @ (temp @ invR_WPE2)) / wpe_beta # [513, 20, 20]
        G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
        return invR_WPE1, invR_WPE2, G_wpe1, G_wpe2

    def update_auxiva(self, V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye):
        V1 = alpha_iva * V1 + (1-alpha_iva) * y1.unsqueeze(-1) / vn1 @ y1.unsqueeze(-2).conj()# [513, 2, 2]
        temp_V1 = V1 + 1e-6*temp_eye*(V1[..., 0, 0] + V1[..., 1, 1]).reshape(V1.shape[0], V1.shape[1], 1, 1) 
        w1 = torch.linalg.inv(W @ temp_V1) @ temp_eye[..., 0].unsqueeze(-1)
        new_w1 = w1 / torch.sqrt(w1.conj().transpose(-1, -2) @ V1 @ w1)
        W[..., [0], :] = new_w1.conj().transpose(-1, -2)
        V2 = alpha_iva * V2 + (1-alpha_iva) * y2.unsqueeze(-1) / vn2 @ y2.unsqueeze(-2).conj() 
        temp_V2 = V2 + 1e-6*temp_eye* (V2[..., 0, 0] + V2[..., 1, 1]).reshape(V2.shape[0], V2.shape[1], 1, 1)
        w2 = torch.linalg.inv(W @ temp_V2) @ temp_eye[..., 1].unsqueeze(-1) # [513, 2, 1]
        new_w2 = w2 / torch.sqrt(w2.conj().transpose(-1, -2) @ V2 @ w2) #[513, 2, 1]
        W[..., [1], :] = new_w2.conj().transpose(-1, -2)
        Wbp = (torch.linalg.inv(W) * temp_eye) @ W # [513, 2, 2] * [513, 2, 2]
        return V1, V2, W, Wbp

    def forward(self, mix_audio, ref_num=5, delay_num=1):
        cal_vn = self.cal_vn
        X_mix_stft = stft(mix_audio, 1024, 256)
        complex_type = X_mix_stft.dtype
        device = X_mix_stft.device
        B, K, N_effective, N_frame = X_mix_stft.shape
        alpha_iva = 0.96
        joint_wpe = self.joint_wpe
        wpe_beta = 0.9995
    
        G_wpe1 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
        G_wpe2 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
        temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
        temp_eye = torch.repeat_interleave(temp_eye.unsqueeze(0), B, dim=0)
        # init
        W = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
        W = torch.repeat_interleave(W.unsqueeze(0), B, dim=0)
        Wbp = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
        Wbp = torch.repeat_interleave(Wbp.unsqueeze(0), B, dim=0)
       
        hid = torch.zeros(B, 256, device=device)
        V1 = torch.zeros((B, N_effective, K, K), dtype = complex_type, device=device)#[513, 2, 2]
        V2  = torch.zeros((B, N_effective, K, K), dtype = complex_type, device=device)#[513, 2, 2]
        invR_WPE1 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
        invR_WPE1 = torch.repeat_interleave(invR_WPE1.unsqueeze(0), B, dim=0)
        invR_WPE2 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
        invR_WPE2 = torch.repeat_interleave(invR_WPE2.unsqueeze(0), B, dim=0)
        X_mix_stft = X_mix_stft.permute(0, 3, 2, 1).contiguous() # [B, T, F, C]
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)

        wpe_buffer = (X_mix_stft).permute(0, 2, 3, 1).contiguous()
        Y_all[:, :ref_num+delay_num] = X_mix_stft[:, :ref_num+delay_num]

        for i in range(ref_num+delay_num, N_frame):
            if torch.prod(torch.prod(X_mix_stft[:, i]==0))==1:
                Y_all[:, i] = X_mix_stft[:, i]
            else:
                if joint_wpe:
                    temp = wpe_buffer[..., i-delay_num-ref_num:i-delay_num] # [10, 513, 2]- >[513, 2, 10]
                    x_D = torch.flatten(temp, -2, -1).unsqueeze(-1).contiguous() #[513, 20, 1]
                    y1 = X_mix_stft[:, i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                    y2 = X_mix_stft[:, i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                else:
                    y1 = X_mix_stft[:, i]
                    y2 = X_mix_stft[:, i]
                if cal_vn:
                    iva_out = torch.cat([Wbp[..., [0], :] @ y1.unsqueeze(-1), Wbp[..., [1], :] @ y2.unsqueeze(-1)], dim=-2).permute(0, 2, 1, 3)
                    vn1, vn2, hid = self.vn_model(iva_out, hid)
                    # iva_vn1 = torch.mean(torch.abs(Wbp[..., [0], :] @ y1.unsqueeze(-1))**2, dim=[-1, -2, -3], keepdim=True)
                    # iva_vn2 = torch.mean(torch.abs(Wbp[..., [1], :] @ y2.unsqueeze(-1))**2, dim=[-1, -2, -3], keepdim=True)
                else:
                    vn1 = torch.mean(torch.abs(Wbp[..., [0], :] @ y1.unsqueeze(-1))**2, dim=[-1, -2, -3], keepdim=True)
                    vn2 = torch.mean(torch.abs(Wbp[..., [1], :] @ y2.unsqueeze(-1))**2, dim=[-1, -2, -3], keepdim=True)
                if joint_wpe:
                    invR_WPE1, invR_WPE2, G_wpe1, G_wpe2 = self.update_wpe(x_D, invR_WPE1, invR_WPE2, vn1, vn2, y1, y2, G_wpe1, G_wpe2, wpe_beta)
                    y1 = X_mix_stft[:, i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                    y2 = X_mix_stft[:, i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                V1, V2, W, Wbp = self.update_auxiva_dnn(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye)
                iva_out = torch.cat([Wbp[..., [0], :] @ y1.unsqueeze(-1), Wbp[..., [1], :] @ y2.unsqueeze(-1)], dim=-2)
                if torch.any(torch.isnan(iva_out)) == True:
                    message = None
                    for name, param in self.vn_model.named_parameters():
                        if torch.any(torch.isnan(param.data)):
                            message = 'model is nan!'
                            break
                    raise Exception(f'输出出现了nan, frame_num:{i} {"model is not nan" if message is None else message}')
                Y_all[:,i] = iva_out.squeeze(-1)
        y_stft = Y_all.permute(0, 3, 2, 1)
        if self.ppflag:
            pred_stft = self.pp_model(y_stft)
        else:
            pred_stft = y_stft
        pred_wav = istft(pred_stft, 1024, 256)
        return pred_stft, pred_wav

    def auxIVA_online(self, mix_audio, ref_num=5, delay_num=1):
        X_mix_stft = stft(mix_audio, 1024, 256)
        complex_type = X_mix_stft.dtype
        device = X_mix_stft.device
        B, K, N_effective, N_frame = X_mix_stft.shape
        alpha_iva = 0.96
    
        joint_wpe = True
        wpe_beta = 0.9995
    
        G_wpe1 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
        G_wpe2 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
        temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
        temp_eye = torch.repeat_interleave(temp_eye.unsqueeze(0), B, dim=0)
        # init
        W = temp_eye.clone()
        Wbp = temp_eye.clone()
        V1 = torch.zeros((B, N_effective, K, K), dtype = complex_type, device=device)
        V2 = torch.zeros((B, N_effective, K, K), dtype = complex_type, device=device)
        # V1 = temp_eye.clone() 
        # V2 = temp_eye.clone() 
        invR_WPE1 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
        invR_WPE1 = torch.repeat_interleave(invR_WPE1.unsqueeze(0), B, dim=0)
        invR_WPE2 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
        invR_WPE2 = torch.repeat_interleave(invR_WPE2.unsqueeze(0), B, dim=0)
        # aux_IVA_online
        # init paras and buffers
        wpe_buffer = (X_mix_stft).permute(0, 2, 1, 3)# [B, F, C, T]
        X_mix_stft = X_mix_stft.permute(0, 3, 2, 1) # [B, T, F, C]
        Y_all = torch.zeros_like(X_mix_stft)
        Y_all[:, :ref_num+delay_num] = X_mix_stft[:, :ref_num+delay_num]
        vn1_plt = []
        vn2_plt = []
        for i in range(ref_num+delay_num, N_frame):
            if torch.prod(torch.prod(X_mix_stft[:, i]==0))==1:
                Y_all[:, i] = X_mix_stft[:, i]
            else:
                if joint_wpe:
                    temp = wpe_buffer[..., i-ref_num-delay_num:i-delay_num] # [513, 2, ref_num]
                    x_D = torch.cat([temp[...,0,:], temp[...,1,:]], dim=-1).unsqueeze(-1) # [513, 2*ref_num] [513, 10]
                    y1 = X_mix_stft[:, i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                    y2 = X_mix_stft[:, i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                else:
                    y1 = X_mix_stft[:, i]
                    y2 = X_mix_stft[:, i]
                iva_out = torch.cat([Wbp[..., [0], :] @ y1.unsqueeze(-1), Wbp[..., [1], :] @ y2.unsqueeze(-1)], dim=-2).squeeze(-1)# [B, 513, 2]
                # Y_all[:,i] = iva_out
                vn1 = torch.mean(torch.abs(iva_out[..., 0])**2, dim=[-1], keepdim=True).unsqueeze(-1).unsqueeze(-1)
                vn2 = torch.mean(torch.abs(iva_out[..., 1])**2, dim=[-1], keepdim=True).unsqueeze(-1).unsqueeze(-1)
                vn1_plt.append(vn1.squeeze().detach().numpy())
                vn2_plt.append(vn2.squeeze().detach().numpy())
                if joint_wpe:
                    # pass
                    invR_WPE1, invR_WPE2, G_wpe1, G_wpe2 = self.update_wpe(x_D, invR_WPE1, invR_WPE2, vn1, vn2, y1, y2, G_wpe1, G_wpe2, wpe_beta)
                    y1 = X_mix_stft[:, i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                    y2 = X_mix_stft[:, i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 2*5] *[513, 2*5, 1]
                V1, V2, W, Wbp = self.update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye)
                iva_out = torch.cat([Wbp[..., [0], :] @ y1.unsqueeze(-1), Wbp[..., [1], :] @ y2.unsqueeze(-1)], dim=-2).squeeze(-1)# [B, F, C]
                assert torch.any(torch.isnan(iva_out)) == False , f'iva_out is nan'
                Y_all[:,i] = iva_out
               
        y_stft = Y_all.permute(0, 3, 2, 1)
        # y_stft = y_stft*std+mean
        y_wav = istft(y_stft, 1024, 256)
        plt.subplot(2, 1, 1)
        plt.plot(vn1_plt)
        plt.subplot(2, 1, 2)
        plt.plot(vn2_plt)
        plt.show()
        plt.savefig('pytorch版本的vn')
        return y_stft, y_wav

if __name__ =='__main__':
    import torchaudio
    import soundfile as sf
    import time
    # reb = 1
    mix_path = 'mix.wav'
    label_path = 'label.wav'
    
    # load singal
    x , sr = torchaudio.load(mix_path)
    label, _ = torchaudio.load(label_path)
    model = my_model(ppflag=False, cal_vn=True)
    # model = torch.load('model_epoch_50').to('cpu')

    start_time = time.time()
    _, wav = model(x.unsqueeze(0))
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write('nara_wpe_iva_dnn_vn.wav', wav[0].detach().cpu().T, sr)

    start_time = time.time()
    iva_stft, wav = model.auxIVA_online(x.unsqueeze(0))
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write('nara_wpe_iva.wav', wav[0].detach().cpu().T, sr)