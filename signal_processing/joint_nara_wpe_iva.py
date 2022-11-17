import torch
import soundfile as sf
from tqdm import tqdm

epsi = torch.finfo(torch.float64).eps
complex_type = torch.complex64
real_type = torch.float32


def inverse_2x2_matrix(mat):
    assert mat.shape[-1] == mat.shape[-2] and mat.shape[-2]==2
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]
    deno = a*d-b*c
    if(torch.any(deno==0)):
        raise Exception('mat can not be inversed')
    new_mat = torch.zeros_like(mat)
    new_mat[..., 0, 0] = d / deno
    new_mat[..., 0, 1] = -b / deno
    new_mat[..., 1, 0] = -c / deno
    new_mat[..., 1, 1] = a / deno
    return new_mat

def update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye):
    yyh1 = y1.unsqueeze(2) @ y1.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    V1 = alpha_iva * V1 + (1-alpha_iva) * yyh1/ vn1 # [513, 2, 2]
    w1 = torch.linalg.inv(W @ V1)[...,[0]]
    w1 = w1 / torch.sqrt(w1.conj().transpose(-1, -2) @ V1 @ w1)
                # W = torch.cat((w1, w2), dim=-1)
    yyh2 = y2.unsqueeze(2) @ y2.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    V2 = alpha_iva * V2 + (1-alpha_iva) * yyh2 / vn2
                # V2 = V2 + 1e-6*temp_eye
    w2 = torch.linalg.inv(W @ V2)[...,[1]]
    w2 = w2 / torch.sqrt(w2.conj().transpose(-1, -2) @ V2 @ w2)
    W[...,[0],:] = w1.conj().transpose(-1, -2)
    W[...,[1],:] = w2.conj().transpose(-1, -2)
    Wbp = (torch.linalg.pinv(W) * temp_eye) @ W # [513, 2, 2] * [513, 2, 2]
    return V1, V2, W, Wbp

def auxIVA_online(x, N_fft = 2048, hop_len = 0, label = None, ref_num=10):
    print(x.shape, x.dtype)
    K, N_y  = x.shape
    # parameter
    N_fft = N_fft
    N_move = hop_len
    N_effective = int(N_fft/2+1) #也就是fft后，频率的最高点
    window = torch.hann_window(N_fft, periodic = True, dtype=real_type)

    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha_iva = 0.96
    
    
    delay_num=1
    joint_wpe = True
    wpe_beta = 0.9999

    # initialization
    # r = torch.zeros((K, 1), dtype = torch.float64)
    
    G_wpe1 = torch.zeros((N_effective, ref_num*K, K), dtype = complex_type)#[513, 20, 2]
    G_wpe2 = torch.zeros((N_effective, ref_num*K, K), dtype = complex_type)#[513, 20, 2]
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    # wpe_sigma = torch.zeros(N_effective, K, K)
    # init
    W = temp_eye.clone()
    Wbp = temp_eye.clone()
    V1 = temp_eye.clone() * 1e-6
    V2 = temp_eye.clone() * 1e-6
    invR_WPE1 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    invR_WPE2 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    X_mix_stft = torch.stft(x, 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True).type(complex_type)
    if label is not None:
        label = torch.stft(label, 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True)
        label = label.permute(2, 1, 0) # [time, fre, C]
    C, N_fre, N_frame = X_mix_stft.shape
    X_mix_stft = X_mix_stft.permute(2, 1, 0).contiguous() # [C, fre, time] -> [time, fre, C]
    # aux_IVA_online
    for iter in range(1):
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)
        wpe_buffer = torch.cat([torch.zeros(ref_num+delay_num, N_effective, K, dtype=complex_type), X_mix_stft], dim=0)
        for i in tqdm(range(N_frame), ascii=True):
            if torch.prod(torch.prod(X_mix_stft[i, :, :]==0))==1:
                Y_all[i, :, :] = X_mix_stft[i, :, :]
            else:
                if joint_wpe and i >= ref_num + delay_num:
                    temp = wpe_buffer[i:i+ref_num].permute(1, 2, 0) # [10, 513, 2]- >[513, 2, 10]
                    x_D = torch.flatten(temp, -2, -1).unsqueeze(-1) #[513, 20, 1]
                    y1 = X_mix_stft[i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                    y2 = X_mix_stft[i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                else:
                    y1 = X_mix_stft[i]
                    y2 = X_mix_stft[i]
                vn1 = torch.mean(torch.abs(Wbp[..., [0], :] @ y1.unsqueeze(-1))**2)
                vn2 = torch.mean(torch.abs(Wbp[..., [1], :] @ y2.unsqueeze(-1))**2)
                # vn1 = torch.mean(torch.abs(label[i,...,0])**2)
                # vn2 = torch.mean(torch.abs(label[i,...,1])**2)
                if joint_wpe and i >= ref_num + delay_num:
                    # update
                    K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + x_D.conj().transpose(-1, -2) @ invR_WPE1 @ x_D) # [513, 20, 1]
                    invR_WPE1 = (invR_WPE1 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE1) / wpe_beta # [513, 20, 20]
                    G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()
                    K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + x_D.conj().transpose(-1, -2) @ invR_WPE2 @ x_D) # [513, 20, 1]
                    invR_WPE2 = (invR_WPE2 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE2) / wpe_beta # [513, 20, 20]
                    G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
                V1, V2, W, Wbp = update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye)
                pred_1 = Wbp[..., [0], :] @ y1.unsqueeze(-1)
                pred_2 = Wbp[..., [1], :] @ y2.unsqueeze(-1)
                Y_all[i] = torch.cat([pred_1, pred_2], dim=-2).squeeze(-1)

    Y_all = Y_all.permute(2, 1, 0).contiguous()
    y_iva = torch.istft(Y_all, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    return y_iva

if __name__ == "__main__":
    import time
    # reb = 1
    mix_path = 'audio\\2Mic_2Src_Mic.wav'
    out_path = 'audio\\nara_wpe_iva.wav'
    # clean_path = 'audio\\2Mic_2Src_Ref.wav'
    nfft = 1024
    # clean, sr = sf.read(clean_path)
    # clean = torch.from_numpy(clean.T)
    # load singal
    x , sr = sf.read(mix_path)
    # x = x[:5*16000]
    print(x.shape, x.dtype)
    x = torch.from_numpy(x.T)
    start_time = time.time()

    # ref_num = int(0.4*reb*sr-4*nfft) // nfft +1
    # print('refnum:', ref_num if ref_num>=1 else 1)
    y = auxIVA_online(x, N_fft = nfft, hop_len=nfft//4, label=None, ref_num=10)
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.T, sr)
    