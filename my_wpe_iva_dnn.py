from logging import exception
import os
from tkinter import Y
from numpy import transpose
import torch
import soundfile as sf
from tqdm import tqdm

epsi = torch.finfo(torch.float64).eps
complex_type = torch.complex128
real_type = torch.float64
device = torch.device('cpu')
from models import my_model
from loss import cal_p_loss, functional


def init(X, alpha, xxh, temp_eye, U, V, Y=None):
    N_effective = max(X.shape)
    K = min(X.squeeze().shape)
    W = torch.repeat_interleave(torch.eye(K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    A = torch.repeat_interleave(torch.eye(K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    p = create_p(W, X, alpha, Y)
    for k in range(K):
        V[:, :, :, k] = (xxh *p[k]) * temp_eye
        U[:, :, :, k] = 1 / (V[:, :, :, k] + epsi) * temp_eye
    return A, W, U, V

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


def update_a_w(A, W, U, V):
    # W [513, 2, 2]
    # return A, W
    K_num = W.shape[-1]
    new_w = []
    for k in range(K_num):
        U_temp = U[:, :, :, k] # [513, 2, 2] # U = V^-1
        A_temp = A[:, :, k].unsqueeze(2) # [513, 2] -> [513, 2, 1]
        temp_inv = torch.matmul(U_temp, A_temp) # temp_inv = V^-1@W^-1 = (W@V)^-1, 就是指wk的意思
        # update W
        W_k = temp_inv.conj().permute(0, 2, 1) # [513, 2, 1] -> [513, 1, 2]
        temp_W_k = torch.sqrt(temp_inv.conj().permute(0, 2, 1) @ V[:, :, :, k] @ temp_inv) # [513, 1, 2] * [513, 2, 2] * [513, 2, 1]
        W_k = W_k / temp_W_k # [513, 1, 2]
        dw = W_k - W[:, k, :].unsqueeze(1)# [513, 1 ,2]
        new_w.append(W_k.squeeze(1))
        Anumer = A[:, :, k].unsqueeze(2) @ dw @ A # [513, 2, 1] * [513, 1, 2] * [513, 2, 2]
        Adenom_new = 1  + dw @ A[:, :, k].unsqueeze(2) # [513, 1, 2] * [513, 2, 1]
        Adenom_new = Adenom_new.real
        epsi_mat = torch.ones_like(Adenom_new) * epsi
        Adenom_new = torch.maximum(Adenom_new, epsi_mat)
        # 先尝试不更新A
        A = A - Anumer / Adenom_new # [513, 2, 2]
    new_w = torch.stack(new_w, dim=1)
    return A, new_w

def update_v(V, alpha, p, xxh):
    # return V
    K_num = V.shape[1]
    temp = []
    for k in range(K_num):
        temp.append(alpha * V[:, :, :, k] + p[k] * xxh) # phi就是x*x^H
        # V[..., k] = alpha * V[:, :, :, k] + p[k] * xxh
    # return V
    new_V = torch.stack(temp, dim=-1)
    return new_V

def update_u(W, xxh, p, alpha, U, X):
    # return U
    N_effective, K_num = W.shape[0], W.shape[-1]
    temp_u = []
    for k in range(K_num):
        Unumer = p[k] *  U[:, :, :, k] @ xxh @  U[:, :, :, k] # x * [513, 2, 2] * [513, 2, 2] * [513, 2, 2]
        # Udenom_temp_X1 = X.conj().unsqueeze(1) # [513, 2] -> [513, 1, 2]
        # Udenom_temp_X2 = X.unsqueeze(2) # [513, 2] -> [513, 2, 1]
        # Udenom_temp_U =  U[:, :, :, k] # [2, 2, 513] -> [513, 2, 2]
        Udenom = alpha**2 + alpha * p[k] * X.conj().unsqueeze(1) @ U[:, :, :, k] @ X.unsqueeze(2)
        epsi_mat = torch.ones((N_effective, 1, 1), dtype = torch.float64, device=device) * epsi
        # Udenom = Udenom.real
        Udenom = torch.maximum(Udenom.real, epsi_mat)
        # U_temp =  U[:, :, :, k] / alpha - Unumer / Udenom
        temp_u.append(U[:, :, :, k] / alpha - Unumer / Udenom) # [513, 2, 2]
    return torch.stack(temp_u, dim=-1)

def create_p(W, X, alpha, Y=None):
    p = torch.zeros((W.shape[1], 1), dtype = real_type)
    K_num = W.shape[-1]
    for k in range(K_num): 
        # r_hat_W1 = W[:, k, :].unsqueeze(1) # [513, 2] -> [513, 1, 2]
        # r_hat_phi = torch.matmul(phi_temp1, phi_temp2) # [513, 2, 2]
        # r_hat_W2 = W[:, k, :].conj().unsqueeze(2) # [513, 2] ->[513, 2, 1]
        if Y is None:
            temp_sum = torch.sum(W[:, k, :].unsqueeze(1) @ X.unsqueeze(2) @ X.unsqueeze(1).conj() @ W[:, k, :].conj().unsqueeze(2))
        else:
            temp_sum = torch.sum(Y[...,k] * Y[...,k].conj())
        # a = r_hat_W1 @ X.unsqueeze(2) @ X.unsqueeze(1).conj() @ r_hat_W2
        deo = torch.sqrt(temp_sum.real)
        if deo < epsi:
            deo = epsi
        p[k] = (1-alpha) / deo # p就是1/r
    return p

def update(V, alpha, xxh, X, W, U, A, Y):
    p = create_p(W, X, alpha, Y)
    V = update_v(V, alpha, p, xxh)
    U = update_u(W, xxh, p, alpha, U, X)
    A, W = update_a_w(A, W, U, V)
    return A, W, U, V

def auxIVA_online(x, N_fft = 2048, hop_len = 0, label = None):
    print(x.shape, x.dtype)
    K, N_y  = x.shape
    # parameter
    N_fft = N_fft
    N_move = hop_len
    N_effective = int(N_fft/2+1) #也就是fft后，频率的最高点
    window = torch.hann_window(N_fft, periodic = True, dtype=real_type)

    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha_iva = 0.96
    
    initial = 0
    ref_num=15
    delay_num=2
    gamma_wpe = 0.995
    wpe_beta = 0.5
    model = my_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # initialization
    Y_all = []
    # r = torch.zeros((K, 1), dtype = torch.float64)
    
    V = torch.zeros((N_effective, K, K, K), dtype = complex_type)
    U = torch.zeros((N_effective, K, K, K), dtype = complex_type)
    W = torch.zeros((N_effective, K, K), dtype = complex_type)
    A = torch.zeros((N_effective, K, K), dtype = complex_type)
    G_wpe = torch.zeros((N_effective, ref_num*(K), 1), dtype = complex_type)
    K_wpe = torch.zeros((N_effective, ref_num*(K), K), dtype = complex_type)
    # phi = torch.zeros((N_effective, K, K), dtype = complex_type)
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    wpe_sigma = torch.zeros(N_effective, K, K)
    # init
    W = temp_eye.clone()
    A = temp_eye.clone()
    Wbp = temp_eye.clone()
    invQ_WPE = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]

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
        label = label.permute(2, 1, 0)
    C, N_fre, N_frame = X_mix_stft.shape
    X_mix_stft = X_mix_stft.permute(2, 1, 0).contiguous() # [C, fre, time] -> [time, fre, C]
    X_D = torch.zeros(N_effective, K, K*ref_num, dtype=torch.complex128)
    # aux_IVA_online
    for iter in range(1):
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)
        y_wpe = torch.zeros_like(X_mix_stft)
        # Y_all[0:ref_num+delay_num, ...] = X_mix_stft[0:ref_num+delay_num, ...]
        y_wpe[0:ref_num+delay_num, ...] = X_mix_stft[0:ref_num+delay_num, ...]
        # wpe_buffer = X_mix_stft[0:ref_num, :, :]

        bar = tqdm(range(N_frame), ascii=True)
        for i in bar:
            optimizer.zero_grad()
            if i >=delay_num+ref_num:
                wpe_buffer = X_mix_stft[i-delay_num-ref_num:i-delay_num].permute(1, 2, 0) # [513, 2, 10]
                # X_D = torch.kron(torch.eye(K).unsqueeze(0), wpe_buffer.permute(1, 2, 0).contiguous()) #[1, 2, 2] * [513, 2, ref_num] -> [513, K**2, K*ref_num]
                # X_D = X_D.reshape(N_effective, K, ref_num*(K**2)) # [513, 2, 2^2*ref_num]
                X_D[...,0,:ref_num] = wpe_buffer[...,0,:]
                X_D[...,1,-ref_num:] = wpe_buffer[...,1,:]
                y_wpe[i, :, :] = X_mix_stft[i, ...] -  (X_D @ G_wpe).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                # temp_diag = torch.zeros_like(W)
                # temp_diag[..., 0, 0] = Y_all[i,..., 0]
                # temp_diag[..., 1, 1] = Y_all[i,..., 1]
                # a = torch.diag_embed(Y_all[i, ...])
                sig = inverse_2x2_matrix(Wbp) @ torch.diag_embed((Wbp @ y_wpe[i,...].unsqueeze(-1)).squeeze(-1))
                wpe_sigma = (1-wpe_beta) * wpe_sigma + wpe_beta * sig @ sig.conj().transpose(-1, -2) # [513, 2, 2]
                nominator = invQ_WPE @ X_D.conj().transpose(-1, -2) # [513, 40, 40] * [513, 40, 2]-> [513, 40, 2]
                K_wpe = nominator @ inverse_2x2_matrix(gamma_wpe * wpe_sigma + X_D @ nominator) # [513, 40, 2]
                invQ_WPE = (invQ_WPE - K_wpe @ X_D @ invQ_WPE) / gamma_wpe
                # G_wpe = G_wpe
                G_wpe = G_wpe + K_wpe @ y_wpe[i, ...].unsqueeze(-1)
                y_wpe[i, :, :] = X_mix_stft[i, ...] -  (X_D @ G_wpe).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
            else:
                y_wpe[i, :, :] = X_mix_stft[i, ...]
            if torch.prod(torch.prod(y_wpe[i, :, :]==0))==1:
                Y_all[i, :, :] = y_wpe[i, :, :]
            else:
                X = y_wpe[i, :, :] # [time, fre, C] -> [fre, C]
                phi_temp1 = X.unsqueeze(2)  # [513, 2] -> [513, 2, 1]
                phi_temp2 = X.unsqueeze(1).conj() # [513, 2] -> [513, 1, 2]
                xxh = torch.matmul(phi_temp1, phi_temp2) # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                
                if initial == 0:
                    A, W, U, V = init(X, alpha_iva, xxh, temp_eye, U, V, label[i] if label is not None else None)
                    initial = 1
                    Wbp =  A * temp_eye  @ W # [513, 2, 2] * [513, 2, 2]
                    Y_temp = Wbp @ X.unsqueeze(2) # [513, 2, 2] * [513, 2, 1] -> [513, 2, 1]
                    Y_all[[i], ...] = (Y_temp.permute(2, 0, 1)) #[2, 513, 1]
                    continue
                else:
                    real_p = create_p(W, label[i], alpha_iva, label[i] if label is not None else None)
                    a = torch.abs(X) / torch.norm(torch.abs(X), p=2, dim=0, keepdim=True) # [513, 2]
                    p = model(a.T.unsqueeze(0)).squeeze(0)
                    loss = cal_p_loss(p, real_p)
                    bar.set_postfix({'loss': f'{loss.item():.5e}'})
                    
                    new_V = update_v(V, alpha_iva, p.clone(), xxh)
                    new_U = update_u(W, xxh, p, alpha_iva, U, X)
                    new_A, new_W = update_a_w(A, W, U, V)
                     # calculate output
                    Wbp = A * temp_eye @ W # [513, 2, 2] * [513, 2, 2]
                    Y_temp = (Wbp @ X.unsqueeze(2)).squeeze(-1) # [513, 2, 2] * [513, 2, 1] -> [513, 2]
                    Y_all[i, ...] = Y_temp
                    loss.backward()
                    optimizer.step()
                    A = new_A.detach().clone()
                    W = new_W.detach().clone()
                    U = new_U.detach().clone()
                    V = new_V.detach().clone()

    Y_all = Y_all.permute(2, 1, 0).contiguous()
    y_wpe = y_wpe.permute(2, 1, 0).contiguous()
    # print(Y_all.shape)
    y_wpe = torch.istft(y_wpe, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    y_iva = torch.istft(Y_all, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    # print(y_iva.shape)
    return y_iva, y_wpe

if __name__ == "__main__":
    import time
    mix_path = r'audio\2Mic_2Src_Mic.wav'
    out_path = r'audio\\wpe_iva_dnn.wav'
    clean_path = r'audio\2Mic_2Src_Ref.wav'
    clean, sr = sf.read(clean_path)
    clean = torch.from_numpy(clean.T)
    # load singal
    x , sr = sf.read(mix_path)
    # x = x[:5*16000]
    # clean = clean[:5*16000]
    x = x[:clean.shape[-1]]
    print(x.shape, x.dtype)
    x = torch.from_numpy(x.T)
    start_time = time.time()
    with torch.autograd.set_detect_anomaly(True):
        y, y_wpe = auxIVA_online(x, N_fft = 1024, hop_len=256, label=clean)
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.T, sr)
    sf.write('audio\\wpe_iva_dnn_wpe_out.wav', y_wpe.T, sr)
