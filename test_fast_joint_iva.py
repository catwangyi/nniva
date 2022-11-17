import torch
import soundfile as sf
from tqdm import tqdm

epsi = torch.finfo(torch.float64).eps
complex_type = torch.complex128
real_type = torch.float64

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
    K_num = W.shape[-1]
    for k in range(K_num):
        U_temp = U[:, :, :, k] # [513, 2, 2] # U = V^-1
        A_temp = A[:, :, k].unsqueeze(2) # [513, 2] -> [513, 2, 1]
        temp_inv = torch.matmul(U_temp, A_temp) # temp_inv = V^-1@W^-1 = (W@V)^-1, 就是指wk的意思
        # update W
        W_k = temp_inv.conj().permute(0, 2, 1) # [513, 2, 1] -> [513, 1, 2]
        # V_temp = V[:, :, :, k] # [513, 2, 2]
        temp_W_k = torch.sqrt(temp_inv.conj().permute(0, 2, 1) @ V[:, :, :, k] @ temp_inv) # [513, 1, 2] * [513, 2, 2] * [513, 2, 1]
        W_k = W_k / temp_W_k # [513, 1, 2]
        dw = W_k - W[:, k, :].unsqueeze(1) # [513, 1 ,2]
        W[:, k, :] = W_k[:, 0, :] # [513, 2]
        #update A
        # A_temp = A # [513, 2, 2]
        # A_temp_2 = A[:, :, k].unsqueeze(2) # [513, 2] -> [513, 2, 1]
        Anumer = A[:, :, k].unsqueeze(2) @ dw @ A # [513, 2, 1] * [513, 1, 2] * [513, 2, 2]
        Adenom_new = 1  + dw @ A[:, :, k].unsqueeze(2) # [513, 1, 2] * [513, 2, 1]
        Adenom_new = Adenom_new.real
        epsi_mat = torch.ones_like(Adenom_new) * epsi
        Adenom_new = torch.maximum(Adenom_new, epsi_mat)
        # A_temp = Anumer / Adenom_new # [513, 2, 2] / [513, 1, 1]
        A = A - Anumer / Adenom_new # [513, 2, 2]
    return A, W

def update_v(V, alpha, p, xxh):
    K_num = V.shape[1]
    for k in range(K_num):
        V[:, :, :, k] = alpha * V[:, :, :, k] + p[k] * xxh # phi就是x*x^H
    return V

def update_u(W, xxh, p, alpha, U, X):
    N_effective, K_num = W.shape[0], W.shape[-1]
    for k in range(K_num):
        Unumer = p[k] *  U[:, :, :, k] @ xxh @  U[:, :, :, k] # x * [513, 2, 2] * [513, 2, 2] * [513, 2, 2]
        # Udenom_temp_X1 = X.conj().unsqueeze(1) # [513, 2] -> [513, 1, 2]
        # Udenom_temp_X2 = X.unsqueeze(2) # [513, 2] -> [513, 2, 1]
        # Udenom_temp_U =  U[:, :, :, k] # [2, 2, 513] -> [513, 2, 2]
        Udenom = alpha**2 + alpha * p[k] * X.conj().unsqueeze(1) @ U[:, :, :, k] @ X.unsqueeze(2)
        epsi_mat = torch.ones((N_effective, 1, 1), dtype = torch.float64) * epsi
        # Udenom = Udenom.real
        Udenom = torch.maximum(Udenom.real, epsi_mat)
        # U_temp =  U[:, :, :, k] / alpha - Unumer / Udenom
        U[:, :, :, k] = U[:, :, :, k] / alpha - Unumer / Udenom # [513, 2, 2]
    return U

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
    # gamma_wpe = 0.995
    wpe_beta = 0.9999

    # initialization
    # r = torch.zeros((K, 1), dtype = torch.float64)
    
    V1 =  torch.zeros((N_effective, K, K), dtype = complex_type)#[513, 20, 2]
    # u1 = V1.clone()
    # u2 = V1.clone()
    V2 = torch.zeros((N_effective, K, K), dtype = complex_type)#[513, 20, 2]
    # W = torch.zeros((N_effective, K, K), dtype = complex_type)
    G_wpe1 = torch.zeros((N_effective, ref_num*K, K), dtype = complex_type)#[513, 20, 2]
    G_wpe2 = torch.zeros((N_effective, ref_num*K, K), dtype = complex_type)#[513, 20, 2]
    # K_wpe = torch.zeros((N_effective, ref_num*K, 1), dtype = complex_type)#[513, 20, 1]
    # K_wpe2 = torch.zeros((N_effective, ref_num*K, 1), dtype = complex_type)#[513, 20, 1]
    # phi = torch.zeros((N_effective, K, K), dtype = complex_type)
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    # wpe_sigma = torch.zeros(N_effective, K, K)
    # init
    W = temp_eye.clone()
    Wbp = temp_eye.clone()
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
        label = label.permute(2, 1, 0)
    C, N_fre, N_frame = X_mix_stft.shape
    X_mix_stft = X_mix_stft.permute(2, 1, 0).contiguous() # [C, fre, time] -> [time, fre, C]
    # aux_IVA_online
    for iter in range(1):
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)
        Y_all[0:ref_num+delay_num, ...] = X_mix_stft[0:ref_num+delay_num, ...]
        # wpe_buffer = X_mix_stft[0:ref_num, :, :]

        for i in tqdm(range(delay_num+ref_num, N_frame), ascii=True):
        # for i in range(delay_num+ref_num, N_frame):
            if torch.prod(torch.prod(X_mix_stft[i, :, :]==0))==1:
                Y_all[i, :, :] = X_mix_stft[i, :, :]
            else:
                wpe_buffer = X_mix_stft[i-delay_num-ref_num:i-delay_num].permute(1, 2, 0) # [10, 513, 2]- >[513, 2, 10]
                x_D = torch.flatten(wpe_buffer, -2, -1).unsqueeze(-1) #[513, 20, 1]
                y1 = X_mix_stft[i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                y2 = X_mix_stft[i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                # s1 = W[:, ..., [0]].conj().transpose(-1, -2) @ y1.unsqueeze(-1) # [513, 1, 1]
                # s2 = W[:, ..., [1]].conj().transpose(-1, -2) @ y2.unsqueeze(-1)
                # a = Wbp[..., [0], :]
                pred_1 = Wbp[..., [0], :] @ y1.unsqueeze(-1)
                pred_2 = Wbp[..., [1], :] @ y2.unsqueeze(-1)
                Y_all[i] = torch.cat([pred_1, pred_2], dim=-2).squeeze(-1)

                # update
                vn1 = torch.mean(torch.abs(pred_1)**2)
                vn2 = torch.mean(torch.abs(pred_2)**2)

                K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + x_D.conj().transpose(-1, -2) @ invR_WPE1 @ x_D) # [513, 20, 1]
                invR_WPE1 = (invR_WPE1 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE1) / wpe_beta # [513, 20, 20]
                G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()

                K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + x_D.conj().transpose(-1, -2) @ invR_WPE2 @ x_D) # [513, 20, 1]
                invR_WPE2 = (invR_WPE2 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE2) / wpe_beta # [513, 20, 20]
                G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
                # y_wpe[i] = torch.cat([s1, s2], dim=1).squeeze(-1)
                
                yyh1 = y1.unsqueeze(2) @ y1.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                V1 = alpha_iva * V1 + (1-alpha_iva) * yyh1/ vn1 # [513, 2, 2]
                V1 = V1 + 1e-6*temp_eye
                # a = torch.linalg.inv(W.conj().transpose(-1, -2) @ V1)
                # print(a.shape)
                w1 = torch.linalg.inv(W @ V1)[...,[0]]
                w1 = w1 / torch.sqrt(w1.conj().transpose(-1, -2) @ V1 @ w1)
                # W = torch.cat((w1, w2), dim=-1)

                yyh2 = y2.unsqueeze(2) @ y2.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                V2 = alpha_iva * V2 + (1-alpha_iva) * yyh2 / vn2
                V2 = V2 + 1e-6*temp_eye
                w2 = torch.linalg.inv(W @ V2)[...,[1]]
                w2 = w2 / torch.sqrt(w2.conj().transpose(-1, -2) @ V2 @ w2)

                W[...,[0],:] = w1.conj().transpose(-1, -2)
                W[...,[1],:] = w2.conj().transpose(-1, -2)
                # calculate output
                Wbp = (torch.linalg.pinv(W) * temp_eye) @ W # [513, 2, 2] * [513, 2, 2]

    Y_all = Y_all.permute(2, 1, 0).contiguous()
    # y_wpe = y_wpe.permute(2, 1, 0).contiguous()
    # print(Y_all.shape)
    # y_wpe = torch.istft(y_wpe, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    y_iva = torch.istft(Y_all, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    # print(y_iva.shape)
    return y_iva

if __name__ == "__main__":
    import time
    mix_path = r'audio\2Mic_2Src_Mic.wav'
    out_path = r'audio\test_wpe_iva.wav'
    clean_path = r'audio\2Mic_2Src_Ref.wav'
    clean, sr = sf.read(clean_path)
    clean = torch.from_numpy(clean.T)
    # load singal
    x , sr = sf.read(mix_path)
    # x = x[:10*16000]
    print(x.shape, x.dtype)
    x = torch.from_numpy(x.T)
    start_time = time.time()
    y = auxIVA_online(x, N_fft = 1024, hop_len=256, label=clean)
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.T, sr)
    

    # mix_path = r'audio\2Mic_2Src_Mic.wav'
    # out_path = r'audio\test_wpe_iva_use_clean.wav'
    # clean_path = r'audio\2Mic_2Src_Ref.wav'
    # clean, sr = sf.read(clean_path)
    # clean = torch.from_numpy(clean.T)
    # # load singal
    # x , sr = sf.read(mix_path)
    # x = x[..., :clean.shape[-1]]
    # print(x.shape, x.dtype)
    # x = torch.from_numpy(x.T)
    # start_time = time.time()
    # y, y_wpe = auxIVA_online(x, N_fft = 1024, hop_len=256, label=clean)
    # end_time = time.time()
    # print('the cost of time {}'.format(end_time - start_time))
    # sf.write(out_path, y.T, sr)
    # sf.write('audio\\test_wpe_iva_wpeout_use_clean.wav', y_wpe.T, sr)