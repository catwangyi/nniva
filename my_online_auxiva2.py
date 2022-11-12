from logging import exception
import os
import torch
import soundfile as sf
from tqdm import tqdm

epsi = torch.finfo(torch.float64).eps

def init(X, alpha, xxh, temp_eye, U, V):
    N_effective = torch.maximum(X.shape)
    K = torch.minimum(X.squeeze().shape)
    W = torch.repeat_interleave(torch.eye(K, dtype = torch.complex128).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    A = torch.repeat_interleave(torch.eye(K, dtype = torch.complex128).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    p = update_p(W, X, alpha, p)
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
    if(torch.any(a*d == b*c)):
        raise Exception('mat can not be inversed')
    new_mat = torch.zeros_like(mat)
    new_mat[..., 0, 0] = d / (a*d-b*c)
    new_mat[..., 0, 1] = -b / (a*d-b*c)
    new_mat[..., 1, 0] = -c / (a*d-b*c)
    new_mat[..., 1, 1] = a / (a*d-b*c)
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

def update_p(W, X, alpha, p):
    K_num = W.shape[-1]
    for k in range(K_num): 
        # r_hat_W1 = W[:, k, :].unsqueeze(1) # [513, 2] -> [513, 1, 2]
        # r_hat_phi = torch.matmul(phi_temp1, phi_temp2) # [513, 2, 2]
        # r_hat_W2 = W[:, k, :].conj().unsqueeze(2) # [513, 2] ->[513, 2, 1]
        temp_sum = torch.sum(W[:, k, :].unsqueeze(1) @ X.unsqueeze(2) @\
        X.unsqueeze(1).conj() @ W[:, k, :].conj().unsqueeze(2))
        # a = r_hat_W1 @ X.unsqueeze(2) @ X.unsqueeze(1).conj() @ r_hat_W2
        deo = torch.sqrt(temp_sum.real)
        if deo < epsi:
            deo = epsi
        p[k] = (1-alpha) / deo # p就是1/r
    return p
def update(V, alpha, p, xxh, X, W, U, A):
    p = update_p(W, X, alpha, p)
    V = update_v(V, alpha, p, xxh)
    U = update_u(W, xxh, p, alpha, U, X)
    A, W = update_a_w(A, W, U, V)
    return A, W, V, U

def auxIVA_online(x, N_fft = 1024, hop_len = 0, clean_sig = None):
    print(x.shape, x.dtype)
    K, N_y  = x.shape
    # parameter
    N_fft = N_fft
    N_move = hop_len
    N_effective = int(N_fft/2+1) #也就是fft后，频率的最高点
    window = torch.sqrt(torch.hann_window(N_fft, periodic = True, dtype=torch.float64))

    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha = 0.96
    
    index = 0
    iter_num = 1

    # initialization
    Y_all = []
    # r = torch.zeros((K, 1), dtype = torch.float64)
    p  = torch.zeros((K, 1), dtype = torch.float64)
    V = torch.zeros((N_effective, K, K, K), dtype = torch.complex128)
    U = torch.zeros((N_effective, K, K, K), dtype = torch.complex128)
    W = torch.zeros((N_effective, K, K), dtype = torch.complex128)
    A = torch.zeros((N_effective, K, K), dtype = torch.complex128)
    # phi = torch.zeros((N_effective, K, K), dtype = torch.complex128)
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = torch.complex128).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]

    # init
    # W = temp_eye.clone()
    # A = temp_eye.clone()

    X_mix_stft = torch.stft(x, 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True)
    if clean_sig != None:
        clean_stft = torch.stft(clean_sig, 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True)
        clean_stft = clean_stft.permute(2, 1, 0).contiguous()
    C, N_fre, N_frame = X_mix_stft.shape
    X_mix_stft = X_mix_stft.permute(2, 1, 0).contiguous() # [C, fre, time] -> [time, fre, C]        

    # aux_IVA_online
    for iter in range(iter_num):
        Y_all = []
        for i in tqdm(range(N_frame), ascii=True):
            if torch.prod(torch.prod(X_mix_stft[i, :, :]==0))==1:
                Y_all.append(X_mix_stft[i, :, :].unsqueeze(2).permute(1, 0, 2))
            else:
                X = X_mix_stft[i, :, :] # [time, fre, C] -> [fre, C]
                phi_temp1 = X.unsqueeze(2)  # [513, 2] -> [513, 2, 1]
                phi_temp2 = X.unsqueeze(1).conj() # [513, 2] -> [513, 1, 2]
                xxh = torch.matmul(phi_temp1, phi_temp2) # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                
                if index == 0 and iter_num==0:
                    A, W, U, V = init(X, alpha, xxh, temp_eye, U, V)
                else:
                    A, W, U, V = update(V, alpha, p, xxh, X, W, U, A)
                # calculate output
                A_temp = A * temp_eye # [513, 2, 2]
                W_temp = W # [513, 2, 2]
                Wbp = A_temp @ W_temp # [513, 2, 2] * [513, 2, 2]
                Y_temp = Wbp @ X.unsqueeze(2) # [513, 2, 2] * [513, 2, 1] 
                Y_all.append(Y_temp.permute(1, 0, 2))
                index = index + N_move

    Y = torch.cat(Y_all, dim = -1)
    print(Y.shape)
    y = torch.istft(Y, 
                    n_fft = N_fft, 
                    hop_length = N_move, 
                    window = window, 
                    length = N_y)
    print(y.shape)
    return y

if __name__ == "__main__":
    import time
    mix_path = r'2Mic_2Src_Mic.wav'
    out_path = r'AuxIVA_online_pytorch.wav'
    clean_path = r'2Mic_2Src_Ref.wav'

    # load singal
    x , sr = sf.read(mix_path)
    clean, _ = sf.read(clean_path)
    print(x.shape, x.dtype)
    x = torch.from_numpy(x.T)
    clean = torch.from_numpy(clean.T)
    x = x[:, :clean.shape[-1]]
    start_time = time.time()
    y = auxIVA_online(x, N_fft = 2048, hop_len=512, clean_sig=None)
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.T, sr)