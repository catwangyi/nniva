
from inspect import FrameInfo
from re import A
import torch
import torchaudio
from tqdm import tqdm

def update_p(W, x_temp):
    '''
    args:
        W : [F, N, N]
        x_temp : [F, N, 1]
    '''
    Y = W @ x_temp # [F, N, 1]
    return 1 / torch.sqrt(torch.sum(torch.abs(Y)**2, dim=0)).clamp_min(1e-12)# [N, 1]

def update_demixing_matrix(U_k, A, x_temp, p_k, alpha, Vk, k, init):
    '''
    args:
        U_k : [F, N, N]
        V_k : [F, N, N]
        A : [F, N, N]
        x_temp : [F, N, 1]
        p_k : [1]
    '''
    A_k = A[..., k].unsqueeze(-1) # [F, N , 1]
    p = (1-alpha) * p_k #[1]
    Unum = p * U_k @ x_temp @ x_temp.conj().transpose(-1, -2) @ U_k #  [F, N, N] * [F, N, 1] * [F, 1, N] * [F, N, N] - >[F, N, N]
    Udeo = alpha**2 + alpha * p * x_temp.conj().transpose(-1, -2) @ U_k @ x_temp # [1, N, 1] * [F, 1, N] * [F, N, N ] * [F, N, 1] -> [F, N, 1]
    Udeo = Udeo
    U_k = U_k / alpha - Unum / (Udeo)

    if init:
        inv = U_k
    else:
        inv = U_k @ A_k # [F, N, N] * [F, N, 1] ->[F, N, 1]
    wk = inv.conj().transpose(-1, -2) # [F, 1, N]
    new_wk = wk / torch.sqrt(wk @ Vk @ inv) # [F, 1, N]
    dw = new_wk - wk # [F, 1, N]
    wk = new_wk.squeeze(1) # [F, N]
    
    Anum = A_k @ dw @ A # [F, N, N] * [F, N, N] * [F, N, N]
    Adeo = 1 + dw @ A_k
    Adeo = Adeo
    # Adeo = Adeo
    # temp_a = A
    # print(torch.min(Anum /Adeo))
    if torch.any(torch.isnan(Anum)):
        print('Anum is nan') 
    if torch.any(torch.isnan(Adeo)):
        print('Adeo is nan')
    A = A - Anum / Adeo
    # if torch.any(torch.isnan(U_k)) or torch.any(torch.isnan(A)):
    #     a = 1
    #     if torch.any(torch.isnan(U_k)) and torch.any(torch.isnan(A)):
    #         a = 1
    #     elif torch.any(torch.isnan(A)):
    #         a = 1
    #     elif torch.any(torch.isnan(U_k)):
    #         a = 1
    if torch.any(torch.isnan(A)):
        print('nan!')
        a = 1
    return A, U_k, wk

    

def online_auxiva(X, iter_num=1, alpha=0.996):
    n_channel, frame_num, freq_num = X.shape
    U = torch.zeros(freq_num, n_channel, n_channel, n_channel, dtype=torch.complex64) # [F, N, N, N]
    W = torch.zeros(freq_num, n_channel,  n_channel, dtype=torch.complex64) # [F, N, N]
    A = torch.zeros(freq_num, n_channel,  n_channel, dtype=torch.complex64) # [F, N, N]
    V = torch.zeros(freq_num, n_channel, n_channel, n_channel, dtype=torch.complex64)# [F, N, N, N]
    EYE = torch.repeat_interleave(torch.eye(n_channel, dtype=torch.complex64).unsqueeze(0), repeats=freq_num, dim=0) # [F, N, N]
    Y = torch.zeros_like(X)
    for frame_idx in tqdm(range(frame_num)):
        # print(frame_idx)
        
        if frame_idx == 0:
            # initialization
            W = EYE.clone() # [F, N, N]
            A = EYE.clone() # [F, N, N]
            Y[:, frame_idx, :] = X[:, frame_idx, :]
            x_temp = X[:, [frame_idx], :].permute(2, 0, 1).contiguous()
            p = update_p(W, x_temp)# [N, 1] 这里的p就是1/r
            for k in range(n_channel):
                V[..., k] = x_temp @ x_temp.conj().transpose(-1, -2) * EYE * p[k]
                # U[..., k] = 1 / (V[..., k] + 1e-6) * EYE
                U[..., k] = EYE.clone()
        else:
            for iter_idx in range(iter_num):
                if torch.prod(torch.prod(X[:, frame_idx,:]==0))==1:
                     Y[:, frame_idx, :] = 0
                else:
                    x_temp = X[:, [frame_idx], :].permute(2, 0, 1).contiguous() # [F, N, 1]
                    p = update_p(W, x_temp)# [N, 1] 这里的p就是1/r
                    for k in range(n_channel):
                        V[:, :, :, k] = alpha * V[:, :, :, k] + (1-alpha) * p[k] * x_temp @ x_temp.conj().transpose(-1, -2)
                        if torch.any(torch.isnan(p[k])):
                            print('nan!')
                        # U_k, A, x_temp, phi_k, alpha, Vk, k
                        if torch.any(torch.isnan(U[..., k])):
                            print('1nan!')
                        if torch.any(torch.isnan(A)):
                            print('2nan!')
                        if torch.any(torch.isnan(x_temp)):
                            print('3nan!')
                        if torch.any(torch.isnan(p[k])):
                            print('4nan!')
                        if torch.any(torch.isnan(V[..., k])):
                            print('5nan!')
                        A, U[..., k], W[:, k, :] = update_demixing_matrix(U[..., k], A, x_temp, p[k], alpha, V[..., k], k, True if frame_idx==0 else False) 
                        if torch.any(torch.isnan(W[:, k, :])):
                            print('6nan!')
                    W_bp = A @ EYE @ W
                    Y[:, frame_idx, :] = (W_bp @ x_temp).squeeze(-1).permute(-1, -2)
    return Y



if __name__ == "__main__":
    # X = torch.rand(2, 1200, 513, dtype=torch.complex64)
    sig, sr = torchaudio.load('2Mic_2Src_Mic.wav')
    X = torch.stft(sig, n_fft=1024, hop_length=256, return_complex=True,window=torch.hann_window(1024))
    Y = online_auxiva(X.permute(0, -1, -2))
    y = torch.istft(Y.transpose(-1, -2), n_fft=1024, hop_length=256,window=torch.hann_window(1024))
    import soundfile
    soundfile.write('my_sep.wav', y.numpy().T, samplerate=sr)
