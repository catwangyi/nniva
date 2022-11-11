from ast import Pass
from numpy import dtype
import torch
import torchaudio


def calcu_harmonic_clique(subbandorharmonic):
    nfft = 1024
    fs = 16000
    K = nfft//2 + 1

    if subbandorharmonic:
        H = 7
        bandsize = (K-1)/(H+1)
        C_HK = torch.zeros(H, K)
        for h in range(H):
            pass
    else:
        # generate harmonic clique matrix
        F1 = 55
        H = 39
        rf = 10
        F = F1*2**(torch.Tensor([i/rf for i in range(H)]))
        freqs = [i*fs/nfft for i in range(K)]
        C_HK = torch.zeros(H+1, K)
        delta = 1-2**(-1/12)
        M=10
        for h in range(H):
            for k in range(K):
                for m in range(M):
                    if abs(freqs[k] - m*F[h]) / (m*F[h]) < delta:
                        C_HK[h, k] = 1
        C_HK[-1, :] = 1
    return C_HK

def joint_WPEIVA(x, beta, ref_frames, ref_delay, n_fft,ref_sig=None, hop_len=None, window=None, alpha_wpe=0.2, beta_wpe=0.9995, alpha_iva=0.96):
    beta_iva = beta
    if hop_len == None:
        hop_len = n_fft // 4 
    if window == None:
        window = torch.hann_window(n_fft, periodic=True)

    X_MIC = torch.stft(x, n_fft, hop_len, window=window, return_complex=True).transpose(-1, -2)
    #[N, T, F]
    if ref_sig!=None:
        REF_SRC = torch.stft(ref_sig, n_fft, hop_len, window=window, return_complex=True).transpose(-1, -2)
    del x

    n_chan, n_frames, n_freq = X_MIC.shape
    
    subbandorharmonic = -1
    if subbandorharmonic >= 0:
        # C_HK = torch.rand(40, 513)
        # 若subbandorharmonic=0， harmonic矩阵为(40, 513)
        # 若subbandorharmonic=1, subband矩阵为（7, 513）
        C_HK =  calcu_harmonic_clique(subbandorharmonic)
        H = C_HK.shape[0]
        r_band = torch.zeros(n_chan, H)
        NC = torch.sum(C_HK, dim=1, keepdim=True) # NC为取band取到的频带数量
    else:
        C_HK = torch.ones(1, 513)
        H = 1
    # WPE part
    K_WPE = torch.zeros(ref_frames*n_chan**2, n_chan, n_freq, dtype=torch.complex64)
    invQ_WPE = torch.eye(ref_frames*n_chan**2, ref_frames*n_chan**2).unsqueeze(-1) * torch.ones(ref_frames*n_chan**2, ref_frames*n_chan**2, n_freq)
    invQ_WPE = torch.complex(invQ_WPE, invQ_WPE)
    G_WPE = torch.zeros(ref_frames*n_chan**2, n_freq, dtype=torch.complex64)
    X_BUFFER = X_MIC[:, [i for i in range(ref_frames+ref_delay, 0, -1)],...] #如何实现逆序
    X_BUFFER = torch.cat((X_BUFFER, torch.zeros(n_chan, 1, n_freq)), dim=1)
    Sigma = torch.zeros(n_chan, n_chan, n_freq, dtype=torch.complex64)
    X = torch.zeros_like(X_MIC) # X是
    X_GWPE = torch.zeros_like(X_MIC)

    #AuxIVA part
    Y = torch.zeros_like(X_MIC, dtype=torch.complex64)
    W = torch.zeros(n_chan, n_chan, n_freq, dtype=torch.complex64)
    Wbp = torch.zeros(n_chan, n_chan, n_freq, dtype=torch.complex64) # wbp是分离用的矩阵
    r = torch.zeros(n_chan, n_frames)
    phi = torch.zeros(n_chan, H, n_frames)
    lamuda = torch.zeros(n_chan, H, n_frames)
    for freq_bin in range(n_freq):
        W[..., freq_bin] = torch.eye(n_chan)
        Wbp[..., freq_bin] = torch.eye(n_chan)
    I = torch.eye(n_chan, dtype=torch.complex64)
    V = torch.zeros(n_chan, n_chan, H, n_chan, n_freq, dtype=torch.complex64)
    # Lb = 2

    tmp = torch.zeros(n_chan, n_frames)
    
    # initialization
    for frame in range(ref_frames + ref_delay):
        for freq in range(n_freq):
            X[:, frame, freq] = X_MIC[:, frame, freq]
            Y[:, frame, freq] = Wbp[...,freq] @ X[:, frame, freq]
        if ref_sig != None:
            r[:, frame] = torch.sqrt(torch.sum(torch.abs(REF_SRC[:,frame,:]**2), dim=-1))
        else:
            r[:, frame] = torch.sqrt(torch.sum(torch.abs(Y[:,frame,:])**2, dim=-1))
        # phi[..., frame] = 2 * beta_iva * (r[:, [frame]])**(2* beta_iva - 2)
        X_GWPE[:, frame, :] = X[:, frame, :]

    for frame in range(ref_frames + ref_delay, n_frames):
        print(frame)
        # GWPE update
        X_BUFFER = torch.cat((X_MIC[:, [frame], :], X_BUFFER[:, :-1, :]), dim=1)
        for freq in range(n_freq):
            # Precalculate the dereverberated signal X using the current WPE
            X_D = X_BUFFER[:, -ref_frames:, freq]
            X_D = torch.kron(torch.eye(n_chan), X_D).reshape(n_chan*n_chan*ref_frames, n_chan)
            X_D = X_D.T
            X[:, frame, freq] = X_BUFFER[:, 0, freq] - X_D @ G_WPE[:, freq] # X_buffer减去混响部分

            # Obtain the spatial covariance matrix using the demixing matrix
            if ref_sig != None:
                D_tmp = torch.linalg.inv(Wbp[:, :, freq]) @ torch.diag(REF_SRC[:, frame, freq])
                Sigma[..., freq] = alpha_wpe * Sigma[..., freq] + (1-alpha_wpe) * (D_tmp @ torch.conj(D_tmp.T))
            else:
                Y[:, frame, freq] = Wbp[..., freq] @ X[:, frame, freq]
                D_tmp = torch.linalg.inv(Wbp[..., freq]) @ torch.diag(Y[:, frame, freq]) # 解混响之后用wbp进行分离，分离完取对角线元素进行合成，得到D_tmp
                Sigma[..., freq] = alpha_wpe * Sigma[..., freq] + (1-alpha_wpe) * (D_tmp @ torch.conj(D_tmp.T)) # spatial covariance matrix 更新
            
            # Update the WPE parameter 'g_wpe'
            nominator = invQ_WPE[..., freq] @ torch.conj(X_D.T)
            K_WPE[..., freq] = nominator @ torch.linalg.inv(beta_wpe * Sigma[..., freq] + X_D @ nominator)
            invQ_WPE[..., freq] = (invQ_WPE[..., freq] - K_WPE[..., freq] @ (X_D @ invQ_WPE[..., freq])) / beta_wpe
            G_WPE[:, freq] = G_WPE[:, freq] + K_WPE[..., freq] @ X[:, frame, freq] # 和论文里不一样

            # Calculate the dereverberation output

        X_GWPE[:, frame, :] = X[:, frame, :]

        if torch.prod(torch.prod(X[:, frame,:]==0))==1:
            Y[:, frame, :] = X[:, frame, :]
        else:
            for freq_bin in range(n_freq):
                Y[:, frame, freq_bin] = Wbp[:, :, freq_bin] @ X[:, frame, freq_bin]
            # ICD modification
            if subbandorharmonic < 0:
                tmp[:, frame] = torch.sum(torch.abs(Y[:, frame,:])**2, dim=-1)
                # Auxiliary variable updates
                r[:, frame] = tmp[:, frame] ** 0.5 # 此处可简化为 torch.sum(torch.abs(Y[:, frame,:])**2, dim=-1) ** 0.5
                # a = r[:, [frame]] ** (2-2*beta)
                phi[..., frame] = 2 * beta * (r[:, [frame]] ** (2*beta-2)) # phi是指 G'/r
            else:
                # SigmaY = torch.abs(Y[:, frame, :])**2
                r_band = (torch.abs(Y[:, frame, :])**2 @ C_HK.T) **0.5
                # a = NC.T
                lamuda[:, :, frame] = torch.sqrt((torch.abs(Y[:, frame, :])**2 @ C_HK.T) / (NC.T)) # 此处NC.T如果为0，会导致nan
                phi[..., frame] = torch.sqrt((torch.abs(Y[:, frame, :])**2 @ C_HK.T) / (NC.T))**(-1) # phi是指 G'/r

            for freq_bin in range(n_freq):
                # a= X[:, [frame], freq_bin]
                # tmp = X[:, [frame], freq_bin] @ torch.conj(X[:, [frame], freq_bin].T) # 经查，发现是tmp中freq_bin写成了freq
                for k in range(n_chan):
                    for c in range(H):
                        # a =  phi[k, c, frame] * tmp # tmp为负，导致出错
                        V[..., c, k, freq_bin] = alpha_iva * V[..., c, k, freq_bin] + (1-alpha_iva) * phi[k, c, frame] * (X[:, [frame], freq_bin] @ torch.conj(X[:, [frame], freq_bin].T)) # V是负的，导致出错
                    # if the current frequency belongs to clique j, we take j
                    # into consideration and average over the cliques.
                    # a = torch.sum(V[..., k, freq_bin] * C_HK[:, None,[freq_bin]].permute(1, 2, 0), dim=2)
                    # b = torch.sum(C_HK[:, freq_bin], dim=0)
                    Vtmp = torch.sum(V[..., k, freq_bin] * C_HK[:, None,[freq_bin]].permute(1, 2, 0), dim=2) / torch.sum(C_HK[:, freq_bin], dim=0)
                    # Here is to avoid singularity
                    gamma = torch.trace(Vtmp)
                    Vtmp = Vtmp + gamma * 1e-6*I

                    # auxIVA update
                    w = torch.linalg.inv(W[..., freq_bin]@ Vtmp) @ I[:, [k]]
                    # a= (torch.conj(w.T) @ Vtmp) @ w
                    # print(a)
                    # b = torch.sqrt(a)
                    # print(b)
                    w = w @ torch.linalg.inv(torch.sqrt(torch.conj(w.T) @ Vtmp @ w)) # 因为 w'@V@w 的结果有负数，所以会导致取sqrt出错
                    W[k, :, freq_bin] = torch.conj(w.T.squeeze())
                # MDP scaling
                Wbp[:, :, freq_bin] = torch.diag(torch.diag(torch.linalg.pinv(W[..., freq_bin]))) @ W[..., freq_bin]
            
            # Calculate outputs
            for k in range(n_freq):
                Y[:, frame, k] = Wbp[:, :, k] @ X[:, frame, k]
    
    y = torch.istft(Y.transpose(-1, -2), n_fft, hop_len, window=window)
    x_GWPE = torch.istft(X_GWPE.transpose(-1, -2), n_fft, hop_len, window=window)
    return y, W, x_GWPE

if __name__ == "__main__":
    import soundfile
    import time

    start = time.perf_counter()
    audio, sr  = torchaudio.load('2Mic_2Src_Mic.wav')
    y,_, gwpe = joint_WPEIVA(audio, beta=0.33, ref_frames=10, ref_delay=1, n_fft=1024)
    
    soundfile.write('seperated.wav', y.T.numpy(), sr)
    soundfile.write('gwpe.wav', gwpe.T.numpy(), sr)
    end = time.perf_counter()
    print('time: %s'%(end-start))