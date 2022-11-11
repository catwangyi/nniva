from numpy import dtype
import torch
import torchaudio


def iva(mix_sig, ref_num=10, nfft=1024, clean_wav=None):
    hop_len = nfft // 4
    window = torch.hann_window(nfft, True)
    mix_spec = torch.stft(input=mix_sig, n_fft=nfft, hop_length=hop_len, window=window, return_complex=True).transpose(-1, -2)
    if clean_wav!=None:
        clean_spec = torch.stft(input=mix_sig, n_fft=nfft, hop_length=hop_len, window=window, return_complex=True).transpose(-1, -2)
    nchans, frames, freqs = mix_spec.shape
    beta = 0.33
    separated = torch.zeros_like(mix_spec)
    W = torch.zeros(nchans, nchans, freqs, dtype=torch.complex64)
    Wbp = torch.zeros(nchans, nchans, freqs, dtype=torch.complex64)
    for freq_idx in range(freqs):
        W[..., freq_idx] = torch.eye(nchans)
    V = torch.zeros(nchans, nchans, nchans, freqs, dtype=torch.complex64)
    V_ref = torch.zeros(nchans, nchans, nchans, freqs, ref_num, dtype=torch.complex64)
    I =  torch.eye(nchans, dtype=torch.complex64)
    
    # init
    for frame_idx in range(ref_num):
        separated[:, frame_idx] = torch.conj(W[..., freq_idx].T) @ mix_spec[:, frame_idx]
        for i in range(nchans):
            for freq_idx in range(freqs):
                if clean_wav==None:
                    r = torch.sum(torch.abs(separated[i,frame_idx]**2)) ** 0.5 
                else:
                    r = torch.sum(torch.abs(clean_spec[i,frame_idx]**2)) ** 0.5 
                V[..., i, freq_idx] = beta*2/(r**(2-2*beta)) * (mix_spec[:, frame_idx] @ torch.conj(mix_spec[:, frame_idx]).T)
                V_ref = torch.cat((V[..., None], V_ref[..., :-1]), dim=-1)

    for frame_idx in range(ref_num, frames):
        print(frame_idx)
        if torch.prod(torch.prod(mix_spec[:, frame_idx,:]==0))==1:
            separated[:, frame_idx] =  mix_spec[:, frame_idx]
        else:
            separated[:, frame_idx] = torch.conj(W[..., freq_idx].T) @ mix_spec[:, frame_idx]
            for i in range(nchans):
                if clean_wav == None:
                    r = torch.sum(torch.abs(separated[i,frame_idx, :]**2))**(1-beta)
                else:
                    r = torch.sum(torch.abs(clean_spec[i,frame_idx, :]**2))**(1-beta)
                for freq_idx in range(freqs):
                    V[..., i, freq_idx] = 2*beta/r * (mix_spec[:, frame_idx] @ torch.conj(mix_spec[:, frame_idx]).T)
                    V_ref = torch.cat((V[..., None], V_ref[..., :-1]), dim=-1)
                    v_e = torch.mean(V_ref, dim=-1)
                    w = torch.linalg.inv(W[..., freq_idx] @ v_e[..., i, freq_idx]) @ I[..., [i]]
                    w = w / (torch.conj(w).T @ v_e[..., i, freq_idx] @ w)**0.5
                    W[i,:, freq_idx] = torch.conj(w.T).squeeze()
                Wbp[:, :, freq_idx] = torch.diag(torch.diag(torch.linalg.pinv(W[..., freq_idx]))) @ W[..., freq_idx]
        separated[:, frame_idx] = torch.conj(Wbp[..., freq_idx].T) @ mix_spec[:, frame_idx]
    y = torch.istft(separated.transpose(-1, -2), n_fft=nfft, hop_length=hop_len, window=window)
    return y

if __name__ == "__main__":
    import soundfile
    mix, sr = torchaudio.load('2Mic_2Src_Ref.wav')
    y = iva(mix)
    soundfile.write('basic_iva.wav', y.T.numpy(), sr)
    # del y 
    # clean, _ = torchaudio.load('2Mic_2Src_Ref.wav')
    # y = iva(mix, clean_wav=clean)
    # soundfile.write('basic_iva_use_clean.wav', y.T.numpy(), sr)


