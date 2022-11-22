import torch
import soundfile as sf
from tqdm import tqdm
from torch.optim import lr_scheduler

epsi = torch.finfo(torch.float64).eps
device = torch.device('cuda:0')
from models import my_model, real_type, complex_type, crn_model
from loss import cal_spec_loss, functional, cal_p_loss

def update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye):
    yyh1 = y1.unsqueeze(2) @ y1.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    new_V1 = alpha_iva * V1 + (1-alpha_iva) * yyh1/ vn1# [513, 2, 2]
    # new_V1 = V1 + 1e-6*temp_eye
    w1 = torch.linalg.inv(W @ new_V1)[...,[0]]
    new_w1 = w1 / torch.sqrt(w1.conj().transpose(-1, -2) @ new_V1 @ w1)

    yyh2 = y2.unsqueeze(2) @ y2.unsqueeze(1).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    new_V2 = alpha_iva * V2 + (1-alpha_iva) * yyh2 / vn2
    # new_V2 = V2 + 1e-6*temp_eye
    w2 = torch.linalg.inv(W @ new_V2)[...,[1]]
    new_w2 = w2 / torch.sqrt(w2.conj().transpose(-1, -2) @ new_V2 @ w2)
    new_W = torch.cat([new_w1.conj().transpose(-1, -2), new_w2.conj().transpose(-1, -2)], dim=-2)
    # W[...,[0],:] = w1.conj().transpose(-1, -2)
    # W[...,[1],:] = w2.conj().transpose(-1, -2)
    Wbp = (torch.linalg.pinv(new_W) * temp_eye) @ new_W # [513, 2, 2] * [513, 2, 2]
    return new_V1, new_V2, new_W, Wbp

def auxIVA_online(x, N_fft = 2048, hop_len = 0, label = None, ref_num=10):
    print(x.shape, x.dtype)
    K, N_y  = x.shape
    # parameter
    N_fft = N_fft
    N_move = hop_len
    N_effective = int(N_fft/2+1) #也就是fft后，频率的最高点
    window = torch.hann_window(N_fft, periodic = True, dtype=real_type, device=device)
    model = my_model(N_effective).to(device)
    # model = crn_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha_iva = 0.96
    sche = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    delay_num=1
    joint_wpe = True
    wpe_beta = 0.9999

    # initialization
    # r = torch.zeros((K, 1), dtype = torch.float64)
    
    G_wpe1 = torch.zeros((N_effective, (ref_num//2)*K, K), dtype = complex_type, device=device)#[513, 20, 2]
    G_wpe2 = torch.zeros((N_effective, (ref_num//2)*K, K), dtype = complex_type, device=device)#[513, 20, 2]
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    # wpe_sigma = torch.zeros(N_effective, K, K)
    # init
    W = temp_eye.clone().detach()
    Wbp = temp_eye.clone().detach()
    V1 = temp_eye.clone().detach() * 1e-6
    V2 = temp_eye.clone().detach() * 1e-6
    invR_WPE1 = torch.repeat_interleave(torch.eye((ref_num//2)*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    invR_WPE2 = torch.repeat_interleave(torch.eye((ref_num//2)*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    X_mix_stft = torch.stft(x.to(device), 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True).type(complex_type)
    if label is not None:
        label = torch.stft(label.to(device), 
                             n_fft = N_fft,
                             hop_length = N_move, 
                             window = window,
                             return_complex=True).type(complex_type)
        label = label.permute(2, 1, 0) # [time, fre, C]
    C, N_fre, N_frame = X_mix_stft.shape
    X_mix_stft = X_mix_stft.permute(2, 1, 0).contiguous() # [C, fre, time] -> [time, fre, C]
    # aux_IVA_online
    for iter in range(50):
        total_loss = torch.tensor(0.)
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)
        start = 2*ref_num
        wpe_buffer = torch.cat([torch.zeros(start+delay_num, N_effective, K, dtype=complex_type, device=device), X_mix_stft], dim=0)
        bar = tqdm(range(start, start+N_frame), ascii=True)
        for i in bar:
            optimizer.zero_grad()
            if torch.prod(torch.prod(X_mix_stft[i-start, :, :]==0))==1:
                Y_all[i-start, :, :] = X_mix_stft[i-start, :, :]
            else:
                if joint_wpe:
                    low = wpe_buffer[i-ref_num//2:i, :N_effective//2]
                    high = wpe_buffer[i-ref_num:i:2, N_effective//2:]
                    temp =  torch.cat([low, high], dim=1).permute(1, 2, 0)# [10, 513, 2]- >[513, 2, 10]
                    # temp = wpe_buffer[i-start:i].permute(1, 2, 0) # [10, 513, 2]- >[513, 2, 10]
                    x_D = torch.flatten(temp, -2, -1).unsqueeze(-1) #[513, 20, 1]
                    y1 = X_mix_stft[i-start, :, :] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                    y2 = X_mix_stft[i-start, :, :] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                else:
                    y1 = X_mix_stft[i-start]
                    y2 = X_mix_stft[i-start]
                # vn1 = torch.mean(torch.abs(label[i-start,...,0])**2)
                # vn2 = torch.mean(torch.abs(label[i-start,...,1])**2)
                real_vn1 = torch.mean(torch.abs(label[i-start,...,0])**2)
                real_vn2 = torch.mean(torch.abs(label[i-start,...,1])**2)

                inpt1 = (Wbp[..., [0], :] @ y1.unsqueeze(-1)).T #[1, 1, 513]
                inpt2 = (Wbp[..., [1], :] @ y2.unsqueeze(-1)).T
                # weight1 = model(inpt1)
                # weight2 = model(inpt2)
                inpt = torch.cat([inpt1, inpt2], dim=-2)
                weights = model(inpt)
                weight1, weight2 = weights[...,0,:], weights[...,1,:]
                vn1 = torch.mean(torch.abs(weight1*inpt1)**2)
                vn2 = torch.mean(torch.abs(weight2*inpt2)**2)
                # vn1 = torch.mean(torch.abs(Wbp[..., [0], :] @ y1.unsqueeze(-1))**2)
                # vn2 = torch.mean(torch.abs(Wbp[..., [1], :] @ y2.unsqueeze(-1))**2)
                if joint_wpe:
                    K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + x_D.conj().transpose(-1, -2) @ invR_WPE1 @ x_D) # [513, 20, 1]
                    new_invR_WPE1 = (invR_WPE1 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE1) / wpe_beta # [513, 20, 20]
                    new_G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()
                    K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + x_D.conj().transpose(-1, -2) @ invR_WPE2 @ x_D) # [513, 20, 1]
                    new_invR_WPE2 = (invR_WPE2 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE2) / wpe_beta # [513, 20, 20]
                    new_G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
                new_V1, new_V2, new_W, new_Wbp = update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye)
                pred_1 = new_Wbp[..., [0], :] @ y1.unsqueeze(-1) # [513, 1, 1]
                pred_2 = new_Wbp[..., [1], :] @ y2.unsqueeze(-1)
                pred = torch.cat([pred_1, pred_2], dim=-2).squeeze(-1)
                Y_all[i-start] = pred
                loss = cal_spec_loss(torch.abs(pred), torch.abs(label[i-start]))
                # loss = functional.mse_loss(real_vn1, vn1) + functional.mse_loss(real_vn2, vn2)
                total_loss += loss.item()
                bar.set_postfix({'loss':f'{total_loss.item() / i:.5e}'})
                loss.backward(retain_graph=False)
                # optimizer.step()
                sche.step()
                if joint_wpe:
                    G_wpe1 = new_G_wpe1.detach()
                    invR_WPE1 = new_invR_WPE1.detach()
                    invR_WPE2 = new_invR_WPE2.detach()
                    G_wpe2 = new_G_wpe2.detach()
                Wbp = new_Wbp.detach()
                W = new_W.detach()
                V1 = new_V1.detach()
                V2 = new_V2.detach()

    Y_all = Y_all.permute(2, 1, 0).contiguous()
    y_iva = torch.istft(Y_all, n_fft=N_fft, hop_length=N_move, window=window, length=N_y)
    return y_iva

if __name__ == "__main__":
    import time
    # reb = 1
    mix_path = r'audio\T60_04\2Mic_2Src_Mic.wav'
    clean_path = r'audio\T60_04\2Mic_2Src_Ref.wav'
    nfft = 1024
    out_path = f'audio\T60_04\dnn_n2n_freq_nara_wpe_iva_{nfft}_50epoch.wav'
    clean, sr = sf.read(clean_path)
    clean = torch.from_numpy(clean.T)
    # load singal
    x , sr = sf.read(mix_path)
    # x = x[:5*16000]
    print(x.shape, x.dtype)
    x = torch.from_numpy(x.T)
    start_time = time.time()

    # ref_num = int(0.4*reb*sr-4*nfft) // nfft +1
    # print('refnum:', ref_num if ref_num>=1 else 1)
    with torch.autograd.set_detect_anomaly(True):
        y = auxIVA_online(x, N_fft = nfft, hop_len=nfft//4, label=clean, ref_num=10)
    end_time = time.time()
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.detach().cpu().T, sr)
    