import torch
import soundfile as sf
from tqdm import tqdm
from torch.optim import lr_scheduler

epsi = torch.finfo(torch.float64).eps
device = torch.device('cpu')
from models import my_model, real_type, complex_type
from loss import cal_spec_loss, functional

def update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye):
    yyh1 = y1.unsqueeze(-1) @ y1.unsqueeze(-2).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    V1 = alpha_iva * V1 + (1-alpha_iva) * yyh1/ vn1# [513, 2, 2]
    new_V1 = V1 + 1e-6*temp_eye
    w1 = torch.linalg.inv(W @ new_V1)[...,[0]]
    new_w1 = w1 / torch.sqrt(w1.conj().permute(0, 1, -1, -2) @ new_V1 @ w1)

    yyh2 = y2.unsqueeze(-1) @ y2.unsqueeze(-2).conj() # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
    V2 = alpha_iva * V2 + (1-alpha_iva) * yyh2 / vn2
    new_V2 = V2 + 1e-6*temp_eye
    w2 = torch.linalg.inv(W @ new_V2)[...,[1]]
    new_w2 = w2 / torch.sqrt(w2.conj().permute(0, 1, -1, -2) @ new_V2 @ w2)
    new_W = torch.cat([new_w1.conj().permute(0, 1, -1, -2), new_w2.conj().permute(0, 1, -1, -2)], dim=-2)
    # W[...,[0],:] = w1.conj().transpose(-1, -2)
    # W[...,[1],:] = w2.conj().transpose(-1, -2)
    Wbp = (torch.linalg.pinv(new_W) * temp_eye) @ new_W # [513, 2, 2] * [513, 2, 2]
    return new_V1, new_V2, new_W, Wbp

def auxIVA_online(X_mix_stft, label = None, ref_num=10):
    B, K, N_effective, N_frame = X_mix_stft.shape
    model = my_model(N_effective).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha_iva = 0.96
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=1498, gamma=0.5)
    
    delay_num = 0
    joint_wpe = True
    wpe_beta = 0.9999
    training = True

    # initialization
    # r = torch.zeros((K, 1), dtype = torch.float64)
    
    G_wpe1 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
    G_wpe2 = torch.zeros((B, N_effective, ref_num*K, K), dtype = complex_type, device=device)#[513, 20, 2]
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    temp_eye = torch.repeat_interleave(temp_eye.unsqueeze(0), B, dim=0)
    # wpe_sigma = torch.zeros(N_effective, K, K)
    # init
    W = temp_eye.clone().detach()
    Wbp = temp_eye.clone().detach()
    V1 = temp_eye.clone().detach() * 1e-6
    V2 = temp_eye.clone().detach() * 1e-6
    invR_WPE1 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    invR_WPE1 = torch.repeat_interleave(invR_WPE1.unsqueeze(0), B, dim=0)
    invR_WPE2 = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    invR_WPE2 = torch.repeat_interleave(invR_WPE2.unsqueeze(0), B, dim=0)
    X_mix_stft = X_mix_stft.permute(0, 3, 2, 1).contiguous()
    label = label.permute(0, 3, 2, 1).contiguous()
    # aux_IVA_online
    for iter in range(1):
        # init paras and buffers
        Y_all = torch.zeros_like(X_mix_stft)
        wpe_buffer = functional.pad(X_mix_stft, pad=[0, 0, 0, 0, delay_num+ref_num, 0]).permute(0, 2, 3, 1).contiguous()
        bar = tqdm(range(N_frame), ascii=True)
        total_loss = torch.tensor(0.)
        for i in bar:
            optimizer.zero_grad()
            if torch.prod(torch.prod(X_mix_stft[:, i]==0))==1:
                Y_all[:, i] = X_mix_stft[:, i]
            else:
                if joint_wpe:
                    temp = wpe_buffer[..., i+delay_num:i+delay_num+ref_num] # [10, 513, 2]- >[513, 2, 10]
                    x_D = torch.flatten(temp, -2, -1).unsqueeze(-1) #[513, 20, 1]
                    y1 = X_mix_stft[:, i] - (G_wpe1.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                    y2 = X_mix_stft[:, i] - (G_wpe2.conj().transpose(-1, -2) @ x_D).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
                else:
                    y1 = X_mix_stft[:, i]
                    y2 = X_mix_stft[:, i]
                real_vn1 = torch.mean(torch.abs(label[:, i,...,0])**2)
                real_vn2 = torch.mean(torch.abs(label[:, i,...,1])**2)
                if training:
                    inpt1 = (Wbp[..., [0], :] @ y1.unsqueeze(-1)).permute(0, 2, 3, 1).squeeze(-2) #[B, 513, 1, 20]* [B, 512, 2, 1]->[B, 513, 1, 1]
                    inpt2 = (Wbp[..., [1], :] @ y2.unsqueeze(-1)).permute(0, 2, 3, 1).squeeze(-2) # -> [B, 1, 513]
                    weight1 = model(inpt1)
                    weight2 = model(inpt2)
                    # inpt = torch.cat([inpt1, inpt2], dim=-2)
                    # weights = model(inpt)
                    # weight1, weight2 = weights[...,0], weights[...,1]
                    vn1 = torch.mean(torch.abs(weight1*inpt1)**2)
                    vn2 = torch.mean(torch.abs(weight2*inpt2)**2)
                else:
                    vn1 = torch.mean(torch.abs(Wbp[..., [0], :] @ y1.unsqueeze(-1))**2)
                    vn2 = torch.mean(torch.abs(Wbp[..., [1], :] @ y2.unsqueeze(-1))**2)
                if joint_wpe:
                    temp = x_D.conj().transpose(-1, -2)
                    K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + temp @ invR_WPE1 @ x_D) # [513, 20, 1]
                    new_invR_WPE1 = (invR_WPE1 -K_wpe @ temp @ invR_WPE1) / wpe_beta # [513, 20, 20]
                    new_G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()
                    K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + temp @ invR_WPE2 @ x_D) # [513, 20, 1]
                    new_invR_WPE2 = (invR_WPE2 -K_wpe @ temp @ invR_WPE2) / wpe_beta # [513, 20, 20]
                    new_G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
                new_V1, new_V2, new_W, new_Wbp = update_auxiva(V1, V2, y1, y2, vn1, vn2, W, alpha_iva, temp_eye)
                pred_1 = new_Wbp[..., [0], :] @ y1.unsqueeze(-1) # [513, 1, 1]
                pred_2 = new_Wbp[..., [1], :] @ y2.unsqueeze(-1)
                pred = torch.cat([pred_1, pred_2], dim=-2).squeeze(-1)
                Y_all[:,i] = pred
                loss = cal_spec_loss(torch.abs(pred), torch.abs(label[:, i]))
                # loss = functional.mse_loss(real_vn1, vn1) + functional.mse_loss(real_vn2, vn2)
                total_loss += loss.item()
                bar.set_postfix({'loss':f'{total_loss/i+1:.5f}', 'lr':f"{optimizer.state_dict()['param_groups'][0]['lr']:.5e}"}, refresh=False)
                if training:
                    loss.backward()
                    scheduler.step()
                if joint_wpe:
                    G_wpe1 = new_G_wpe1.detach()
                    invR_WPE1 = new_invR_WPE1.detach()
                    invR_WPE2 = new_invR_WPE2.detach()
                    G_wpe2 = new_G_wpe2.detach()
                Wbp = new_Wbp.detach()
                W = new_W.detach()
                V1 = new_V1.detach()
                V2 = new_V2.detach()
                # torch.cuda.empty_cache()
    Y_all = Y_all.permute(0, 3, 2, 1)
    return Y_all

if __name__ == "__main__":
    import time
    import torchaudio
    # reb = 1
    mix_path = 'audio\\2Mic_2Src_Mic.wav'
    out_path = 'audio\\nara_wpe_iva.wav'
    clean_path = 'audio\\2Mic_2Src_Ref.wav'

    clean, sr = torchaudio.load(clean_path)
    x, _ = torchaudio.load(mix_path)
    
    # batch stft
    batch_x = torch.repeat_interleave(x.unsqueeze(0), repeats=4, dim=0)
    batch_x = torch.flatten(batch_x, 0, 1).contiguous()
    mix_spec = torch.stft(batch_x, 1024, 256, window=torch.hann_window(1024), return_complex=True).to(device)
    B, F, T = mix_spec.shape
    mix_spec = mix_spec.reshape(B//2, 2, F, T)

    batch_clean = torch.repeat_interleave(clean.unsqueeze(0), repeats=4, dim=0)
    batch_clean = torch.flatten(batch_clean, 0, 1).contiguous()
    clean_spec = torch.stft(batch_clean, 1024, 256, window=torch.hann_window(1024), return_complex=True).to(device)
    # B, F, T = mix_spec.shape
    clean_spec = clean_spec.reshape(B//2, 2, F, T)

    print(mix_spec.shape)

    start_time = time.time()
    batch_sep_spec = auxIVA_online(mix_spec, label=clean_spec, ref_num=5)
    end_time = time.time()

    batch_sep_spec = torch.flatten(batch_sep_spec, 0, 1).contiguous()
    batch_y = torch.istft(batch_sep_spec, 1024, 256, window=torch.hann_window(1024))
    batch_y = batch_y.reshape(B//2, 2, -1)
    y = batch_y[0]
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, y.detach().cpu().T, sr)
    