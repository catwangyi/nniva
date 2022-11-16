import torch
import soundfile as sf
import torchaudio
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
epsi = torch.finfo(torch.float64).eps
complex_type = torch.complex128
real_type = torch.float64
device = torch.device('cpu')
from models import my_model
from loss import cal_p_loss


def init(X, alpha, xxh, temp_eye, Y=None):
    N_effective = max(X.shape)
    K = min(X.squeeze().shape)
    V = torch.zeros((N_effective, K, K, K), dtype = complex_type, device=device)
    N_effective = max(X.shape)
    K = min(X.squeeze().shape)
    W = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    A = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    p = create_p(W, X, alpha, Y)
    temp_v = []
    temp_u = []
    for k in range(K):
        temp_v.append((xxh *p[k]) * temp_eye)
        temp_u.append(1 / (V[:, :, :, k] + epsi) * temp_eye)
    return A, W, torch.stack(temp_u, dim=-1), torch.stack(temp_v, dim=-1)

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

def update(V, alpha_iva, p, xxh, X, W, U, A):
    p = create_p(W, X, alpha_iva, p)
    V = update_v(V, alpha_iva, p, xxh)
    U = update_u(W, xxh, p, alpha_iva, U, X)
    A, W = update_a_w(A, W, U, V)
    return A, W, U, V

def auxIVA_online(inpt, target=None, using_dnn=True):
    K, N_effective, N_frame = inpt.shape
    #注意matlab的hanning不是从零开始的，而python的hanning是从零开始
    alpha_iva = 0.96
    
    initial = 0
    ref_num=10
    delay_num=2
    gamma_wpe = 0.995
    wpe_beta = 0.5
    model = my_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # initialization
    Y_all = []
    p  = torch.zeros((K, 1), dtype = real_type, device=device)
    
    U = torch.zeros((N_effective, K, K, K), dtype = complex_type, device=device)
    W = torch.zeros((N_effective, K, K), dtype = complex_type, device=device)
    A = torch.zeros((N_effective, K, K), dtype = complex_type, device=device)
    G_wpe = torch.zeros((N_effective, ref_num*(K), 1), dtype = complex_type, device=device)
    K_wpe = torch.zeros((N_effective, ref_num*(K), K), dtype = complex_type, device=device)
    temp_eye = torch.repeat_interleave(torch.eye(K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 2, 2]
    wpe_sigma = torch.zeros(N_effective, K, K, device=device)
    # init
    W = temp_eye.clone()
    A = temp_eye.clone()
    Wbp = temp_eye.clone()
    invQ_WPE = torch.repeat_interleave(torch.eye(ref_num*K, dtype = complex_type, device=device).unsqueeze(0), N_effective, dim = 0) # [513, 40, 40]
    X_mix_stft = inpt.permute(2, 1, 0)# [C, fre, time] -> [time, fre, C]
    # print(f'inpt:{inpt._version}')
    label = target.permute(2, 1, 0)
    
    y_wpe = torch.zeros_like(X_mix_stft, dtype=complex_type)
    loss=torch.tensor(0.)
    X_D = torch.zeros(N_effective, K, K*ref_num, dtype=torch.complex128)
    # aux_IVA_online
    for iter in range(1):
        Y_all = []
        # init paras and buffers
        for i in range(ref_num+delay_num):
            Y_all.append(X_mix_stft[i])
        y_wpe[0:ref_num+delay_num, :, :] = X_mix_stft[0:ref_num+delay_num, :, :]
        wpe_buffer = X_mix_stft[0:ref_num, :, :]
        bar = tqdm(range(N_frame), ascii=True)
        for i in bar:
        # for i in range(ref_num+delay_num, N_frame):
            # print(f'frame:{i}')
            # wpe_buffer = X_mix_stft[i-delay_num-ref_num:i-delay_num].permute(1, 2, 0) # [513, 2, 10]
            # X_D[...,0,:ref_num] = wpe_buffer[...,0,:]
            # X_D[...,1,-ref_num:] = wpe_buffer[...,1,:]
            # y_wpe[i, :, :] = X_mix_stft[i, ...] -  (X_D @ G_wpe).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]
            # Y_all.append((Wbp @ y_wpe[i,...].unsqueeze(-1)).squeeze(-1))
            # # temp_diag = torch.zeros_like(W)
            # # temp_diag[..., 0, 0] = Y_all[i,..., 0]
            # # temp_diag[..., 1, 1] = Y_all[i,..., 1]
            # # a = torch.diag_embed(Y_all[i, ...])
            # sig = torch.linalg.inv(Wbp) @ torch.diag_embed(Y_all[i])
            # wpe_sigma = (1-wpe_beta) * wpe_sigma + wpe_beta * sig @ sig.conj().transpose(-1, -2).contiguous() # [513, 2, 2]
            # nominator = invQ_WPE @ X_D.conj().transpose(-1, -2) # [513, 40, 40] * [513, 40, 2]-> [513, 40, 2]
            # K_wpe = nominator @ torch.linalg.inv(gamma_wpe * wpe_sigma + X_D @ nominator) # [513, 40, 2]
            # invQ_WPE = (invQ_WPE - K_wpe @ X_D @ invQ_WPE) / gamma_wpe
            # # G_wpe = G_wpe
            # G_wpe = G_wpe + K_wpe @ y_wpe[i, ...].unsqueeze(-1)
            # y_wpe[i, :, :] = X_mix_stft[i, ...] -  (X_D @ G_wpe).squeeze(-1) # [513, 2] - [513, 2, 40] *[513, 40, 1]

            if torch.prod(torch.prod(X_mix_stft[i, :, :]==0))==1:
                Y_all.append(X_mix_stft[i, :, :])
            else:
                X = X_mix_stft[i,...] # [time, fre, C] -> [fre, C]
                # print(f'x :{X_mix_stft._version}')
                # phi_temp1 = X.unsqueeze(2)  # [513, 2] -> [513, 2, 1]
                # phi_temp2 = X.unsqueeze(1).conj() # [513, 2] -> [513, 1, 2]
                xxh = torch.matmul(X.unsqueeze(2), X.unsqueeze(1).conj()) # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                if initial == 0:
                    A, W, U, V = init(X, alpha_iva, xxh, temp_eye)
                    initial = 1
                    Wbp =  A * temp_eye  @ W # [513, 2, 2] * [513, 2, 2]
                    Y_temp = Wbp @ X.unsqueeze(2) # [513, 2, 2] * [513, 2, 1] -> [513, 2, 1]
                    Y_all.append(Y_temp.squeeze()) #[513, 2]
                    continue
                else:
                    if using_dnn:
                        a = torch.abs(X) / torch.norm(torch.abs(X), p=2, dim=0, keepdim=True) # [513, 2]
                        p = model(a.T.unsqueeze(0)).squeeze(0)
                    else:
                        p = create_p(W, X, alpha_iva)
                    # print(f'model:{model.cov.weight._version}')
                    # print(f'a:{a._version}')
                    real_p = create_p(W, X, alpha_iva, Y=label[i])
                    new_V = update_v(V, alpha_iva, p, xxh) # update_v, 使用了V, 在backward完成后，释放了V的中间结果，但是这里又使用了V
                    new_U = update_u(W, xxh, p, alpha_iva, U, X) # update_u， 使用了W， U
                    new_A, new_W = update_a_w(A, W, new_U, new_V) #update_a_w， 使用了A, W, U, V，其中W, U, V都使用过
                    Wbp = new_A * temp_eye @ new_W
                    Y_temp = (Wbp @ X.unsqueeze(2)).squeeze(-1) # [513, 2, 2] * [513, 2, 1] -> [513, 2, 1]
                    Y_all.append(Y_temp) #[513, 2]
                
                loss = cal_p_loss(p, real_p)
                bar.set_postfix({'loss':f'{loss.item():.5e}'})
                # print(f'new_A:{new_A._version}')
                # print(f'new_U:{new_U._version}')
                # print(f'new_V:{new_V._version}')
                # print(f'new_W:{new_W._version}')
                # print(f'Wbp:{Wbp._version}')
                # print(f'>>>>>loss{loss.item()}')
                if using_dnn:
                    optimizer.zero_grad()
                    # print(f'Y_temp:{Y_temp._version}')
                    loss.backward(retain_graph=False) # backward完成后，释放了中间结果，如果再次使用，会出现报错
                    # print(f'model:{model.cov.weight._version}')
                    optimizer.step()        # calculate output
                    # print(f'model:{model.cov.weight._version}')
                    A = new_A.detach().clone()
                    W = new_W.detach().clone()
                    U = new_U.detach().clone()
                    V = new_V.detach().clone()
                # print('--------------------------------------------------------------')
                # A_temp = A * temp_eye # [513, 2, 2]
                # W_temp = W # [513, 2, 2]
                 # [513, 2, 2] * [513, 2, 2]
                # print('wbp', Wbp._version)
               
                    
                # print('Y_temp', Y_temp._version)
                # # loss = F.mse_loss(torch.abs(Y_temp.squeeze(-1)), torch.abs(label[i]))
        Y_all = torch.stack(Y_all, dim=0)
    # y_all [T, F, 2]
    return Y_all.permute(2, 1, 0), y_wpe.permute(2, 1, 0)

if __name__ == "__main__":
    import time
    dnn = True
    mix_path = r'audio\2Mic_2Src_Mic.wav'
    out_path = r'audio\dnn_iva_out.wav'

    # load singal
    x , sr = torchaudio.load(mix_path)
    target, sr = torchaudio.load(r'audio\2Mic_2Src_Ref.wav')
    # x = x[..., :5*16000]
    # target = target[..., :5*16000]
    mix_stft = torch.stft(x, n_fft=1024, hop_length=256, window=torch.hann_window(1024), return_complex=True).type(complex_type).to(device)
    target = torch.stft(target, n_fft=1024, hop_length=256, window=torch.hann_window(1024), return_complex=True).type(complex_type).to(device)
    print(mix_stft.shape, mix_stft.dtype)
    start_time = time.time()
    y_joint, _ = auxIVA_online(mix_stft, target, dnn)
    end_time = time.time()

    joint_out = torch.istft(y_joint, n_fft=1024, hop_length=256, window=torch.hann_window(1024, True).to(device))
    # wpe_out = torch.istft(y_wpe, n_fft=1024, hop_length=256, window=torch.hann_window(1024, True).to(device))
    print('the cost of time {}'.format(end_time - start_time))
    sf.write(out_path, joint_out.detach().cpu().T, sr)
    # sf.write('audio\dnn_wpe_out.wav', wpe_out.detach().cpu().T, sr)