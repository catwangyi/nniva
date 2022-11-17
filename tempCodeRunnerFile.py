
                # vn1 = torch.mean(torch.abs(s1)**2)
                # vn2 = torch.mean(torch.abs(s2)**2)
                # K_wpe = (invR_WPE1 @ x_D) / (wpe_beta * vn1 + x_D.conj().transpose(-1, -2) @ invR_WPE1 @ x_D) # [513, 20, 1]
                # invR_WPE1 = (invR_WPE1 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE1) / wpe_beta # [513, 20, 20]
                # G_wpe1 = G_wpe1 + K_wpe @ y1.unsqueeze(-1).transpose(-1, -2).conj()

                # K_wpe = (invR_WPE2 @ x_D) / (wpe_beta * vn2 + x_D.conj().transpose(-1, -2) @ invR_WPE2 @ x_D) # [513, 20, 1]
                # invR_WPE2 = (invR_WPE2 -K_wpe @ x_D.conj().transpose(-1, -2) @ invR_WPE2) / wpe_beta # [513, 20, 20]
                # G_wpe2 = G_wpe2 + K_wpe @ y2.unsqueeze(-1).transpose(-1, -2).conj()
                # # y_wpe[i] = torch.cat([s1, s2], dim=1).squeeze(-1)
                
                # X1 = y1 # [time, fre, C] -> [fre, C]
                # xxh1 = torch.matmul(X1.unsqueeze(2), X1.unsqueeze(1).conj()) # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                # V1 = alpha_iva * V1 + (1-alpha_iva) * xxh1/ vn1 # [513, 2, 2]
                # # a = torch.linalg.inv(W.conj().transpose(-1, -2) @ V1)
                # # print(a.shape)
                # w1 = torch.linalg.inv(W.conj().transpose(-1, -2) @ V1)[...,[0]]
                # w1 = w1 @ torch.linalg.inv(w1.conj().transpose(-1, -2) @ V1 @ w1)

                # X2 = y2 # [time, fre, C] -> [fre, C]
                # xxh2 = torch.matmul(X2.unsqueeze(2), X2.unsqueeze(1).conj()) # [513, 2, 1] * [513, 1, 2] -> [513, 2, 2]
                # V2 = alpha_iva * V2 + (1-alpha_iva)*xxh2 / vn2
                # w2 = torch.linalg.inv(W.conj().transpose(-1, -2) @ V2)[...,[1]]
                # w2 = w2 @ torch.linalg.inv(w2.conj().transpose(-1, -2) @ V2 @ w2)

                # W = torch.cat((w1, w2), dim=-1)
                # # calculate output
                # A_temp = torch.linalg.inv(W) * temp_eye # [513, 2, 2]
                # W_temp = W # [513, 2, 2]