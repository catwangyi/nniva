import torch
import torch.nn.functional as functional

def cal_spec_loss(pred, label):
    # pred [513, 2]
    pred0_label0 = functional.mse_loss(pred[..., 0], label[..., 0])
    pred1_label0 = functional.mse_loss(pred[..., 1], label[..., 0])
    pred0_label1 = functional.mse_loss(pred[..., 0], label[..., 1])
    pred1_label1 = functional.mse_loss(pred[..., 1], label[..., 1])
    
    if pred0_label0 < pred0_label1: # 说明pred0和label0对应
        loss1 = pred0_label0
        loss2 = pred1_label1
    else:
        loss1 = pred0_label1
        loss2 = pred1_label0

    return (loss1 +loss2) / 2

def cal_p_loss(pred_p, real_p):
    # p [2, 1]
    pred0_label0 = functional.mse_loss(pred_p[..., 0, :], real_p[..., 0, :])
    pred1_label0 = functional.mse_loss(pred_p[..., 1, :], real_p[..., 0, :])
    pred0_label1 = functional.mse_loss(pred_p[..., 0, :], real_p[..., 1, :])
    pred1_label1 = functional.mse_loss(pred_p[..., 1, :], real_p[..., 1, :])
    
    if pred0_label0 < pred0_label1: # 说明pred0和label0对应
        loss1 = pred0_label0
        loss2 = pred1_label1
    else:
        loss1 = pred0_label1
        loss2 = pred1_label0

    return (loss1 +loss2) / 2