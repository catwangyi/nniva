from pystoi import stoi
import soundfile
import torchaudio
import torch
import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq

def cal_STOIi(src_ref, src_est, mix, sr):
    """Calculate Short-Time Objective Intelligibility improvement (STOIi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T] or [C, T]
    Returns:
        average_SISNRi
    """
    stoi_model = []
    stoi_orgin = []
    if mix.ndim == 2:
        mix = mix[0]
    for src_ref_one, src_est_one in zip( src_ref, src_est):
        stoi_estimate = stoi(src_ref_one, src_est_one, sr, extended=False)
        stoi_model.append(stoi_estimate)
        stoi_mix = stoi(src_ref_one, mix, sr, extended=False)
        stoi_orgin.append(stoi_mix)
    avg_model_STOI = (stoi_model[0] + stoi_model[1]) / 2
    avg_orgin_STOI = (stoi_orgin[0] + stoi_orgin[1]) / 2
    avg_STOIi = ((stoi_model[0] - stoi_orgin[0]) + (stoi_model[1] - stoi_orgin[1])) / 2
    return avg_STOIi, avg_model_STOI, avg_orgin_STOI

def cal_PESQi(src_ref, src_est, mix, sr):
    """Calculate Perceptual evaluation of speech quality improvement (PESQi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T] or [C, T]
    Returns:
        average_SISNRi
    """
    pesq_model = []
    pesq_orgin = []
    if mix.ndim == 2:
        mix = mix[0]
    for src_ref_one, src_est_one in zip( src_ref, src_est):
        pesq1 = pesq(sr, src_ref_one, src_est_one)
        pesq_model.append(pesq1)
        pesq1_b = pesq(sr, src_ref_one, mix)
        pesq_orgin.append(pesq1_b)
    avg_model_PESQ = (pesq_model[0] + pesq_model[1]) / 2
    avg_orgin_PESQ = (pesq_orgin[0] + pesq_orgin[1]) / 2
    avg_PESQi = ((pesq_model[0] - pesq_orgin[0]) + (pesq_model[1] - pesq_orgin[1])) / 2
    return avg_PESQi, avg_model_PESQ, avg_orgin_PESQ

def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    if mix.ndim == 2:
        mix = mix[0]
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T] or [C, T], but we choose the first channel
    Returns:
        average_SDRi
    """
    if mix.ndim == 1:
        src_anchor = np.stack([mix, mix], axis=0)
    else:
        src_anchor = np.stack([mix[0], mix[0]], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    avg_SIRi = ((sir[0]-sir0[0]) + (sir[1]-sir0[1])) / 2
    avg_SARi = ((sar[0]-sar0[0]) + (sar[1]-sar0[1])) / 2

    return avg_SDRi, avg_SIRi, avg_SARi

if __name__ == "__main__":
    label, sr = torchaudio.load(r'audio\2Mic_2Src_Ref.wav')
    esti, _ = torchaudio.load('audio\wpe_iva.wav')

    print(f'channel:{0}\tstoi:{max(stoi(label[0], esti[0], fs_sig=sr), stoi(label[0], esti[1], fs_sig=sr)):.5f}')
    print(f'pesq:{pesq(sr, label[0].numpy(), esti[0].numpy()):.5f}')
    print(f'channel:{1}\tstoi:{max(stoi(label[1], esti[0], fs_sig=sr), stoi(label[1], esti[1], fs_sig=sr)):.5f}')
    print(f'pesq:{pesq(sr, label[1].numpy(), esti[0].numpy()):.5f}')