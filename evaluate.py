
# -*- coding: utf-8 -*-
"""
Created on 2021.12.9
at Tencent in Beijing 
@author: little Wang
"""

import os
import argparse
import librosa
from mir_eval.separation import bss_eval_sources
from pit_criterion import cal_loss
import numpy as np
import torch
import logging
from logger import set_logger
from pesq import pesq
from pystoi import stoi

# General config
parser = argparse.ArgumentParser('Evaluate separation performance')
parser.add_argument('--cal_sdr', type=bool, default=True, help='Whether calculate SDR')
parser.add_argument('--cal_sisnr', type=bool, default=True, help='Whether calculate SISNR')
parser.add_argument('--cal_pesq', type=bool, default=True, help='Whether calculate PESQ')
parser.add_argument('--cal_stoi', type=bool, default=True, help='Whether calculate STOI')
parser.add_argument('--log_path',default='./',help='Location to save the logging')

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

def order(estimate, reference):
    reference_torch = torch.from_numpy(reference)
    estimate_torch = torch.from_numpy(estimate)
    mixture_lengths = np.array([reference_one.size(-1) for reference_one in reference_torch])
    mixture_lengths_torch = torch.from_numpy(mixture_lengths)
    _, _, _, reorder_estimate_source = \
                cal_loss(reference_torch, estimate_torch, mixture_lengths_torch)
    reorder_estimate = reorder_estimate_source.numpy()
    return reorder_estimate
    
def run(mixture_origin, estimate_origin, reference_origin, sr, args):
    """ The main function
    Args:
        mixture:    numpy.ndarray, [T] or [C, T]
		estimate:   numpy.ndarray, [C, T], reordered by best PIT permutation
		reference:  numpy.ndarray, [C, T]
    Returns:
        SDRi, SIRi, SARi, SISNRi, PESQi, STOIi; (The criterions which you need)
    """

    logger = logging.getLogger('evaluate_tools')

    # if input is single sample
    if mixture_origin.ndim == 2:
        mixture_origin = mixture_origin[np.newaxis, :, :]
        estimate_origin = estimate_origin[np.newaxis, :, :]
        reference_origin = reference_origin[np.newaxis, :, :]
    
    # For Maximizing SISNR to reorder the channel
    estimate_reorder = order(estimate_origin, reference_origin)

    total_cnt = 0
    for reference, estimate, mixture in zip(reference_origin, estimate_reorder, mixture_origin):
        logger.info("Utt:{}".format(total_cnt + 1))
        if args.cal_sdr:      
            avg_SDRi, avg_SIRi, avg_SARi = cal_SDRi(reference, estimate, mixture)
            logger.info("This is BSS_eval")
            logger.info("\tSDRi={0:.2f}".format(avg_SDRi))
            logger.info("\tSIRi={0:.2f}".format(avg_SIRi))
            logger.info("\tSARi={0:.2f}".format(avg_SARi))

        if args.cal_sisnr:
            avg_SISNRi = cal_SISNRi(reference, estimate, mixture)
            logger.info("This is SISNR")
            logger.info("\tSI-SNRi={0:.2f}".format(avg_SISNRi))

        if args.cal_pesq:
            avg_PESQi, avg_model_PESQ, avg_orgin_PESQ = cal_PESQi(reference, estimate, mixture, sr)
            logger.info("This is PESQ")
            logger.info("\tmodel_PESQ={0:.2f}".format(avg_model_PESQ))
            logger.info("\torgin_PESQ={0:.2f}".format(avg_orgin_PESQ))
            logger.info("\tPESQi={0:.2f}".format(avg_PESQi))

        if args.cal_pesq:
            avg_STOIi, avg_model_STOI, avg_orgin_STOI = cal_STOIi(reference, estimate, mixture, sr)
            logger.info("This is STOI")
            logger.info("\tmodel_STOI={0:.2f}".format(avg_model_STOI))
            logger.info("\torgin_STOI={0:.2f}".format(avg_orgin_STOI))
            logger.info("\tSTOIi={0:.2f}".format(avg_STOIi))
        total_cnt = total_cnt + 1

if __name__ == "__main__":
    args = parser.parse_args()
    args.log_path = 'log'
    set_logger.setup_logger('evaluate_tools', args.log_path,
                        screen=True, tofile=True)


    path_mixture = 'audio\2Mic_2Src_Mic.wav'
    path_estimate = 'audio\wzy_AuxIVA_online_pytorch.wav'
    path_reference = 'audio\2Mic_2Src_Ref.wav'
    mixture, sr = librosa.load(path_mixture, mono=False, sr=None)
    estimate, sr = librosa.load(path_estimate, mono=False, sr=None)
    reference, sr = librosa.load(path_reference, mono=False, sr=None)
    run(mixture, estimate, reference, sr, args)

    # path_mixture = os.path.join(args.log_path, 'mixture.wav')
    # path_estimate = os.path.join(args.log_path, 'ausIS_new7.wav')
    # path_reference = os.path.join(args.log_path, 'reference_spk1_spk2.wav')
    # mixture, sr = librosa.load(path_mixture, mono=False, sr=None)
    # estimate_all, sr = librosa.load(path_estimate, mono=False, sr=None)
    # reference, sr = librosa.load(path_reference, mono=False, sr=None)
    # frame = 64000
    # for i in range(5):
    #     estimate = estimate_all[:,i*64000:(i+1)*64000]
    #     run(mixture, estimate, reference, sr, args)

