from tqdm import tqdm
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
import numpy as np
import soundfile as sf
import librosa


stft_options = dict(size=512, shift=128)
channels = 2
sampling_rate = 16000
delay = 3
alpha=0.9999
taps = 10
frequency_bins = stft_options['size'] // 2 + 1

y, sr = librosa.load('rt_3.wav', sr=None, mono=False)


Y = stft(y, **stft_options).transpose(1, 2, 0)
T, _, _ = Y.shape

def aquire_framebuffer():
    buffer = list(Y[:taps+delay, :, :])
    for t in range(taps+delay+1, T):
        buffer.append(Y[t, :, :])
        yield np.array(buffer)
        buffer.pop(0)

Z_list = []
online_wpe = OnlineWPE(
    taps=taps,
    delay=delay,
    alpha=alpha,
    channel=2
)

for Y_step in tqdm(aquire_framebuffer()):
    Z_list.append(online_wpe.step_frame(Y_step))

Z = np.stack(Z_list)
z = istft(np.asarray(Z).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])
sf.write('wpe_output.wav', z.T, samplerate=sr)