import torch
from torch.utils.data import Dataset
import os
import torchaudio

class audio_dataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.mix_path = []
        self.label_path = []

        for root, dirs, files in os.walk(path):
            if len(dirs)==0:
                for file in files:
                    if file == '2Mic_2Src_Mic.wav':
                        self.mix_path.append(os.path.join(root, file))
                    elif file == '2Mic_2Src_Ref.wav':
                        self.label_path.append(os.path.join(root, file))
        self.mix_path = sorted(self.mix_path)
        self.label_path = sorted(self.label_path)
    
    def __getitem__(self, index):
        mix_path = self.mix_path[index]
        clean_path = self.label_path[index]

        mix, _ = torchaudio.load(mix_path)
        label, _ = torchaudio.load(clean_path)
        # mix = mix[..., :2*16000]
        # label = label[..., :2*16000]
        mix_spec = torch.stft(mix, 1024, 256, window=torch.hann_window(1024), return_complex=True)
        label_spec = torch.stft(label, 1024, 256, window=torch.hann_window(1024), return_complex=True)

        return mix_spec, label_spec


    def __len__(self):
        return len(self.mix_path)


if __name__ =="__main__":
    dataset = audio_dataset('audio')
    for x, y in dataset:
        print(x.shape)
        print(y.shape)