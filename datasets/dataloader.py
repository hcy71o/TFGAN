import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import DistributedSampler, DataLoader, Dataset
from collections import Counter

from utils.utils import read_wav_np
from utils.stft import TacotronSTFT

#TODO condition 추가!

def create_dataloader(hp, args, train, device):
    if train:
        dataset = MelFromDisk(hp, hp.data.train_dir, hp.data.train_meta, args, train, device)
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=False,
                          num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)

    else:
        dataset = MelFromDisk(hp, hp.data.val_dir, hp.data.val_meta, args, train, device)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, data_dir, metadata_path, args, train, device):
        random.seed(hp.train.seed)
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = data_dir
        metadata_path = os.path.join(data_dir, metadata_path)
        self.meta = self.load_metadata(metadata_path)
        self.stft = TacotronSTFT(hp.audio.filter_length, hp.audio.hop_length, hp.audio.win_length,
                                 hp.audio.n_mel_channels, hp.audio.sampling_rate,
                                 hp.audio.mel_fmin, hp.audio.mel_fmax, center=False, device=device)
        
        self.cond1_stft = TacotronSTFT(hp.cond1.filter_length, hp.cond1.hop_length, hp.cond1.win_length,
                                 hp.cond1.n_mel_channels, hp.cond1.sampling_rate,
                                 hp.cond1.mel_fmin, hp.cond1.mel_fmax, center=False, device=device)

        self.cond2_stft = TacotronSTFT(hp.cond2.filter_length, hp.cond2.hop_length, hp.cond2.win_length,
                                hp.cond2.n_mel_channels, hp.cond2.sampling_rate,
                                hp.cond2.mel_fmin, hp.cond2.mel_fmax, center=False, device=device)

        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.shuffle = hp.train.spk_balanced

        if train and hp.train.spk_balanced:
            # balanced sampling for each speaker
            speaker_counter = Counter((spk_id \
                                       for audiopath, text, spk_id in self.meta))
            weights = [1.0 / speaker_counter[spk_id] \
                       for audiopath, text, spk_id in self.meta]

            self.mapping_weights = torch.DoubleTensor(weights)

        elif train:
            weights = [1.0 / len(self.meta) for _, _ in self.meta]
            self.mapping_weights = torch.DoubleTensor(weights)


    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = torch.multinomial(self.mapping_weights, 1).item()
            return self.my_getitem(idx)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping_weights)

    def my_getitem(self, idx):
        #! 데이터셋에 따라 다르게 설정
        # wavpath, _, _ = self.meta[idx]
        wavpath, _ = self.meta[idx]
        wavpath = os.path.join(self.data_dir, 'wavs', wavpath+'.wav')
        sr, audio = read_wav_np(wavpath)

        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    mode='constant', constant_values=0.0)

        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = self.get_mel(wavpath)
        cond1, cond2 = self.get_condition(wavpath)

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length -1
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]
            #TODO 
            ratio1 = self.hp.audio.hop_length//self.hp.cond1.hop_length
            ratio2 = self.hp.audio.hop_length//self.hp.cond2.hop_length

            cond1 = cond1[:, mel_start*ratio1:(mel_start+self.mel_segment_length)*ratio1]
            cond2 = cond2[:, mel_start*ratio2:(mel_start+self.mel_segment_length)*ratio2]

            audio_start = mel_start * self.hp.audio.hop_length
            audio_len = self.hp.audio.segment_length
            audio = audio[:, audio_start:audio_start + audio_len]

        return mel, audio, cond1, cond2

    def get_condition(self, wavpath):
        cond1path = wavpath.replace('.wav', '.cond1')
        cond2path = wavpath.replace('.wav', '.cond2')
        try:
            cond1 = torch.load(cond1path, map_location='cpu')
            assert cond1.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, cond1.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            cond1 = self.cond1_stft.mel_spectrogram(wav)
            cond1 = cond1.squeeze(0)

            torch.save(cond1, cond1path)

        try:
            cond2 = torch.load(cond2path, map_location='cpu')
            assert cond2.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, cond2.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            cond2 = self.cond2_stft.mel_spectrogram(wav)
            cond2 = cond2.squeeze(0)

            torch.save(cond2, cond2path)

        return cond1, cond2

    def get_mel(self, wavpath):
        melpath = wavpath.replace('.wav', '.mel')
        try:
            mel = torch.load(melpath, map_location='cpu')
            assert mel.size(0) == self.hp.audio.n_mel_channels, \
                'Mel dimension mismatch: expected %d, got %d' % \
                (self.hp.audio.n_mel_channels, mel.size(0))

        except (FileNotFoundError, RuntimeError, TypeError, AssertionError):
            sr, wav = read_wav_np(wavpath)
            assert sr == self.hp.audio.sampling_rate, \
                'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, wavpath)

            if len(wav) < self.hp.audio.segment_length + self.hp.audio.pad_short:
                wav = np.pad(wav, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(wav)), \
                             mode='constant', constant_values=0.0)

            wav = torch.from_numpy(wav).unsqueeze(0)
            mel = self.stft.mel_spectrogram(wav)

            mel = mel.squeeze(0)

            torch.save(mel, melpath)

        return mel

    def load_metadata(self, path, split="|"):
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip().split(split)
                metadata.append(stripped)

        return metadata