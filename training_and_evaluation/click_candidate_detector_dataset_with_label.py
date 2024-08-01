import os
import numpy as np 
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_audio_file(audio_path,sr):
    raw_audio , sr = librosa.load(audio_path,mono=False,sr=sr)
    return raw_audio#-np.mean(raw_audio)


class LongFileDataset(Dataset):

    def __init__(self, audio, context_radius, center_radius, click_list_frames): 
        self.full_audio = audio
        self.context_radius = context_radius
        self.center_radius = center_radius
        self.click_list_frames = click_list_frames

        self.moments = []

        for window_center in range(self.context_radius,self.full_audio.shape[1]-self.context_radius+1,2*self.center_radius):
            self.moments.append(window_center)
            

    def __len__(self):
        return len(self.moments)

    
    def __getitem__(self, idx):
        window_center = self.moments[idx]

        norm_window_start = max(0,window_center-5020)
        norm_window_end = min(self.full_audio.shape[-1],window_center+5020)

        norm_val = np.mean(self.full_audio[:,norm_window_start:norm_window_end])

        context_window_start = int(window_center-self.context_radius)
        context_window_end   = int(window_center+self.context_radius)

        center_window_start = window_center-self.center_radius
        center_window_end   = window_center+self.center_radius

        has_click = np.any(np.logical_and(center_window_start <= self.click_list_frames,self.click_list_frames < center_window_end))

        label = 0
        if has_click:
            label = 1

        return self.full_audio[:,context_window_start:context_window_end]-norm_val, label