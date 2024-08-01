import os
import numpy as np 
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_audio_file(audio_path,sr):
    raw_audio , sr = librosa.load(audio_path,mono=False,sr=sr)
    return raw_audio-np.mean(raw_audio)


### New data ###

def get_all_data(audio_path,gt_path,sr):

    audio_file_names = os.listdir(audio_path)
    audios_and_clicks = []

    for i,audio_name in enumerate(audio_file_names):
        print('Loading audio file ',audio_name,' (',i+1,'/',len(audio_file_names),')',sep='')
        audio = load_audio_file(audio_path+audio_name,sr)

        gt_file_name = audio_name[:-3]+'csv'
        click_df = pd.read_csv(gt_path+gt_file_name)

        all_click_times_frames = np.array(click_df['click time (seconds)'])*sr
        audios_and_clicks.append((audio,all_click_times_frames))
    
    return audios_and_clicks
 
class LongFileDataset(Dataset):
    def get_clicks(self):
        if not self.is_train:
            print("Should not be happening")

        self.moments = []
        file_id = 0
        for audio, clicks in self.all_data:
            for click_time in clicks:
                self.moments.append((file_id,click_time))
            file_id += 1

    def get_all_moments(self):
        if self.is_train:
            print("Should not be happening")

        self.moments = []
        file_id = 0
        for audio, clicks in self.all_data:
            for window_center in range(self.context_radius,audio.shape[1]-self.context_radius,2*self.center_radius):
                self.moments.append((file_id,window_center))

            file_id += 1

    
    def extract_moment(self,file_id,center_time):
        audio, clicks = self.all_data[file_id]

        center_time = int(center_time)

        if self.is_train:
            center_time += np.random.randint(-self.max_offset,self.max_offset)

        context_window_start = center_time-self.context_radius
        context_window_end   = center_time+self.context_radius

        center_window_start = center_time-self.center_radius
        center_window_end   = center_time+self.center_radius

        has_click = np.any(np.logical_and(center_window_start <= clicks,clicks < center_window_end))

        label = 0
        if has_click:
            label = 1

        return audio[:,context_window_start:context_window_end], label



    def __init__(self, all_data, is_train, context_radius, center_radius, max_offset): 
        self.all_data = all_data
        self.is_train = is_train
        self.do_data_augmentation = is_train
        self.context_radius = context_radius
        self.center_radius = center_radius
        self.max_offset = max_offset

        if is_train:
            self.get_clicks()
        else:
            self.get_all_moments()


    def __len__(self):
        return len(self.moments)
    
    def __getitem__(self, idx):
        file_id, window_center = self.moments[idx]
        return self.extract_moment(file_id,window_center)


### Old data ###

class OneClickFileDataset(Dataset):
    def load_settings(self, settings):
        self.context_radius = settings["context_window_radius"]
        self.center_radius = settings["center_window_radius"]
        self.max_offset = settings["max_offset"]

    def __init__(self, old_data, sr):

        self.all_data = []
        for _ , row in old_data.iterrows():
            audio_path = row["audio_path"]
            click_time = sr*(row["click_time"]/22050)
            audio = load_audio_file(audio_path, sr)

            self.all_data.append((audio,click_time))


            
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        audio, click_time = self.all_data[idx]

        offset = np.random.randint(-self.max_offset,self.max_offset)
        context_window_start = int(click_time-self.context_radius+offset)
        context_window_end   = int(click_time+self.context_radius+offset)

        label = 1
        if offset > self.center_radius:
            label = 0

        return audio[:,context_window_start:context_window_end], label



class RandomNoiseDataset(Dataset):
    def load_settings(self, settings):
        self.context_window_radius = settings["context_window_radius"]


    def __init__(self, old_data, sr, dataset_size):
        self.dataset_size = dataset_size

        self.all_data = []
        for _ , row in old_data.iterrows():
            audio_path = row["audio_path"]
            audio = load_audio_file(audio_path, sr)
            self.all_data.append(audio)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _):
        idx = np.random.choice(len(self.all_data))
        audio = self.all_data[idx]
        center_time = np.random.randint(self.context_window_radius,audio.shape[-1]-self.context_window_radius)
        return audio[:,(center_time-self.context_window_radius):(center_time+self.context_window_radius)], 0

class DeterministicNoiseDataset(Dataset):
    def load_settings(self, settings):
        self.context_window_radius = settings["context_window_radius"]

    def __init__(self, old_data, sr):

        self.all_data = []
        for _ , row in old_data.iterrows():
            audio_path = row["audio_path"]
            audio = load_audio_file(audio_path, sr)

            self.all_data.append(audio)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        audio = self.all_data[idx]
        center_time = int(audio.shape[-1]/2)
        return audio[:,(center_time-self.context_window_radius):(center_time+self.context_window_radius)], 0

        
### Combined dataset ###

class CombinedDataset(Dataset):
    def __init__(self, one_click_dataset,noise_file_dataset,noise_std_scale,min_scale_coeff,max_scale_coeff,do_scaling_augmentation,do_noise_augmentation):
        self.one_click_dataset = one_click_dataset
        #self.long_file_dataset = long_file_dataset
        self.noise_file_dataset = noise_file_dataset

        self.noise_std_scale = noise_std_scale
        self.min_scale_coeff = min_scale_coeff
        self.max_scale_coeff = max_scale_coeff

        self.do_scaling_augmentation = do_scaling_augmentation
        self.do_noise_augmentation = do_noise_augmentation

    def augment_audio(self,audio_and_label):
        audio, label = audio_and_label

        audio = audio.copy()
        #print('big before scale',np.max(np.abs(audio)))

        if self.do_scaling_augmentation:
            scale_change = np.random.uniform(self.min_scale_coeff,self.max_scale_coeff)
            audio *= scale_change


        if np.random.uniform() > 0.5 and self.do_noise_augmentation:
            audio += np.random.randn(audio.shape[0],audio.shape[1])*np.max(np.abs(audio))*self.noise_std_scale

        return audio, label

    def __len__(self):
        return  len(self.one_click_dataset) + len(self.noise_file_dataset) #+ len(self.long_file_dataset) 

    def get_audio_and_label(self, idx):

        if idx < len(self.one_click_dataset):
            return self.one_click_dataset[idx]
        idx -= len(self.one_click_dataset)

        #if idx < len(self.long_file_dataset):
        #    return self.long_file_dataset[idx]
        #idx -= len(self.long_file_dataset)

        return self.noise_file_dataset[idx]

    def __getitem__(self, idx):
        return self.augment_audio(self.get_audio_and_label(idx))


