import numpy as np 
from torch.utils.data import Dataset

class LongFileDataset(Dataset):

    def __init__(self, audio, context_radius, center_radius): 
        self.full_audio = audio
        self.context_radius = context_radius
        self.center_radius = center_radius

        self.moments = []

        for window_center in range(self.context_radius,self.full_audio.shape[1]-self.context_radius+1,2*self.center_radius):
            self.moments.append(window_center)
            

    def __len__(self):
        return len(self.moments)

    
    def __getitem__(self, idx):
        window_center = self.moments[idx]
        
        # Remove the mean of a window bigger than the context window
        norm_window_start = max(0,window_center-5020)
        norm_window_end = min(self.full_audio.shape[-1],window_center+5020)
        norm_val = np.mean(self.full_audio[:,norm_window_start:norm_window_end])

        context_window_start = int(window_center-self.context_radius)
        context_window_end   = int(window_center+self.context_radius)

        return self.full_audio[:,context_window_start:context_window_end]-norm_val