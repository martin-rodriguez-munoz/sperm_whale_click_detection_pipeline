import numpy as np 
from torch.utils.data import Dataset
embedding_size = 4096

class DatasetForTransformer(Dataset):

    def one_file_to_samples(self,data_df, full_audio):
      
      #predictions_df = data_df[data_df["supressed_predictions"] >= self.sending_th]
      predictions_df = data_df #data_df[data_df["supressed_predictions"] >= self.sending_th]

      file_group_audio = []
      file_group_position_mask = []
      file_group_time_sec = []
      file_group_label = []
      file_group_center_mask = []
      file_group_same_coda = []
      file_group_same_whale = []

      for index , center_row in predictions_df.iterrows():
        center_time = center_row["window_center_seconds"]
        close_predictions_df = (predictions_df[np.abs(predictions_df["window_center_seconds"] - center_time) < self.max_context_time/2]).reset_index()

        neg_dist_to_center = -np.abs(np.array(close_predictions_df["window_center_seconds"])-center_time)
        num_predictions_in_group = min(len(close_predictions_df),self.num_windows_to_send)

        closest_predictions = sorted(np.argpartition(neg_dist_to_center, -num_predictions_in_group)[-num_predictions_in_group:])
        group_predictions = close_predictions_df.loc[closest_predictions]
        
        group_audio =  np.zeros((self.num_windows_to_send,2,1000)) # Audio of windows
        group_position_mask = np.full(self.window_size_with_padding, False, dtype=bool) # Mask with window positions in time
        group_time_sec = np.zeros(self.num_windows_to_send)  # Time of windows in second
        group_label = np.zeros(self.num_windows_to_send) # Indicates if windows contain clicks
        group_center_mask = np.full(self.num_windows_to_send, False, dtype=bool) # Indicates which window is the center window
        group_same_coda = np.zeros(self.num_windows_to_send) # Indicates if there is a click in the window that belongs to the same coda as the center click
        group_same_whale = np.zeros(self.num_windows_to_send) # Indicates if there is a click in the window said by the same whale as the center click

        chosen_pos = 0
        center_pos = int(np.ceil(self.max_context_time*22050)/80)+1
        for _ , candidate_row in group_predictions.iterrows():
          candidate_time = candidate_row["window_center_seconds"]

          audio_start = int(candidate_time*22050)-500
          audio_end = int(candidate_time*22050)+500

          norm_window_start = max(0,audio_start+500-5020)
          norm_window_end = min(full_audio.shape[-1],audio_end-500+5020)
          norm_val = np.mean(full_audio[:,norm_window_start:norm_window_end])
          
          cropped_audio = full_audio[:,audio_start:audio_end].copy()
          group_audio[chosen_pos] = cropped_audio - norm_val

          group_time_sec[chosen_pos] = candidate_time
          group_center_mask[chosen_pos] = center_time == candidate_time

          global_pos = int(np.round((((candidate_time-center_time)*22050)/40)))+center_pos
          group_position_mask[global_pos] = True

          group_label[chosen_pos] = candidate_row["is_correct"]
          group_same_coda[chosen_pos] = 1*(candidate_row["Coda"] == center_row["Coda"])
          group_same_whale[chosen_pos] = 1*(candidate_row["Whale"] == center_row["Whale"])

          chosen_pos += 1
        
        zero_pad_locations = self.window_size_with_padding - 1 - np.arange(self.num_windows_to_send-chosen_pos)
        group_position_mask[zero_pad_locations] = True

        file_group_audio.append(group_audio)
        file_group_position_mask.append(group_position_mask)
        file_group_time_sec.append(group_time_sec)
        file_group_label.append(group_label)
        file_group_center_mask.append(group_center_mask)
        file_group_same_coda.append(group_same_coda)
        file_group_same_whale.append(group_same_whale)

      return file_group_audio, file_group_position_mask, file_group_time_sec, file_group_label, file_group_center_mask, file_group_same_coda, file_group_same_whale
    


    def __init__(self, num_windows_to_send, max_context_time,sending_th):
        self.num_windows_to_send = num_windows_to_send
        self.max_context_time = max_context_time
        self.num_windows_in_context = int(np.ceil(max_context_time*22050)/40)+2

        self.window_size_with_padding = num_windows_to_send + self.num_windows_in_context
        self.sending_th = sending_th

        self.all_audio = [] # Each element of this list stores the audio windows of an entire group
        self.all_position_masks = [] # Each element of this list stores the relative position of the windows of an entire group
        self.all_time_sec = [] # Each element of this list stores the position in time of the windows of an entire group
        self.all_label = []
        self.all_center_mask = []
        self.all_same_coda = []
        self.all_same_whale = []
    
    def load_file(self, data_df, full_audio):

        file_group_audio, file_group_position_mask, file_group_time_sec, file_group_label, file_group_center_mask, file_same_coda, file_same_whale = self.one_file_to_samples(data_df,full_audio)

        self.all_audio += file_group_audio
        self.all_position_masks += file_group_position_mask
        self.all_time_sec += file_group_time_sec
        self.all_label += file_group_label
        self.all_center_mask += file_group_center_mask
        self.all_same_coda += file_same_coda
        self.all_same_whale += file_same_whale


    def __len__(self):
      return len(self.all_audio)


    def __getitem__(self, idx):
      return self.all_audio[idx], self.all_position_masks[idx], self.all_time_sec[idx], self.all_label[idx], self.all_center_mask[idx], self.all_same_coda[idx], self.all_same_whale[idx]