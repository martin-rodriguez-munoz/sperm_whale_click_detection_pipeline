import pandas as pd
import numpy as np
import librosa
import os
from scipy.io.wavfile import write

if not os.path.exists("audio_crops"):
    os.mkdir("audio_crops")

if not os.path.exists("audio_crops/clicks"):
    os.mkdir("audio_crops/clicks")

if not os.path.exists("audio_crops/noise"):
    os.mkdir("audio_crops/noise")

def get_clicks_gt(partition_df,original_gt_df):

    click_time_coda_whale = {}
    for audio_name in sorted(np.unique(partition_df["file_name"])):
        click_time_coda_whale[audio_name] = []

    coda_id = 0
    # Get click times across multiple files
    for _ , row in original_gt_df.iterrows():
        coda_id += 1
        current_audio_root_name = row["REC"]
        
        if current_audio_root_name not in click_time_coda_whale.keys():
            continue

        ct = row["TsTo"]
        click_time_coda_whale[current_audio_root_name].append((ct,coda_id,row["Whale"]))

        if row["nClicks"] > 40:
            continue
        
        for click_num in range(1,min(40,row['nClicks'])):
            ct += row["ICI"+str(click_num)]
            click_time_coda_whale[current_audio_root_name].append((ct,coda_id,row["Whale"]))

    for k in click_time_coda_whale.keys():
        click_time_coda_whale[k] = sorted(click_time_coda_whale[k])

    return click_time_coda_whale

train_df = pd.read_csv("dataset/p2_train_dataset.csv")
gt_df = pd.read_csv("dataset/p2_all_annotations.csv")

audio_clicks_coda_whale = get_clicks_gt(train_df,gt_df)

name_to_path = {}
for _ , row in train_df.iterrows():
    name_to_path[row["file_name"]] = row["audio_path"]

audio_and_ct = []
for audio_name in audio_clicks_coda_whale.keys():
    for i in range(len(audio_clicks_coda_whale[audio_name])):
        audio_and_ct.append((name_to_path[audio_name],audio_clicks_coda_whale[audio_name][i][0]))

audio_path_list = []
audios = {}
audio_clicks = {}

for audio_path, ct in audio_and_ct:
    if audio_path not in audio_path_list:
        audio_path_list.append(audio_path)
        audios[audio_path], sr = librosa.load(audio_path,mono=False)
        audio_clicks[audio_path] = []
    audio_clicks[audio_path].append(ct)


output_folder = "audio_crops/"

output_path_list = []
click_time_list = []

idx = 0
for audio_path in audio_path_list:
    for click_time in audio_clicks[audio_path]:
        audio = audios[audio_path]
        ct_frame = int(click_time*22050)
        cropped_audio = audio[:,(ct_frame-5020):(ct_frame+5020)]
        cropped_audio = cropped_audio / np.max(np.abs(cropped_audio))
        cropped_audio *= 32767
        cropped_audio = cropped_audio.astype(np.int16)


        idx += 1
        output_path = output_folder+"clicks/cropped_click_"+str(idx)+".wav"
        output_path_list.append(output_path)
        click_time_list.append(5020)
        
        print("Saving",output_path)
        write(output_path,22050,cropped_audio)

df = pd.DataFrame()
df["audio_path"] = output_path_list
df["click_time"] = click_time_list
df.to_csv("dataset/p1_dataset_click.csv",index=False)


output_path_list = []

for i in range(100*len(click_time_list)):
    audio_path = np.random.choice(audio_path_list)
    audio = audios[audio_path]
    audio_len = audio.shape[1]
    
    center_pos = np.random.randint(5020,audio_len-5020)
    clicks_frame = np.array(audio_clicks[audio_path])*sr
    while np.min(np.abs(center_pos-clicks_frame)) < 60:
        center_pos = np.random.randint(5020,audio_len-5020)
    
    cropped_audio = audio[:,(center_pos-5020):(center_pos+5020)]
    
    output_path = output_folder+"clicks/cropped_noise_"+str(i)+".wav"
    output_path_list.append(output_path)
    print("Saving",output_path)
    write(output_path,22050,cropped_audio)
    

df = pd.DataFrame()
df["audio_path"] = output_path_list
df.to_csv("dataset/p1_dataset_noise.csv",index=False)