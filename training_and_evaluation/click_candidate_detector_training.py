print("Starting...")
sr = 22050
output_folder = "phase_1_checkpoints/"

experiment_settings = {}
experiment_settings["context_window_radius"] = 500
experiment_settings["center_window_radius"] = 20

experiment_settings["num_epochs"] = 100
experiment_settings["max_offset"] = 15
experiment_settings["batch_size"] = 128
experiment_settings["lr_soundnet"] = 1e-4
experiment_settings["lr_mlp"] = 1e-3

experiment_settings["do_scaling_augmentation"] = True
experiment_settings["do_noise_augmentation"] = True
experiment_settings["noise_std_scale"] = 1/12
experiment_settings["min_scale_coeff"] = 0.9
experiment_settings["max_scale_coeff"] = 1.1

import os
import numpy as np 
import pandas as pd
import pickle
from click_candidate_detector_training_dataset import load_audio_file, get_all_data, LongFileDataset, OneClickFileDataset, RandomNoiseDataset, DeterministicNoiseDataset, CombinedDataset
from click_candidate_detector_evaluation import get_prediction_times, get_TP_FP_FN
from torch.utils.data import DataLoader
from click_candidate_detector_model import SoundNet, ManyLayerMlp

import torch
import torch.optim as optim
import torch.nn as nn

def get_only_train(file_list,train_dataset_info_path):
    train_file_names = np.unique(pd.read_csv(train_dataset_info_path)["file_name"])
    return [v for v in file_list if v[0].split('/')[-1].split('_')[0] in train_file_names]


# Create datasets
train_clicks = pd.read_csv("dataset/p1_train_clicks.csv")
train_noise = pd.read_csv("dataset/p1_train_noise.csv")


print("Loading train clicks")
train_click_dataset = OneClickFileDataset(train_clicks,sr)

print("Loading train noise")
train_noise_dataset = RandomNoiseDataset(train_noise, sr,len(train_click_dataset))

train_click_dataset.load_settings(experiment_settings)
train_noise_dataset.load_settings(experiment_settings)
all_train_dataset = CombinedDataset(train_click_dataset,train_noise_dataset,experiment_settings["noise_std_scale"],experiment_settings["min_scale_coeff"],experiment_settings["max_scale_coeff"],experiment_settings["do_scaling_augmentation"],experiment_settings["do_noise_augmentation"])
train_dataloader = DataLoader(all_train_dataset,batch_size=experiment_settings["batch_size"],shuffle=True, num_workers=20)

def gt_to_click_list(gt_df):
    output = []
    for index, row in gt_df.iterrows():
        click_time = row["TsTo"]
        output.append(click_time)
        for i in range(1,41):
            if row["ICI"+str(i)] == 0:
                break
            click_time += row["ICI"+str(i)]
            output.append(click_time)
    return sorted(output)

val_audio_path = "dataset/"
val_audio_name = "p1_validation_audio.wav"

val_gt_path = "dataset/"
val_gt_name = 'p1_validation_annotations.csv'

val_audio = load_audio_file(val_audio_path+val_audio_name,sr)

val_gt_df = pd.read_csv(val_gt_path+val_gt_name)
val_gt_clicks_sec = np.array(sorted(gt_to_click_list(val_gt_df)))
val_gt_clicks_frames = val_gt_clicks_sec*sr

all_val_data = [(val_audio,val_gt_clicks_frames)]

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

val_dataset = LongFileDataset(all_val_data,False,experiment_settings["context_window_radius"], experiment_settings["center_window_radius"], None)
val_dataloader = DataLoader(val_dataset,batch_size=experiment_settings["batch_size"],shuffle=False, num_workers=20)

criterion = nn.CrossEntropyLoss()   
soft = torch.nn.Softmax(dim=1)

model_soundnet = SoundNet().cuda()

if experiment_settings["context_window_radius"] >= 4335:
    model_mlp = ManyLayerMlp(6144).cuda()
else:
    model_mlp = ManyLayerMlp(4096).cuda()

optimizer_soundnet = optim.Adam(model_soundnet.parameters(), lr=experiment_settings["lr_soundnet"])
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=experiment_settings["lr_mlp"])

best_f_score = -1

avg_train_losses = []
avg_val_losses = []

batch_range = torch.arange(0,experiment_settings["batch_size"])
for epoch in range(experiment_settings["num_epochs"]):
    print('epoch',epoch)

    model_soundnet.train()
    model_mlp.train()

    train_losses = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        optimizer_soundnet.zero_grad()
        optimizer_mlp.zero_grad()
        

        audio = sample_batched[0].type(torch.cuda.FloatTensor)
        label = sample_batched[1].type(torch.cuda.LongTensor)
        
        embeddings = model_soundnet(audio)
        out = model_mlp(embeddings)

        loss = criterion(out,label)
        loss.backward()

        optimizer_soundnet.step()
        optimizer_mlp.step()
        train_losses.append(loss.data.item())

    current_train_loss = np.mean(train_losses)
    avg_train_losses.append(current_train_loss)
    print('average train loss',current_train_loss)

        
    # Validation loop
    model_soundnet.eval()
    model_mlp.eval()

    out_probs = []
    out_label = []
    out_window_center = []

    with torch.no_grad():
        context_window_start_of_batch = 0
        for i_batch, sample_batched in enumerate(val_dataloader):
            audio = sample_batched[0].type(torch.cuda.FloatTensor)
            label = sample_batched[1].type(torch.cuda.LongTensor)

            embeddings = model_soundnet(audio)
            out = model_mlp(embeddings)

            probs = soft(out)[:,1]

            all_window_center = context_window_start_of_batch + experiment_settings["context_window_radius"] + batch_range[:probs.shape[0]]*2*experiment_settings["center_window_radius"]
           
            out_probs += probs.tolist()
            out_window_center += all_window_center.tolist()
            out_label += label.tolist()

            context_window_start_of_batch += experiment_settings["batch_size"]*2*experiment_settings["center_window_radius"]

    pred_times = get_prediction_times(out_probs,out_window_center,0.5)/sr
    real_click_times = np.array(out_window_center)[np.array(out_label).astype(bool)]/sr

    TP, FP, FN = get_TP_FP_FN(pred_times, real_click_times)
    print("val TP",TP)
    print("val FP",FP)
    print("val FN",FN)
    f_score = (2*TP)/(2*TP+FP+FN)
    print("val F1",f_score)


    if best_f_score < f_score:
        best_f_score = f_score
        print('new best f-score',best_f_score)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_soundnet.state_dict(),
                }, output_folder + "phase_1_soundnet.pt")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_mlp.state_dict(),
                }, output_folder + "phase_1_mlp.pt")

            
