print("Starting")
import numpy as np

batch_size = 8
num_windows_to_send = 200
max_context_time = 30
num_windows_in_context = int(np.ceil(max_context_time*22050)/40)+2
sending_th = 0.7
input_embedding_size = 4096

sending_th = 0.7 # Confidence required to send prediction to transformer
pred_th = 0.5

import json

all_audio_files = {}

import os
from torch.utils.data import Dataset, DataLoader
from candidate_revision_models import SoundNet, ViT
from candidate_revision_dataset_inference_with_label import DatasetForTransformer
import torch
import torch.nn as nn
import pandas as pd
import librosa

def get_audio(audio_path):
    if audio_path in all_audio_files.keys():
        return all_audio_files[audio_path]
    all_audio_files[audio_path], sr = librosa.load(audio_path,mono=False)
    return all_audio_files[audio_path]


checkpoint_folder = "transformer_training_output/"
soft = torch.nn.Softmax(dim=1)

for partition_name in ["val","test"]:
    paths_df = pd.read_csv("dataset/p2_"+partition_name+"_dataset.csv")
    for file_id in range(len(paths_df)):
        print(partition_name,file_id+1,"out of",len(paths_df))
        audio_path = paths_df["audio_path"][file_id]


        dataframe_path = "transformer_dataset/" + partition_name + "/"+ paths_df["file_name"][file_id]+"_"+str(paths_df["part"][file_id]) + ".csv"

        output_folder_csv =  "transformer_inference_csv/" + partition_name +"/"
        if not os.path.exists(output_folder_csv):
            os.mkdir(output_folder_csv)
        output_path_csv = output_folder_csv + paths_df["file_name"][file_id]+"_"+str(paths_df["part"][file_id]) + ".csv"


        output_folder_json =  "transformer_inference_json/" + partition_name + "/"
        if not os.path.exists(output_folder_json):
            os.mkdir(output_folder_json)
        output_path_json = output_folder_json + paths_df["file_name"][file_id]+"_"+str(paths_df["part"][file_id]) + ".json"

        all_annotations_data_frame = pd.read_csv(dataframe_path)
        annotations_data_frame = all_annotations_data_frame[all_annotations_data_frame["supressed_predictions"] >= 0.7]
        annotations_data_frame.reset_index()
        
        audio = get_audio(audio_path)
        transformer_dataset = DatasetForTransformer(num_windows_to_send, max_context_time,sending_th)
        transformer_dataset.load_file(annotations_data_frame,audio)        

        dataloader = DataLoader(transformer_dataset,batch_size=batch_size,shuffle=False, num_workers=20)

        model_soundnet = SoundNet().cuda()
        model_trans = ViT(unit_size=input_embedding_size, dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head = 64, dropout = 0.1, emb_dropout = 0.1, max_len = num_windows_to_send+num_windows_in_context).cuda()
        model_linear = nn.Linear(1024,2).cuda()
        model_coda = nn.Linear(1024,2).cuda()
        model_whale = nn.Linear(1024,2).cuda()

        checkpoint_soundnet = torch.load(checkpoint_folder+"best_soundnet.pt")
        model_soundnet.load_state_dict(checkpoint_soundnet["model_state_dict"])

        checkpoint_transformer = torch.load(checkpoint_folder+"best_transformer.pt")
        model_trans.load_state_dict(checkpoint_transformer["model_state_dict"])

        checkpoint_linear = torch.load(checkpoint_folder+"best_linear.pt")
        model_linear.load_state_dict(checkpoint_linear["model_state_dict"])

        checkpoint_coda = torch.load(checkpoint_folder+"best_coda.pt")
        model_coda.load_state_dict(checkpoint_coda["model_state_dict"])

        checkpoint_whale = torch.load(checkpoint_folder+"best_whale.pt")
        model_whale.load_state_dict(checkpoint_whale["model_state_dict"])

        model_soundnet.eval()
        model_trans.eval()
        model_linear.eval()
        model_coda.eval()
        model_whale.eval()


        all_output = []
        all_time = []
        all_label = []

        all_click_info = {}

        all_same_different_coda_targets = []
        all_same_different_coda_probs = []

        all_same_different_whale_targets = []
        all_same_different_whale_probs = []

        all_same_different_times = []


        for sample_batched in dataloader:
            with torch.no_grad():
                transformer_audio = sample_batched[0].type(torch.cuda.FloatTensor)
                transformer_position_mask = sample_batched[1].type(torch.cuda.BoolTensor)
                transformer_time = sample_batched[2].type(torch.cuda.FloatTensor)
                transformer_target = sample_batched[3].type(torch.cuda.LongTensor)
                transformer_center_mask = sample_batched[4].type(torch.cuda.BoolTensor)
                transformer_same_different_coda_target = sample_batched[5].type(torch.cuda.LongTensor)
                transformer_same_different_whale_target = sample_batched[6].type(torch.cuda.LongTensor)

                not_center_mask = ~transformer_center_mask

                num_batch = transformer_audio.shape[0]
                transformer_input = model_soundnet(transformer_audio.reshape(-1,2,1000)).reshape(num_batch,-1,input_embedding_size)

                embeddings = model_trans(transformer_input,transformer_position_mask)

                click_logit = model_linear(embeddings[transformer_center_mask])
                center_prob = soft(click_logit)[:,1]
                click_target = transformer_target[transformer_center_mask]
                center_time = transformer_time[transformer_center_mask]

                
                not_center_mask_center_real = not_center_mask*click_target.reshape(-1,1) > 0.5                

                for i in range(num_batch):
                    
                    same_different_coda_logit = model_coda(embeddings[i,not_center_mask[i,:]])
                    same_different_coda_prob = soft(same_different_coda_logit)[:,1]

                    same_different_whale_logit = model_whale(embeddings[i,not_center_mask[i,:]])
                    same_different_whale_prob = soft(same_different_whale_logit)[:,1]


                    same_different_coda_target = transformer_same_different_coda_target[i,not_center_mask[i,:]]
                    same_different_whale_target = transformer_same_different_whale_target[i,not_center_mask[i,:]]

                    not_center_time = transformer_time[i,not_center_mask[i,:]]

                    all_same_different_coda_probs.append(same_different_coda_prob.tolist())
                    all_same_different_coda_targets.append(same_different_coda_target.tolist())

                    all_same_different_whale_probs.append(same_different_whale_prob.tolist())
                    all_same_different_whale_targets.append(same_different_whale_target.tolist())

                    all_same_different_times.append(not_center_time.tolist())
                    

                #all_input += input_flat[not_padding].tolist()
                all_output += center_prob.tolist()
                all_time += center_time.tolist()
                all_label += click_target.tolist()


        counter = 0

        for i in range(len(all_time)):
            current_info = {}

            if all_time[i] > 0:
                current_info["time"] = all_time[i]
                current_info["click_probability"] = all_output[i]
                current_info["click_label"] = all_label[i]
                current_info["coda_id"] = list(annotations_data_frame["Coda"])[counter]
                current_info["whale_id"] = list(annotations_data_frame["Whale"])[counter]
                counter += 1

                current_info["same_different_coda_probability"] = all_same_different_coda_probs[i]
                current_info["same_different_coda_targets"] = all_same_different_coda_targets[i]

                current_info["same_different_whale_probability"] = all_same_different_whale_probs[i]
                current_info["same_different_whale_targets"] = all_same_different_whale_targets[i]


                current_info["same_different_times"] = all_same_different_times[i]
                all_click_info[all_time[i]] = current_info

        with open(output_path_json, 'w') as f:
            json.dump(all_click_info, f)

        out_df = pd.DataFrame()
        out_df["window_center_seconds"] = all_time
        out_df["transformer_confidence"] = all_output
        out_df["is_correct"] = all_label
        out_df["Coda"] = list(annotations_data_frame["Coda"])
        out_df["Whale"] = list(annotations_data_frame["Whale"])
        out_df.to_csv(output_path_csv,index=False)