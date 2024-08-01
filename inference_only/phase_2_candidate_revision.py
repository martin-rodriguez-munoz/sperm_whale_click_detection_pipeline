import numpy as np

batch_size = 8

num_windows_to_send = 200
max_context_time = 30
num_windows_in_context = int(np.ceil(max_context_time*22050)/40)+2
input_embedding_size = 4096


import json

all_audio_files = {}

import os
from torch.utils.data import Dataset, DataLoader
from transformer_model import ViT
from click_detector_models import SoundNet
from transformer_dataset_no_label import DatasetForTransformer
import torch
import torch.nn as nn
import pandas as pd

import torch.optim as optim

import librosa


def phase_2(click_candidates_df,audio,output_file,
            path_to_phase_2_soundnet_checkpoint,path_to_phase_2_transformer_checkpoint,path_to_phase_2_linear_checkpoint,path_to_phase_2_coda_checkpoint,path_to_phase_2_whale_checkpoint
            ,store_phase_2_output,print_p2_output,pred_th):

        print("Running phase 2")
        transformer_dataset = DatasetForTransformer(num_windows_to_send, max_context_time)
        transformer_dataset.load_file(click_candidates_df,audio)        

        dataloader = DataLoader(transformer_dataset,batch_size=batch_size,shuffle=False, num_workers=20)

        model_soundnet = SoundNet().cuda()
        model_trans = ViT(unit_size=input_embedding_size, dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head = 64, dropout = 0.1, emb_dropout = 0.1, max_len = num_windows_to_send+num_windows_in_context).cuda()
        model_linear = nn.Linear(1024,2).cuda()
        model_coda = nn.Linear(1024,2).cuda()
        model_whale = nn.Linear(1024,2).cuda()

        checkpoint_soundnet = torch.load(path_to_phase_2_soundnet_checkpoint)
        model_soundnet.load_state_dict(checkpoint_soundnet["model_state_dict"])

        checkpoint_transformer = torch.load(path_to_phase_2_transformer_checkpoint)
        model_trans.load_state_dict(checkpoint_transformer["model_state_dict"])

        checkpoint_linear = torch.load(path_to_phase_2_linear_checkpoint)
        model_linear.load_state_dict(checkpoint_linear["model_state_dict"])

        checkpoint_coda = torch.load(path_to_phase_2_coda_checkpoint)
        model_coda.load_state_dict(checkpoint_coda["model_state_dict"])

        checkpoint_whale = torch.load(path_to_phase_2_whale_checkpoint)
        model_whale.load_state_dict(checkpoint_whale["model_state_dict"])

        model_soundnet.eval()
        model_trans.eval()
        model_linear.eval()
        model_coda.eval()
        model_whale.eval()


        all_output = []
        all_time = []

        all_click_info = {}

        all_same_different_coda_probs = []
        all_same_different_whale_probs = []

        all_same_different_times = []
        soft = torch.nn.Softmax(dim=1)

        for sample_batched in dataloader:
            with torch.no_grad():
                transformer_audio = sample_batched[0].type(torch.cuda.FloatTensor)
                transformer_position_mask = sample_batched[1].type(torch.cuda.BoolTensor)
                transformer_time = sample_batched[2].type(torch.cuda.FloatTensor)
                transformer_center_mask = sample_batched[3].type(torch.cuda.BoolTensor)

                not_center_mask = ~transformer_center_mask

                num_batch = transformer_audio.shape[0]
                transformer_input = model_soundnet(transformer_audio.reshape(-1,2,1000)).reshape(num_batch,-1,input_embedding_size)

                embeddings = model_trans(transformer_input,transformer_position_mask)

                click_logit = model_linear(embeddings[transformer_center_mask])
                center_prob = soft(click_logit)[:,1]
                center_time = transformer_time[transformer_center_mask]

                
                for i in range(num_batch):
                    
                    same_different_coda_logit = model_coda(embeddings[i,not_center_mask[i,:]])
                    same_different_coda_prob = soft(same_different_coda_logit)[:,1]

                    same_different_whale_logit = model_whale(embeddings[i,not_center_mask[i,:]])
                    same_different_whale_prob = soft(same_different_whale_logit)[:,1]

                    not_center_time = transformer_time[i,not_center_mask[i,:]]

                    all_same_different_coda_probs.append(same_different_coda_prob.tolist())
                    all_same_different_whale_probs.append(same_different_whale_prob.tolist())

                    all_same_different_times.append(not_center_time.tolist())


                #all_input += input_flat[not_padding].tolist()
                all_output += center_prob.tolist()
                all_time += center_time.tolist()

                if print_p2_output:
                    for t,c in zip(all_time[-num_batch:],all_output[-num_batch:]):
                        if c > pred_th:
                            print("time",t,"phase 2 confidence",c)




        for i in range(len(all_time)):
            current_info = {}

            if all_time[i] > 0:
                current_info["time"] = all_time[i]
                current_info["click_probability"] = all_output[i]
    

                current_info["same_different_coda_probability"] = all_same_different_coda_probs[i]
                current_info["same_different_whale_probability"] = all_same_different_whale_probs[i]
                current_info["same_different_times"] = all_same_different_times[i]
                all_click_info[all_time[i]] = current_info

        if store_phase_2_output:
            with open(output_file[:-4]+"_all_phase_2_output.json", 'w') as f:
                json.dump(all_click_info, f)
        
        return all_click_info