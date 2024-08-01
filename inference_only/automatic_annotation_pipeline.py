# Code written by Martín Rodríguez Muñoz under the supervision of Pratyusha Sharma and Professor Antonio Torralba

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
import numpy as np
import pandas as pd
from click_detector_models import SoundNet, ManyLayerMlp
from click_detector_dataset_no_label import LongFileDataset
from scipy.ndimage import label as get_connected_components
from transformer_model import ViT
from transformer_dataset_no_label import DatasetForTransformer
from phase_1_candidate_identification import phase_1
from phase_2_candidate_revision import phase_2
from phase_3_coda_and_whale_clustering import phase_3

# Phase 1 identifies candidate windows among the entire audio
# Phase 2 classifies candidate windows identified by Phase 1 as either containing a communication click or not

# In order to avoid having the same click in multiple candidate windows the below function eliminates all but the most likely window
# in every connected component of windows above the threshold to send to Phase 2
sr = 22050
def annotate(input_file,output_file,prediction_th,sending_th,
                        path_to_phase_1_soundnet_checkpoint,path_to_phase_1_mlp_checkpoint,
                        path_to_phase_2_soundnet_checkpoint,path_to_phase_2_transformer_checkpoint,path_to_phase_2_linear_checkpoint,path_to_phase_2_coda_checkpoint,path_to_phase_2_whale_checkpoint,
                        store_phase_1_predictions,store_all_phase_1_confidences,store_phase_2_output,print_p1_output,print_p2_output):



    

    print("Loading",input_file," duration ", librosa.get_duration(filename=input_file),"seconds")
    full_audio, _ = librosa.load(input_file,mono=False,sr=sr)

    if len(full_audio.shape) != 2 or full_audio.shape[0] == 1:
        print("Error:",input_file,"has only 1 channel")
        return
    
    if full_audio.shape[0] > 2:
        print("Warning: Too many audio channels, ignoring all channels past the first two")
        full_audio = full_audio[:2,:]

    
    click_candidates = phase_1(full_audio,output_file,sending_th,
                        path_to_phase_1_soundnet_checkpoint,path_to_phase_1_mlp_checkpoint,
                        store_phase_1_predictions,store_all_phase_1_confidences,print_p1_output)

    click_info_json = phase_2(click_candidates,full_audio,output_file,path_to_phase_2_soundnet_checkpoint,path_to_phase_2_transformer_checkpoint,path_to_phase_2_linear_checkpoint,path_to_phase_2_coda_checkpoint,path_to_phase_2_whale_checkpoint,store_phase_2_output,print_p2_output,prediction_th)

    final_results = phase_3(click_info_json,prediction_th,input_file)
    final_results.to_csv(output_file,index=False)
    print("Done")