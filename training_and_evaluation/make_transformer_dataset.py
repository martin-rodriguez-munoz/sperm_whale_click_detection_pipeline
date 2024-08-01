print('Starting...')
sr = 22050
debug = False

context_radius = 500
center_radius = 20

batch_size = 128

import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import librosa
import numpy as np
from click_candidate_detector_model import SoundNet, ManyLayerMlp
import pandas as pd
from click_candidate_detector_dataset_with_label import LongFileDataset
from scipy.ndimage import label as get_connected_components

criterion = nn.CrossEntropyLoss()   
soft = torch.nn.Softmax(dim=1)

def add_decimals(n):
    s = str(n)
    if n < 10:
        return "00"+s
    return "0"+s

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

def supress_multiple_pred(data_df,sending_th):
    model_conf = data_df["model_confidence"]

    index_range = np.arange(len(model_conf))
    above_th = np.ones(len(model_conf))*(model_conf>sending_th)

    components, num_components = get_connected_components(above_th)
    pred_center = [(model_conf[components == i+1]).idxmax() for i in range(num_components)]
    best_prob = [np.max(model_conf[components == i+1]) for i in range(num_components)]

    adjusted_pred = np.zeros_like(index_range).astype(float)
    
    for pred_loc, pred_prob in zip(pred_center,best_prob):
        adjusted_pred[pred_loc] = pred_prob
    return adjusted_pred


def get_TP_FP_FN(pred_list, gt_list, time_diff_threshold = 0.0045351):
    """
    Returns the number of true positives, false positives and false negatives
    """
    TP = 0
    FP = 0
    FN = 0



    # Here we keep track of: The id of the closest prediction to each ground truth coda, the difference in start times and, the different in end times
    gt_closest_pred_start_dist_end_dist = -np.ones((len(gt_list),2))

    # Here we keep track of: The id of the closest ground truth to each prediction coda, the difference in start times and, the different in end times
    pred_closest_gt_start_dist_end_dist = -np.ones((len(pred_list),2))

    # In this loop we calculate these values
    # An ID of -1 just means it hasn't been initialized yet
    gt_id = 0
    is_good = np.zeros_like(pred_list)
    is_caught = np.zeros_like(gt_list)

    for gt_click_time in gt_list:

        
        pred_id = 0
        for pred_click_time in pred_list:
            time_diff = np.abs(gt_click_time-pred_click_time)

            if  gt_closest_pred_start_dist_end_dist[gt_id][0] == -1 or time_diff < gt_closest_pred_start_dist_end_dist[gt_id][1]:
                gt_closest_pred_start_dist_end_dist[gt_id][0] = pred_id
                gt_closest_pred_start_dist_end_dist[gt_id][1] = time_diff

            if pred_closest_gt_start_dist_end_dist[pred_id][0] == -1 or time_diff < pred_closest_gt_start_dist_end_dist[pred_id][1]:
                pred_closest_gt_start_dist_end_dist[pred_id][0] = gt_id
                pred_closest_gt_start_dist_end_dist[pred_id][1] = time_diff

            pred_id += 1

        gt_id += 1


    # For every value in the ground truth if we found a match that's a true positive otherwise it's a false negative.
    for gt_id in range(len(gt_list)):
        if gt_closest_pred_start_dist_end_dist[gt_id][0] == -1:
            #print(gt_list[gt_id])
            FN += 1
            continue


        if pred_closest_gt_start_dist_end_dist[int(gt_closest_pred_start_dist_end_dist[gt_id][0])][0] == gt_id and gt_closest_pred_start_dist_end_dist[gt_id][1] < time_diff_threshold:
            is_good[int(gt_closest_pred_start_dist_end_dist[gt_id][0])] = 1
            is_caught[gt_id] = 1
            TP += 1
        else:
            #print(gt_list[gt_id])
            FN += 1

    # For every value in the prediction if we found a match that's a true positive otherwise it's a false positive.
    for pred_id in range(len(pred_list)):
        if pred_closest_gt_start_dist_end_dist[pred_id][0] == -1:
            FP += 1
            continue

        if gt_closest_pred_start_dist_end_dist[int(pred_closest_gt_start_dist_end_dist[pred_id][0])][0] == pred_id and pred_closest_gt_start_dist_end_dist[pred_id][1] < time_diff_threshold:
            continue
            # True positives have already been counted in the previous loop, don't count them again.
        else:
            FP += 1
    

    return is_good, is_caught, TP, FP, FN

batch_range = torch.arange(0,batch_size)

model_path = "phase_1_checkpoints/"

def load_model(checkpoint_folder):
    model_soundnet = SoundNet().cuda()
    model_mlp = ManyLayerMlp(4096).cuda()

    checkpoint_soundnet = torch.load(checkpoint_folder + "phase_1_soundnet.pt")
    model_soundnet.load_state_dict(checkpoint_soundnet["model_state_dict"])

    checkpoint_mlp = torch.load(checkpoint_folder + "phase_1_mlp.pt")
    model_mlp.load_state_dict(checkpoint_mlp["model_state_dict"])

    model_soundnet.eval()
    model_mlp.eval()

    return model_soundnet, model_mlp

model_soundnet, model_mlp = load_model(model_path)

def make_dataset(partition_name,partition_path,sending_th,original_gt_df):
    partition_df = pd.read_csv(partition_path)
    clicks_gt = get_clicks_gt(partition_df,original_gt_df)
    k = 0
    for _ , file_info in partition_df.iterrows():
        k += 1

        output_folder = "transformer_dataset/"+partition_name+"/"
        out_name = file_info["file_name"]+"_"+str(file_info["part"])+".csv"

        print(partition_name,k,"out of",len(partition_df))


        start_frame = int(file_info["first_context_start_frame"])
        last_start_frame = int(file_info["last_context_start_frame"])

        num_windows = int( (last_start_frame-start_frame) / (2*center_radius)) + 1

        end_frame = last_start_frame + 2*context_radius

        part_clicks_gt = [a for a in clicks_gt[file_info["file_name"]] if a[0] >= start_frame/22050 and a[0] <= end_frame/22050]
        click_list_sec = [a[0] for a in part_clicks_gt]
        coda_list_original = [a[1] for a in part_clicks_gt]
        whale_list_original = [a[2] for a in part_clicks_gt]

        sr = 22050
        click_list_frames = np.array(click_list_sec) * sr - start_frame # This gives us position of clicks in the section of the audio we are analyzing

        print("Loading",file_info["audio_path"])
        full_audio, sr = librosa.load(file_info["audio_path"],mono=False)
        
        print("Loading finished")

        cropped_audio = full_audio[:,start_frame:end_frame]
        dataset =  LongFileDataset(cropped_audio, context_radius, center_radius, click_list_frames)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False, num_workers=20)

        out_probs = []
        out_label = []

        out_context_window_start = []
        out_window_center = []
        out_context_window_end = []

        out_center_window_start = []
        out_center_window_end = []

        #all_embeddings = np.zeros((num_windows,4096))

        with torch.no_grad():
            context_window_start_of_batch = start_frame
            for i_batch, sample_batched in enumerate(dataloader):
                audio = sample_batched[0].type(torch.cuda.FloatTensor)
                cur_batch_size = audio.shape[0]
                label = sample_batched[1].type(torch.cuda.LongTensor)

                embeddings = model_soundnet(audio)
                out = model_mlp(embeddings)

                probs = soft(out)[:,1]

                all_context_window_start = context_window_start_of_batch+batch_range[:cur_batch_size]*2*center_radius
                all_window_center = all_context_window_start + context_radius
                all_context_window_end = all_window_center + context_radius

                all_center_window_start = all_window_center - center_radius
                all_center_window_end = all_window_center - center_radius

                #embeddings_np = embeddings.cpu().numpy()
                #all_embeddings[i_batch*batch_size:i_batch*batch_size + cur_batch_size] = embeddings_np

                out_probs += probs.tolist()
        
                out_context_window_start += all_context_window_start.tolist()
                out_window_center += all_window_center.tolist()
                out_context_window_end += all_context_window_end.tolist()

                out_center_window_start += all_center_window_start.tolist()
                out_center_window_end += all_center_window_end.tolist()

                out_label += label.tolist()

                context_window_start_of_batch += batch_size*2*center_radius

        out_df = pd.DataFrame()
        out_df["model_confidence"] = out_probs
        out_df["context_window_start_seconds"] = np.array(out_context_window_start)/sr
        out_df["window_center_seconds"] = np.array(out_window_center)/sr
        out_df["context_window_end_seconds"] = np.array(out_context_window_end)/sr

        out_df["center_window_seconds_start"] = np.array(out_center_window_start)/sr
        out_df["center_window_seconds_end"] = np.array(out_center_window_end)/sr


        out_df["context_window_start_frames"] = out_context_window_start
        out_df["window_center_frames"] = out_window_center
        out_df["context_window_end_frames"] = out_context_window_end

        out_df["center_window_frames_start"] = out_center_window_start
        out_df["center_window_frames_end"] = out_center_window_end

        out_df["label"] = out_label
        
        preds = supress_multiple_pred(out_df,sending_th)
        labels = out_df["label"]

        pred_times = np.array(out_df["window_center_seconds"])[preds >= sending_th]
        gt_times = np.array(out_df["window_center_seconds"])[labels == 1]

        is_good, is_caught, TP, FP, FN = get_TP_FP_FN(pred_times,gt_times)

        print("Phase 1 performance at sending threshold:","TP",TP,"FP",FP,"FN",FN)
        pred_id = 0

        is_a_good_prediction = []
        out_df["supressed_predictions"] = preds
        for _ , row in out_df.iterrows():
            if row["supressed_predictions"] >= sending_th:
                is_a_good_prediction.append(is_good[pred_id])
                pred_id += 1
            else:
                is_a_good_prediction.append(0)
        out_df["is_correct"] = is_a_good_prediction

        gt_id = 0
        is_a_good_label = []
        for _ , row in out_df.iterrows():
            if row["label"] == 1:
                is_a_good_label.append(is_caught[gt_id])
                gt_id += 1
            else:
                is_a_good_label.append(0)
        out_df["is_caught"] = is_a_good_label

        out_name = file_info["file_name"]+"_"+str(file_info["part"])

        coda_id_list = []
        whale_id_list = []
        click_list_sec_np = np.array(click_list_sec)
        for _ , row in out_df.iterrows():
            if row["is_correct"] == 1:# or (row["label"] == 1 and row["is_caught"] == 0):
                center_time = row["window_center_seconds"]
                click_pos = np.argmin(np.abs(center_time-click_list_sec_np))
                coda_id_list.append(coda_list_original[click_pos])
                whale_id_list.append(whale_list_original[click_pos])
            else:
                coda_id_list.append(-1)
                whale_id_list.append(-1)
        
        out_df["Coda"] = coda_id_list
        out_df["Whale"] = whale_id_list
        out_df.to_csv(output_folder+"/"+out_name+".csv")

original_gt_df = pd.read_csv("dataset/p2_all_annotations.csv")

sending_th = 0.7
partition_name = "train"
partition_path = "dataset/p2_train_dataset.csv"
make_dataset(partition_name,partition_path,sending_th,original_gt_df)

sending_th = 0.7
partition_name = "val"
partition_path = "dataset/p2_val_dataset.csv"
make_dataset(partition_name,partition_path,sending_th,original_gt_df)

sending_th = 0.7
partition_name = "test"
partition_path = "dataset/p2_test_dataset.csv"
make_dataset(partition_name,partition_path,sending_th,original_gt_df)