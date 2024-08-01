import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
import numpy as np
import pandas as pd
from click_detector_models import SoundNet, ManyLayerMlp
from click_detector_dataset_no_label import LongFileDataset
from scipy.ndimage import label as get_connected_components

# Phase 1 identifies candidate windows among the entire audio
# Phase 2 classifies candidate windows identified by Phase 1 as either containing a communication click or not

# In order to avoid having the same click in multiple candidate windows the below function eliminates all but the most likely window
# in every connected component of windows above the threshold to send to Phase 2

def supress_multiple_pred(data_df,sending_th):
    model_conf = data_df["model_confidence"]

    # Identify connected components of predictions above the threshold
    index_range = np.arange(len(model_conf))
    above_th = np.ones(len(model_conf))*(model_conf>sending_th)
    components, num_components = get_connected_components(above_th)

    # Keep only the highest probability window of each connected component
    pred_center = [(model_conf[components == i+1]).idxmax() for i in range(num_components)]
    best_prob = [np.max(model_conf[components == i+1]) for i in range(num_components)]
    adjusted_pred = np.zeros_like(index_range).astype(float)
    for pred_loc, pred_prob in zip(pred_center,best_prob):
        adjusted_pred[pred_loc] = pred_prob

    return adjusted_pred

def phase_1(full_audio,output_file,sending_th,
                        path_to_phase_1_soundnet_checkpoint,path_to_phase_1_mlp_checkpoint,
                        store_phase_1_predictions,store_all_phase_1_confidences,print_output):

    # Click detector settings
    sr = 22050
    context_radius = 500
    center_radius = 20
    click_batch_size = 128

    soft = torch.nn.Softmax(dim=1)
    batch_range = torch.arange(0,click_batch_size)

    click_detector_dataset = LongFileDataset(full_audio, context_radius, center_radius)
    click_detector_dataloader = DataLoader(click_detector_dataset,batch_size=click_batch_size,shuffle=False, num_workers=20)

    print("Loading phase 1 models")
    model_soundnet = SoundNet().cuda()
    model_mlp = ManyLayerMlp(4096).cuda()

    checkpoint_soundnet = torch.load(path_to_phase_1_soundnet_checkpoint)
    model_soundnet.load_state_dict(checkpoint_soundnet["model_state_dict"])

    checkpoint_mlp = torch.load(path_to_phase_1_mlp_checkpoint)
    model_mlp.load_state_dict(checkpoint_mlp["model_state_dict"])

    model_soundnet.eval()
    model_mlp.eval()

    print("Running phase 1")

    out_probs = []

    out_context_window_start = []
    out_window_center = []
    out_context_window_end = []

    out_center_window_start = []
    out_center_window_end = []

    with torch.no_grad():
        context_window_start_of_batch = 0
        for i_batch, audio in enumerate(click_detector_dataloader):
            audio = audio.type(torch.cuda.FloatTensor)
            cur_batch_size = audio.shape[0]

            embeddings = model_soundnet(audio)
            out = model_mlp(embeddings)

            probs = soft(out)[:,1]

            all_context_window_start = context_window_start_of_batch+batch_range[:cur_batch_size]*2*center_radius
            all_window_center = all_context_window_start + context_radius
            all_context_window_end = all_window_center + context_radius

            all_center_window_start = all_window_center - center_radius
            all_center_window_end = all_window_center - center_radius

            out_probs += probs.tolist()

            out_context_window_start += all_context_window_start.tolist()
            out_window_center += all_window_center.tolist()
            out_context_window_end += all_context_window_end.tolist()

            out_center_window_start += all_center_window_start.tolist()
            out_center_window_end += all_center_window_end.tolist()

            context_window_start_of_batch += click_batch_size*2*center_radius

            if print_output:
                for t,c in zip(out_window_center[-cur_batch_size:],out_probs[-cur_batch_size:]):
                    if c > sending_th:
                        print("time",t/sr,"phase 1 confidence",c)


    out_df = pd.DataFrame()
    out_df["model_confidence"] = out_probs

    out_df["context_window_start_seconds"] = np.array(out_context_window_start)/sr
    out_df["window_center_seconds"] = np.array(out_window_center)/sr
    out_df["context_window_end_seconds"] = np.array(out_context_window_end)/sr

    out_df["center_window_seconds"] = np.array(out_center_window_start)/sr
    out_df["center_window_seconds"] = np.array(out_center_window_end)/sr

    out_df["context_window_start_frames"] = out_context_window_start
    out_df["window_center_frames"] = out_window_center
    out_df["context_window_end_frames"] = out_context_window_end

    out_df["center_window_frames"] = out_center_window_start
    out_df["center_window_frames"] = out_center_window_end

    out_df["supressed_predictions"] = supress_multiple_pred(out_df,sending_th)

    if store_all_phase_1_confidences:
        out_df.to_csv(output_file[:-4]+"_all_phase_1_confidences.csv")

    prediction_df = out_df[out_df["supressed_predictions"] >= sending_th]
    prediction_df = prediction_df.reset_index()
    prediction_df = prediction_df.drop("index",axis=1)
    if store_phase_1_predictions:
        prediction_df.to_csv(output_file[:-4]+"_phase_1_predictions.csv")

    return prediction_df