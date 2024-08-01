print("Starting")

import os
from torch.utils.data import Dataset, DataLoader
from candidate_revision_models import SoundNet, ViT
from candidate_revision_dataset_train import DatasetForTransformer
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import librosa

all_audio_files = {}
def get_audio(audio_path):
    if audio_path in all_audio_files.keys():
        return all_audio_files[audio_path]
    all_audio_files[audio_path], sr = librosa.load(audio_path,mono=False)
    return all_audio_files[audio_path]

batch_size = 8
num_epochs = 30

num_windows_to_send = 200
max_context_time = 30
num_windows_in_context = int(np.ceil(max_context_time*22050)/40)+2
sending_th = 0.7
do_augmentation = True
input_embedding_size = 4096

sending_th = 0.7 # Confidence required to send prediction to transformer
pred_th = 0.5 # Threshold for calculating metrics during evaluation

lr_soundnet = 1e-6
lr_transformer = 1e-6
lr_linear = 1e-6


output_folder = "transformer_training_output/"

if not os.path.exists(output_folder):
        os.mkdir(output_folder)


train_csv_path = "transformer_dataset/train/"
val_csv_path = "transformer_dataset/val/"
test_csv_path = "transformer_dataset/test/"

train_dataset_path = "dataset/p2_train_dataset.csv"
val_dataset_path = "dataset/p2_val_dataset.csv"

train_dataset_info = pd.read_csv(train_dataset_path)
val_dataset_info = pd.read_csv(val_dataset_path)

print("Loading val")
val_dataset = DatasetForTransformer(num_windows_to_send, max_context_time,sending_th,False)

k = 0
for data_name in os.listdir(val_csv_path):
    k += 1
    print("Loading",k,"out of",len(os.listdir(val_csv_path)),data_name)

    all_annotations_data_frame = pd.read_csv(val_csv_path+data_name)
    annotations_data_frame = all_annotations_data_frame[all_annotations_data_frame["supressed_predictions"] >= 0.7]
    annotations_data_frame.reset_index()

    file_name = data_name.split('_')[0]
    audio_path = list(val_dataset_info[val_dataset_info["file_name"] == file_name]["audio_path"])[0]  #"/data/vision/torralba/scratch/fjacob/martin/all_audio/"+file_name+".wav" 
    audio = get_audio(audio_path)
    val_dataset.load_file(annotations_data_frame,audio)
    


print("Loading train")
train_dataset = DatasetForTransformer(num_windows_to_send, max_context_time,sending_th,True)
k = 0
for data_name in os.listdir(train_csv_path):
    k += 1
    print("Loading",k,"out of",len(os.listdir(train_csv_path)),data_name)

    all_annotations_data_frame = pd.read_csv(train_csv_path+data_name)
    annotations_data_frame = all_annotations_data_frame[all_annotations_data_frame["supressed_predictions"] >= 0.7]
    annotations_data_frame.reset_index()

    file_name = data_name.split('_')[0]
    audio_path =  list(train_dataset_info[train_dataset_info["file_name"] == file_name]["audio_path"])[0] #"/data/vision/torralba/scratch/fjacob/martin/all_audio/"+file_name+".wav" 
    audio = get_audio(audio_path)         
    train_dataset.load_file(annotations_data_frame,audio)

#print("lenghts",len(train_dataset),len(val_dataset))
click_criterion = nn.CrossEntropyLoss()   
coda_criterion = nn.CrossEntropyLoss() 
whale_criterion = nn.CrossEntropyLoss() 
soft = torch.nn.Softmax(dim=1)

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=20)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=20)

model_soundnet = SoundNet().cuda()
model_trans = ViT(unit_size=input_embedding_size, dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head = 64, dropout = 0.1, emb_dropout = 0.1, max_len = num_windows_to_send+num_windows_in_context).cuda()
model_linear = nn.Linear(1024,2).cuda()
model_coda = nn.Linear(1024,2).cuda()
model_whale = nn.Linear(1024,2).cuda()

opt_s = optim.Adam(model_soundnet.parameters(), lr=lr_soundnet)
opt_t = optim.Adam(model_trans.parameters(), lr=lr_transformer)
opt_l = optim.Adam(model_linear.parameters(), lr=lr_linear)
opt_c = optim.Adam(model_coda.parameters(), lr=lr_linear)
opt_w = optim.Adam(model_whale.parameters(), lr=lr_linear)

model_soundnet.train()
model_trans.train()
model_linear.train()
model_coda.train()
model_whale.train()

#best_f_score = -1
best_val_loss = 10000

def calc_metrics(TP,FP,FN,TN):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)
    f_score = (2*TP)/(2*TP+FP+FN)

    print("TP",TP)
    print("FP",FP)
    print("FN",FN)
    print("TN",TN)

    print('accuracy',accuracy)
    print('recall',recall)
    print('precision',precision)
    print('f_score',f_score)

    return accuracy, recall, precision, f_score

all_train_total_loss = []
all_train_coda_loss = []
all_train_whale_loss = []
all_train_click_loss = []

all_val_total_loss = []
all_val_coda_loss = []
all_val_whale_loss = []
all_val_click_loss = []


for epoch in range(num_epochs):
    print('--------------')
    print('num epoch',epoch)

    model_soundnet.train()
    model_trans.train()
    model_linear.train()
    model_coda.train()
    model_whale.train()

    epoch_train_total_loss = []
    epoch_train_coda_loss = []
    epoch_train_whale_loss = []
    epoch_train_click_loss = []

    for sample_batched in train_dataloader:
        opt_t.zero_grad()
        opt_l.zero_grad()
        opt_s.zero_grad()
        opt_c.zero_grad()
        opt_w.zero_grad()

        transformer_audio = sample_batched[0].type(torch.cuda.FloatTensor)
        transformer_position_mask = sample_batched[1].type(torch.cuda.BoolTensor)
        transformer_target = sample_batched[3].type(torch.cuda.LongTensor)
        transformer_center_mask = sample_batched[4].type(torch.cuda.BoolTensor)
        transformer_same_different_coda_target = sample_batched[5].type(torch.cuda.LongTensor)
        transformer_same_different_whale_target = sample_batched[6].type(torch.cuda.LongTensor)

        not_center_mask = ~transformer_center_mask
        

        num_batch = transformer_audio.shape[0]
        transformer_input = model_soundnet(transformer_audio.reshape(-1,2,1000)).reshape(num_batch,-1,input_embedding_size)

        embeddings = model_trans(transformer_input,transformer_position_mask)

        # Click loss is only calculated for center click candidate
        click_logit = model_linear(embeddings[transformer_center_mask])
        click_target = transformer_target[transformer_center_mask]
        click_loss = click_criterion(click_logit,click_target)

        # Coda and whale loss are only calculated for not-central click candidates and only if the central click candidate is a real click
        not_center_mask_center_real = (not_center_mask*transformer_target)*click_target.reshape(-1,1) > 0.5
        
        same_different_coda_logit = model_coda(embeddings[not_center_mask_center_real])
        same_different_coda_target = transformer_same_different_coda_target[not_center_mask_center_real]

        same_different_whale_logit = model_whale(embeddings[not_center_mask_center_real])
        same_different_whale_target = transformer_same_different_whale_target[not_center_mask_center_real]
        
       
        epoch_train_click_loss.append(click_loss.item())


        coda_loss = coda_criterion(same_different_coda_logit,same_different_coda_target)
        whale_loss = whale_criterion(same_different_whale_logit,same_different_whale_target)
        
        loss = click_loss

        if not np.isnan(coda_loss.item()):
            epoch_train_coda_loss.append(coda_loss.item())
            loss += coda_loss

        if not np.isnan(whale_loss.item()):
            epoch_train_whale_loss.append(whale_loss.item())
            loss += whale_loss

        loss.backward()

        opt_s.step()
        opt_t.step()
        opt_l.step()
        opt_c.step()
        opt_w.step()

        epoch_train_total_loss.append(loss.item())

    avg_train_total_loss = np.mean(epoch_train_total_loss)
    print('average train total loss',avg_train_total_loss)
    all_train_total_loss.append(avg_train_total_loss)

    avg_train_coda_loss = np.mean(epoch_train_coda_loss)
    print('average train coda loss',avg_train_coda_loss)
    all_train_coda_loss.append(avg_train_coda_loss)

    avg_train_whale_loss = np.mean(epoch_train_whale_loss)
    print('average train whale loss',avg_train_whale_loss)
    all_train_whale_loss.append(avg_train_whale_loss)

    avg_train_click_loss = np.mean(epoch_train_click_loss)
    print('average train click loss',avg_train_click_loss)
    all_train_click_loss.append(avg_train_click_loss)
    

    model_soundnet.eval()
    model_trans.eval()
    model_linear.eval()
    model_coda.eval()
    model_whale.eval()

    correct = 0
    total = 0

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    epoch_val_total_loss = []
    epoch_val_coda_loss = []
    epoch_val_whale_loss = []
    epoch_val_click_loss = []

    for sample_batched in val_dataloader:
        with torch.no_grad():
            transformer_audio = sample_batched[0].type(torch.cuda.FloatTensor)
            transformer_position_mask = sample_batched[1].type(torch.cuda.BoolTensor)
            transformer_target = sample_batched[3].type(torch.cuda.LongTensor)
            transformer_center_mask = sample_batched[4].type(torch.cuda.BoolTensor)
            transformer_same_different_coda_target = sample_batched[5].type(torch.cuda.LongTensor)
            transformer_same_different_whale_target = sample_batched[6].type(torch.cuda.LongTensor)

            not_center_mask = ~transformer_center_mask
            

            num_batch = transformer_audio.shape[0]
            transformer_input = model_soundnet(transformer_audio.reshape(-1,2,1000)).reshape(num_batch,-1,input_embedding_size)

            embeddings = model_trans(transformer_input,transformer_position_mask)

            click_logit = model_linear(embeddings[transformer_center_mask])
            click_target = transformer_target[transformer_center_mask]
            not_center_mask_center_real = (not_center_mask*transformer_target)*click_target.reshape(-1,1) > 0.5
            
            same_different_coda_logit = model_coda(embeddings[not_center_mask_center_real])
            same_different_coda_target = transformer_same_different_coda_target[not_center_mask_center_real]

            same_different_whale_logit = model_whale(embeddings[not_center_mask_center_real])
            same_different_whale_target = transformer_same_different_whale_target[not_center_mask_center_real]


            center_prob = soft(click_logit)[:,1]

            click_loss = click_criterion(click_logit,click_target)
            epoch_val_click_loss.append(click_loss.item())

            coda_loss = coda_criterion(same_different_coda_logit,same_different_coda_target)
            whale_loss = whale_criterion(same_different_whale_logit,same_different_whale_target)

            loss = click_loss
            if not np.isnan(coda_loss.item()):
                epoch_val_coda_loss.append(coda_loss.item())
                loss += coda_loss

            if not np.isnan(whale_loss.item()):
                epoch_val_whale_loss.append(whale_loss.item())
                loss += whale_loss
    
            epoch_val_total_loss.append(loss.item())
            
            TP += torch.sum(torch.logical_and((click_target > pred_th),(center_prob > pred_th))).item()
            FP += torch.sum(torch.logical_and((click_target <= pred_th),(center_prob > pred_th))).item()
            FN += torch.sum(torch.logical_and((click_target > pred_th),(center_prob <= pred_th))).item()
            TN += torch.sum(torch.logical_and((click_target <= pred_th),(center_prob <= pred_th))).item()


            
    print('val metrics')
    avg_val_total_loss = np.mean(epoch_val_total_loss)
    print('average val total loss',avg_val_total_loss)
    all_val_total_loss.append(avg_val_total_loss)

    avg_val_coda_loss = np.mean(epoch_val_coda_loss)
    print('average val coda loss',avg_val_coda_loss)
    all_val_coda_loss.append(avg_val_coda_loss)

    avg_val_whale_loss = np.mean(epoch_val_whale_loss)
    print('average val whale loss',avg_val_whale_loss)
    all_val_whale_loss.append(avg_val_whale_loss)

    avg_val_click_loss = np.mean(epoch_val_click_loss)
    print('average val click loss',avg_val_click_loss)
    all_val_click_loss.append(avg_val_click_loss)

    accuracy, recall, precision, f_score = calc_metrics(TP,FP,FN,TN)


    if avg_val_total_loss < best_val_loss:
        best_val_loss = avg_val_total_loss
        print('new best val loss',best_val_loss)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_soundnet.state_dict(),
                }, output_folder + "best_soundnet.pt")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_trans.state_dict(),
                }, output_folder + "best_transformer.pt")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_linear.state_dict(),
                }, output_folder + "best_linear.pt")

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_coda.state_dict(),
                }, output_folder + "best_coda.pt")
        

        torch.save({
                'epoch': epoch,
                'model_state_dict': model_whale.state_dict(),
                }, output_folder + "best_whale.pt")


    


import altair as alt
alt.data_transformers.disable_max_rows()

graph_df = pd.DataFrame()

graph_df["epoch"] = [i for i in range(num_epochs)]*2
graph_df["train or val"] = ["train"]*num_epochs+["val"]*num_epochs
graph_df["loss"] = all_train_total_loss + all_val_total_loss

loss_graph = alt.Chart(graph_df).mark_line().encode(
    x="epoch",
    y="loss",
    color="train or val"
).properties(title="Loss evolution")

loss_graph.save(output_folder+"total_losses.png")


graph_df = pd.DataFrame()

graph_df["epoch"] = [i for i in range(num_epochs)]*2
graph_df["train or val"] = ["train"]*num_epochs+["val"]*num_epochs
graph_df["loss"] = all_train_click_loss + all_val_click_loss

loss_graph = alt.Chart(graph_df).mark_line().encode(
    x="epoch",
    y="loss",
    color="train or val"
).properties(title="Loss evolution")

loss_graph.save(output_folder+"click_losses.png")

graph_df = pd.DataFrame()

graph_df["epoch"] = [i for i in range(num_epochs)]*2
graph_df["train or val"] = ["train"]*num_epochs+["val"]*num_epochs
graph_df["loss"] = all_train_coda_loss + all_val_coda_loss

loss_graph = alt.Chart(graph_df).mark_line().encode(
    x="epoch",
    y="loss",
    color="train or val"
).properties(title="Loss evolution")

loss_graph.save(output_folder+"coda_losses.png")


graph_df["epoch"] = [i for i in range(num_epochs)]*2
graph_df["train or val"] = ["train"]*num_epochs+["val"]*num_epochs
graph_df["loss"] = all_train_whale_loss + all_val_whale_loss

loss_graph = alt.Chart(graph_df).mark_line().encode(
    x="epoch",
    y="loss",
    color="train or val"
).properties(title="Loss evolution")

loss_graph.save(output_folder+"whale_losses.png")