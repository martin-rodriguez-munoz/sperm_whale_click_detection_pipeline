from automatic_annotation_pipeline import annotate
import os
import argparse

parser = argparse.ArgumentParser(description='Run the automatic click detection pipeline on a folder full of sound files.')

parser.add_argument('--input_folder', type=str, help='Path to input folder.',required=True)
parser.add_argument('--output_folder', type=str, help="Path to output folder. Folder will be created if it doesn't exist.",required=True)

parser.add_argument('--prediction_th', type=float, help='Phase 2 predictions with a confidence above this threshold will be in the final output. Default value is 0.5.',default=0.5)
parser.add_argument('--sending_th', type=float, help='Windows with a phase 1 prediction confidence above this threshold will be passed to phase 2. Default value is 0.7.',default=0.7)

parser.add_argument('--path_to_phase_1_soundnet_checkpoint',help='Path to the checkpoint for the SoundNet model used in phase 1. Default value is "phase_1_checkpoints/phase_1_soundnet.pt"', type=str,default="phase_1_checkpoints/phase_1_soundnet.pt")
parser.add_argument('--path_to_phase_1_mlp_checkpoint', help='Path to the checkpoint for the MLP model used in phase 1. Default value is "phase_1_checkpoints/phase_1_mlp.pt"', type=str,default="phase_1_checkpoints/phase_1_mlp.pt")

parser.add_argument('--path_to_phase_2_soundnet_checkpoint', help='Path to the checkpoint for the SoundNet model used in phase 2. Default value is "phase_2_checkpoints/phase_2_soundnet.pt"', type=str,default="phase_2_checkpoints/phase_2_soundnet.pt")
parser.add_argument('--path_to_phase_2_transformer_checkpoint', help='Path to the checkpoint for the transformer model used in phase 2. Default value is "phase_2_checkpoints/phase_2_transformer.pt"', type=str,default="phase_2_checkpoints/phase_2_transformer.pt")
parser.add_argument('--path_to_phase_2_linear_checkpoint', help='Path to the checkpoint for the linear model used in phase 2. Default value is "phase_2_checkpoints/phase_2_linear.pt"', type=str,default="phase_2_checkpoints/phase_2_linear.pt")
parser.add_argument('--path_to_phase_2_coda_checkpoint', help='Path to the checkpoint for the coda model used in phase 2. Default value is "phase_2_checkpoints/phase_2_coda.pt"', type=str,default="phase_2_checkpoints/phase_2_coda.pt")
parser.add_argument('--path_to_phase_2_whale_checkpoint', help='Path to the checkpoint for the coda model used in phase 2. Default value is "phase_2_checkpoints/phase_2_whale.pt"', type=str,default="phase_2_checkpoints/phase_2_whale.pt")

parser.add_argument('--store_phase_1_predictions', help='If you want an additional output file with all the candidate windows found by phase 1. Default value is False.',type=bool, default=False)
parser.add_argument('--store_all_phase_1_confidences', help='If you want an additional output file with the confidences of phase 1 for every window. Default value is False.', type=bool, default=False)
parser.add_argument('--store_phase_2_output', help='If you want an additional output file with the raw output of phase 2. Default value is False.', type=bool, default=False)

args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

k = 0
print(input_folder)
for input_file_name in os.listdir(input_folder):
    k += 1
    print("Processing",input_file_name,k,"/",len(os.listdir(input_folder)))
    input_file_path = os.path.join(input_folder,input_file_name)
    output_file_name = input_file_name[:-4]+"_annotations.csv"
    output_file_path = os.path.join(output_folder,output_file_name)

    print(input_file_name)
    annotate(input_file=input_file_path,
                        output_file=output_file_path,
                        prediction_th=args.prediction_th,
                        sending_th=args.sending_th,
                        path_to_phase_1_soundnet_checkpoint=args.path_to_phase_1_soundnet_checkpoint,
                        path_to_phase_1_mlp_checkpoint=args.path_to_phase_1_mlp_checkpoint,
                        path_to_phase_2_soundnet_checkpoint=args.path_to_phase_2_soundnet_checkpoint,
                        path_to_phase_2_transformer_checkpoint=args.path_to_phase_2_transformer_checkpoint,
                        path_to_phase_2_linear_checkpoint=args.path_to_phase_2_linear_checkpoint,
                        path_to_phase_2_coda_checkpoint=args.path_to_phase_2_coda_checkpoint,
                        path_to_phase_2_whale_checkpoint=args.path_to_phase_2_whale_checkpoint,
                        store_phase_1_predictions=args.store_phase_1_predictions,
                        store_all_phase_1_confidences=args.store_all_phase_1_confidences,
                        store_phase_2_output=args.store_phase_2_output)


