from automatic_annotation_pipeline import annotate
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10,11"

parser = argparse.ArgumentParser(description='Run the automatic click detection pipeline on a sound file.')

parser.add_argument('--input_file', type=str, help='Path to the input audio file, including the file name.',required=True)
parser.add_argument('--output_file', type=str, help='Desired path of the output file, including filename, should end in .csv. Default value is "full_pipeline_predictions.csv".',default="full_pipeline_predictions.csv")

parser.add_argument('--prediction_th', type=float, help='Phase 2 predictions with a confidence above this threshold will be in the final output. Default value is 0.5.',default=0.5)
parser.add_argument('--sending_th', type=float, help='Windows with a phase 1 prediction confidence above this threshold will be passed to phase 2. Default value is 0.7.',default=0.7)

parser.add_argument('--path_to_phase_1_soundnet_checkpoint',help='Path to the checkpoint for the SoundNet model used in phase 1. Default value is "phase_1_checkpoints/phase_1_soundnet.pt"', type=str,default="phase_1_checkpoints/phase_1_soundnet.pt")
parser.add_argument('--path_to_phase_1_mlp_checkpoint', help='Path to the checkpoint for the MLP model used in phase 1. Default value is "phase_1_checkpoints/phase_1_mlp.pt"', type=str,default="phase_1_checkpoints/phase_1_mlp.pt")

parser.add_argument('--path_to_phase_2_soundnet_checkpoint', help='Path to the checkpoint for the SoundNet model used in phase 2. Default value is "phase_2_checkpoints/phase_2_soundnet.pt"', type=str,default="phase_2_checkpoints/phase_2_soundnet.pt")
parser.add_argument('--path_to_phase_2_transformer_checkpoint', help='Path to the checkpoint for the transformer model used in phase 2. Default value is "phase_2_checkpoints/phase_2_transformer.pt"', type=str,default="phase_2_checkpoints/phase_2_transformer.pt")
parser.add_argument('--path_to_phase_2_linear_checkpoint', help='Path to the checkpoint for the linear model used in phase 2. Default value is "phase_2_checkpoints/phase_2_linear.pt"', type=str,default="phase_2_checkpoints/phase_2_linear.pt")
parser.add_argument('--path_to_phase_2_coda_checkpoint', help='Path to the checkpoint for the coda model used in phase 2. Default value is "phase_2_checkpoints/phase_2_coda.pt"', type=str,default="phase_2_checkpoints/phase_2_coda.pt")
parser.add_argument('--path_to_phase_2_whale_checkpoint', help='Path to the checkpoint for the whale model used in phase 2. Default value is "phase_2_checkpoints/phase_2_whale.pt"', type=str,default="phase_2_checkpoints/phase_2_whale.pt")


parser.add_argument('--store_phase_1_predictions', help='If you want an additional output file with all the candidate windows found by phase 1. Default value is False.',type=bool, default=False)
parser.add_argument('--store_all_phase_1_confidences', help='If you want an additional output file with the confidences of phase 1 for every window. Default value is False.', type=bool, default=False)
parser.add_argument('--store_phase_2_output', help='If you want an additional output file with the raw output of phase 2. Default value is False.', type=bool, default=False)

args = parser.parse_args()

annotate(input_file=args.input_file,
            output_file=args.output_file,
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


