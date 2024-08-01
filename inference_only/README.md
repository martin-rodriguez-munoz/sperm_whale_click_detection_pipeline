### Running on 1 file
In order to run the pipeline on a single file run the following command:
python run_pipeline_on_file.py --input_file="[Path to the input audio file, including the file name]"

If you want the output in a specific location
python run_pipeline_on_file.py --input_file="[Path to the file, including the file name]" --output_file="[Desired path of output, including filename, should end in .csv]"

If no output_file is specified the output file will be named "full_pipeline_predictions.csv" and be placed in the current folder.

### Running on an entire folder
In order to run the pipeline on an entire folder run the following command:
python run_pipeline_on_folder.py --input_folder="[Path to the folder with the input audio files]" --output_folder="[Path to folder where you want the output. Folder will be created if it doesn't exist]"

The output will have the name [Name of the file]_annotations.csv

### Settings
It also possible the adjust the following settings, both when running the pipeline on a single file and on an entire folder:

--prediction_th: Phase 2 predictions with a confidence above this threshold will be in the final output. Default value is 0.5.
--sending_th: Windows with a phase 1 prediction confidence above this threshold will be passed to phase 2. Default value is 0.7.

--path_to_phase_1_soundnet_checkpoint: Path to the checkpoint for the SoundNet model used in phase 1. Default value is "phase_1_checkpoints/phase_1_soundnet.pt".
--path_to_phase_1_mlp_checkpoint: Path to the checkpoint for the MLP model used in phase 1. Default value is "phase_1_checkpoints/phase_1_mlp.pt".
--path_to_phase_2_soundnet_checkpoint: Path to the checkpoint for the SoundNet model used in phase 2. Default value is "phase_2_checkpoints/phase_2_soundnet.pt".
--path_to_phase_2_transformer_checkpoint: Path to the checkpoint for the transformer model used in phase 2. Default value is "phase_2_checkpoints/phase_2_transformer.pt".
--path_to_phase_2_linear_checkpoint: Path to the checkpoint for the linear model used in phase 2. Default value is "phase_2_checkpoints/phase_2_linear.pt".
--path_to_phase_2_coda_checkpoint: Path to the checkpoint for the same coda prediction model used in phase 2. Default value is "phase_2_checkpoints/phase_2_coda.pt".
--path_to_phase_2_whale_checkpoint: Path to the checkpoint for the same whale prediction model used in phase 2. Default value is "phase_2_checkpoints/phase_2_whale.pt".

--store_phase_1_predictions: If you want an additional output file with all the candidate windows found by phase 1. Default value is False.
--store_all_phase_1_confidences: If you want an additional output file with the confidences of phase 1 for every window. Default value is False.
--store_all_phase_2_output: If you want an additional output with everything the transformer calculates during phase 2. Default value is False.

### Example commands
```
python run_pipeline_on_file.py --input_file="/raid/lingo/martinrm/original_data/dataset/2015/sw061b003.wav"
```
```
python run_pipeline_on_folder.py --input_folder="/raid/lingo/martinrm/original_data/dataset/all_birth/" --output_folder="all_birth_outputs/"
```

```
python run_pipeline_on_file.py --input_file="/raid/lingo/martinrm/original_data/dataset/2015/sw061b003.wav"
--output_file="sw061b003_lower_threshold_predictions.csv" --sending_th=0.4 --prediction_th=0.3
```

```
python run_pipeline_on_folder.py --input_folder="/raid/lingo/martinrm/original_data/dataset/all_birth" --output_folder="all_birth_outputs_more_info/" --store_phase_1_predictions=True --store_all_phase_1_confidences=True --store_all_phase_2_confidences=True --path_to_phase_1_soundnet_checkpoint="/raid/lingo/martinrm/for_github/click_detector_train_output/best_soundnet.pt" --path_to_phase_1_mlp_checkpoint="/raid/lingo/martinrm/for_github/click_detector_train_output/best_mlp.pt"
```

### Author
Created by Martín Rodríguez Muñoz during his stay as a visiting student at MIT under the supervision of Pratyusha Sharma and Professor Antonio Torralba.