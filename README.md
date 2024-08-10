# Automatic whale language detection pipeline
This code detects communications done by sperm whales within audio.
The pipeline works in three phases:
1) The first phase identifies audio windows that may contain a sperm whale click within the entire audio file. These become our click candidates.

![image](https://github.com/user-attachments/assets/08ec682f-4111-44c8-be68-16bd949b232a)

2) The second phase looks at multiple click candidates at the same time to give each of them a more informed click probability. It also calculates the pairwise probability of two click candidates belonging to the same coda and the pairwise probability of two click candidates being said by the same speaker.
   
![image](https://github.com/user-attachments/assets/2d5f2e9f-5086-4d3d-a88b-794c7f5badf3)

3) The third phase takes the click candidates that were assigned a high revised probability in phase 2, clusters them by coda, and then clusters the codas by speaker. This give us the final output which consists of the predicted codas with their click times and speaker id.

![image](https://github.com/user-attachments/assets/88572008-eebc-4b37-8ed4-2b02d2b5e964)


## Inference

If you are only interested in running the pipeline, check out `inference_only/audio_file_to_annotations_and_book_of_whales.ipynb` for a simple guide on how to get annotations from an audio file, and how to get a book of whales plot from an annotation files. Alternatively, check out `inference_only/read_me.md` for a guide on how to extract annotatons from an audio file or a folder of audio files using the command line.

In order to run the pipeline you will need to download the checkpoints from the releases section of this github (see the image below) or train your own.
![image](https://github.com/user-attachments/assets/220c00e1-e4b9-4b9b-aed6-3f20d3c07f3b)


## Training

If you wish to train the pipeline, see `training_and_evaluation/training_on_your_own_dataset.ipynb` for a guide on how to format your dataset and which files to run in order to train the full pipeline.

## Evaluation

In order to evaluate the pipeline, see `training_and_evaluation/evaluation.ipynb`. Note that you in order to get results you must first have ran inference on the dataset by running `training_and_evaluation/candidate_revision_inference.py`.

### Author
Created by Martín Rodríguez Muñoz during his stay as a visiting student at MIT under the supervision of Pratyusha Sharma and Professor Antonio Torralba.
