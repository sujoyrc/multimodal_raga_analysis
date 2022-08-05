import wandb
import pandas as pd
import os
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath

video = 'snnithya/Gesture Analysis/Predictions-SCh:v3'
audio = 'snnithya/Gesture Analysis/Predictions-SCh:v0'
destFolder = checkPath('../Final Video Data/PredExamples/ensemble-SChdata/')
# os.mkdir(destFolder)

def df_extract(art):
    temp_dict = {}
    for col in art.columns:
        temp_dict[col] = art.get_column(col)
    return pd.DataFrame(temp_dict)

with wandb.init(project='Gesture Analysis') as run:
    video_train = {}
    video_test = {}
    audio__train = {} 
    audio_test = {}

    video_table = run.use_artifact(video)
    video_train_preds = video_table.get('train_table')
    video_train_df = df_extract(video_train_preds)
    video_test_preds = video_table.get('test_table')
    video_test_df = df_extract(video_test_preds)

    audio_table = run.use_artifact(audio)
    audio_train_preds = audio_table.get('train_table')
    audio_train_df = df_extract(audio_train_preds)
    audio_test_preds = audio_table.get('test_table')
    audio_test_df = df_extract(audio_test_preds)

    video_train_df.to_csv(os.path.join(destFolder, 'v_train.csv'), index=False)
    video_test_df.to_csv(os.path.join(destFolder, 'v_test.csv'), index=False)
    audio_train_df.to_csv(os.path.join(destFolder, 'a_train.csv'), index=False)
    audio_test_df.to_csv(os.path.join(destFolder, 'a_test.csv'), index=False)
