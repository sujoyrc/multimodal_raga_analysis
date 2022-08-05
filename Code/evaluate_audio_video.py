import wandb
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath
import pdb

raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']
audio_table_names = [
    'snnithya/Gesture Analysis/Predictions-AG:v1',
    'snnithya/Gesture Analysis/Predictions-CC:v2',
    'snnithya/Gesture Analysis/Predictions-SCh:v0'
    ]
video_table_names = [
    'snnithya/Gesture Analysis/Predictions-AG:v5', 
    'snnithya/Gesture Analysis/Predictions-CC:v4',
    'snnithya/Gesture Analysis/Predictions-SCh:v3'
    ]
table_names = ['AG', 'CC', 'SCh']
destFolder = '../Final Video Data/AudioVsVideo/EnsembleModels/'
runName = 'Easy_1-ensemble-models'

def compare_preds(audio_table_names, video_table_names):
    with wandb.init(project='Gesture Analysis', name=runName) as run:
        video_df = None   # dataframe to store all video predictions
        audio_df = None   # dataframe to store all audio predictions
        for i in range(len(audio_table_names)):
            # video data
            video_table = run.use_artifact(video_table_names[i])
            # video_preds = video_table.get('val')
            video_preds = video_table.get('test_table')
            temp_vid_dict = {}
            for col in video_preds.columns:
                temp_vid_dict[col] = video_preds.get_column(col)
            temp_vid_dict['table_name'] = [video_table_names[i]] * len(temp_vid_dict['unique_id'])
            if video_df is None:
                video_df = pd.DataFrame(temp_vid_dict)
            else:
                video_df = pd.concat([video_df, pd.DataFrame(temp_vid_dict)])

            # audio data
            audio_table = run.use_artifact(audio_table_names[i])
            audio_preds = audio_table.get('test_table')
            temp_audio_dict = {}
            for col in audio_preds.columns:
                temp_audio_dict[col] = audio_preds.get_column(col)
            temp_audio_dict['table_name'] = [audio_table_names[i]] * len(temp_audio_dict['unique_id'])
            if audio_df is None:
                audio_df = pd.DataFrame(temp_audio_dict)
            else:
                audio_df = pd.concat([audio_df, pd.DataFrame(temp_audio_dict)])
        
        # set index of dfs
        audio_df = audio_df.set_index('unique_id')
        video_df = video_df.set_index('unique_id')
        # audio_df = audio_df.join(video_df.loc[:, 'true_label'], on='unique_id')

        # pdb.set_trace()
        # generate confusion matrices
        # audio
        accs = {}
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        i = 0
        for table_name, grouped_audio_df in audio_df.groupby('table_name'):
            cm = confusion_matrix(grouped_audio_df['true_class'], grouped_audio_df['predicted_class'])
            sns.heatmap(cm, annot=True, 
            fmt='d', xticklabels=raga_labels, yticklabels=raga_labels, ax=axs[i//2, i%2])
            axs[i//2, i%2].set_xlabel('Predicted Class')
            axs[i//2, i%2].set_ylabel('True Class')
            axs[i//2, i%2].set_title(table_names[audio_table_names.index(table_name)])
            accs[table_names[audio_table_names.index(table_name)]] = (cm.diagonal().sum())/cm.sum()
            i += 1
        cm = confusion_matrix(audio_df['true_class'], audio_df['predicted_class'])
        sns.heatmap(cm, annot=True, 
        fmt='d', xticklabels=raga_labels, yticklabels=raga_labels, ax=axs[1, 1])
        axs[1, 1].set_xlabel('Predicted Class')
        axs[1, 1].set_ylabel('True Class')
        axs[1, 1].set_title('All')
        fig.tight_layout()
        fig.savefig(checkPath(os.path.join(destFolder, 'audio_CM.png')))
        accs['All'] = (cm.diagonal().sum())/cm.sum()

        art = wandb.Artifact('AudioVideo', type='evaluation')
        art.add_file(os.path.join(destFolder, 'audio_CM.png'), name='audio_CM.png')
        with art.new_file('audio_CM.txt') as f:
            for k, v in accs.items():
                f.write(f'{k}: {v}\n')
        # video
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        i = 0
        accs = {}
        for table_name, grouped_video_df in video_df.groupby('table_name'):
            cm = confusion_matrix(grouped_video_df['true_class'], grouped_video_df['predicted_class'])
            sns.heatmap(cm, annot=True, 
            fmt='d', xticklabels=raga_labels, yticklabels=raga_labels, ax=axs[i//2, i%2])
            accs[table_names[video_table_names.index(table_name)]] = (cm.diagonal().sum())/cm.sum()
            axs[i//2, i%2].set_xlabel('Predicted Class')
            axs[i//2, i%2].set_ylabel('True Class')
            axs[i//2, i%2].set_title(table_names[video_table_names.index(table_name)])
            i += 1
        cm = confusion_matrix(video_df['true_class'], video_df['predicted_class'])
        sns.heatmap(cm, annot=True, 
        fmt='d', xticklabels=raga_labels, yticklabels=raga_labels, ax=axs[1, 1])
        accs['All'] = (cm.diagonal().sum())/cm.sum()
        axs[1, 1].set_xlabel('Predicted Class')
        axs[1, 1].set_ylabel('True Class')
        axs[1, 1].set_title('All')
        fig.tight_layout()
        fig.savefig(os.path.join(destFolder, 'video_CM.png'))

        art.add_file(os.path.join(destFolder, 'video_CM.png'), name='video_CM.png')
        with art.new_file('video_CM.txt') as f:
            for k, v in accs.items():
                f.write(f'{k}: {v}\n')

        # pdb.set_trace()
        # sample wise analysis
        incorrect_audios = audio_df.loc[audio_df['true_class'] != audio_df['predicted_class']]
        incorrect_videos = video_df.loc[video_df['true_class'] != video_df['predicted_class']]
        correct_audios = audio_df.loc[audio_df['true_class'] == audio_df['predicted_class']]
        correct_videos = video_df.loc[video_df['true_class'] == video_df['predicted_class']]

        c_video_i_audio = incorrect_audios.join(correct_videos, how='inner', lsuffix='_audio', rsuffix='_video')
        c_video_i_audio.to_csv(os.path.join(destFolder, 'c_video_i_audio.csv'))
        art.add_file(os.path.join(destFolder, 'c_video_i_audio.csv'), name='c_video_i_audio.csv')

        i_video_c_audio = correct_audios.join(incorrect_videos, how='inner', lsuffix='_audio', rsuffix='_video')
        i_video_c_audio.to_csv(os.path.join(destFolder, 'i_video_c_audio.csv'))
        art.add_file(os.path.join(destFolder, 'i_video_c_audio.csv'), name='i_video_c_audio.csv')

        i_video_i_audio = incorrect_audios.join(incorrect_videos, how='inner', lsuffix='_audio', rsuffix='_video')
        i_video_i_audio.to_csv(os.path.join(destFolder, 'i_video_i_audio.csv'))
        art.add_file(os.path.join(destFolder, 'i_video_i_audio.csv'), name='i_video_i_audio.csv')

        c_video_c_audio = correct_audios.join(correct_videos, how='inner', lsuffix='_audio', rsuffix='_video')
        c_video_c_audio.to_csv(os.path.join(destFolder, 'c_video_c_audio.csv'))
        art.add_file(os.path.join(destFolder, 'c_video_c_audio.csv'), name='c_video_c_audio.csv')

        accs = {}
        i=0
        fig, axs = plt.subplots(2, 2)
        for ind in range(len(video_table_names)):
            cm = np.array([
                [
                    i_video_i_audio.loc[i_video_i_audio['table_name_audio'] == audio_table_names[ind]].shape[0],
                    c_video_i_audio.loc[c_video_i_audio['table_name_audio'] == audio_table_names[ind]].shape[0]
                ],
                [
                    i_video_c_audio.loc[i_video_c_audio['table_name_audio'] == audio_table_names[ind]].shape[0],
                    c_video_c_audio.loc[c_video_c_audio['table_name_audio'] == audio_table_names[ind]].shape[0]
                ]
            ])
            accs[table_names[ind]] = (cm.diagonal().sum())/cm.sum()
            sns.heatmap(cm/cm.sum(), cmap='Oranges', annot=True, annot_kws={'fontsize': 'medium', 'fontweight': 'bold'},
            fmt='.2g', xticklabels=['Incorrect', 'Correct'], yticklabels=['Incorrect', 'Correct'], ax=axs[i//2, i%2])
            axs[i//2, i%2].set_xlabel('Video Predictions')
            axs[i//2, i%2].set_ylabel('Audio Predictions')
            axs[i//2, i%2].set_title(table_names[ind])
            i += 1
        cm = np.array([
            [
                i_video_i_audio.shape[0],
                c_video_i_audio.shape[0]
            ],
            [
                i_video_c_audio.shape[0],
                c_video_c_audio.shape[0]
            ]
        ])
        cm = cm/cm.sum()
        accs['All'] = (cm.diagonal().sum())/cm.sum()
        sns.heatmap(cm, cmap='Oranges', annot=True, annot_kws={'fontsize': 'medium', 'fontweight': 'bold'},
        fmt='.2g', xticklabels=['Incorrect', 'Correct'], yticklabels=['Incorrect', 'Correct'], ax=axs[1, 1])
        axs[1, 1].set_xlabel('Video Predictions')
        axs[1, 1].set_ylabel('Audio Predictions')
        axs[1, 1].set_title('All')
        fig.tight_layout()
        fig.savefig(os.path.join(destFolder, 'audio_vs_video_predictions.png'))

        art.add_file(os.path.join(destFolder, 'audio_vs_video_predictions.png'), name='audio_vs_video_predictions.png')
        with art.new_file('audio_vs_video_predictions.txt') as f:
            for k, v in accs.items():
                f.write(f'{k}: {v}\n')
        run.log_artifact(art)

compare_preds(audio_table_names, video_table_names)