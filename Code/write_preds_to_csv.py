import pandas as pd
import numpy as np
from write_pred_to_vide import read_preds
import wandb
from collections import defaultdict
from tqdm import tqdm

summary_file = '../Seqs/mod-summary.csv'
csvPath = '../Final Video Predictions/preds.csv'

arts = {
    "AG": {
        "A": "Predictions-AG:v1",
        "V": "Predictions-AG:v5",
        "AV": "Predictions-AG:v7"
    },
    "CC": {
        "A": "Predictions-CC:v2",
        "V": "Predictions-CC:v4",
        "AV": "Predictions-CC:v5"
    },
    "SCh": {
        "A": "Predictions-SCh:v0",
        "V": "Predictions-SCh:v3",
        "AV": "Predictions-SCh:v4"
    }
}

raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']

def main():
    summary = pd.read_csv(summary_file)

    data = defaultdict(list)
    with wandb.init(
        job_type='evaluation',
        project='Gesture Analysis'
    ) as run:
        for singer in list(arts.keys()):
            print(f'Processing {singer}')
            audio_df = read_preds(arts[singer]["A"], run)
            video_df = read_preds(arts[singer]["V"], run)
            audio_and_video_df = read_preds(arts[singer]["AV"], run)

            for id, row in tqdm(audio_df.iterrows()):
                u_id = row['unique_id']
                fileName = summary.loc[summary['unique_id'] == u_id, 'filename'].values[0].rsplit('/', 2)[1]
                start_time = summary.loc[summary["unique_id"] == u_id, 'start_times'].values[0]
                audio_pred = audio_df.loc[audio_df['unique_id'] == u_id]['predicted_class'].values[0]
                audio_pred_prob = audio_df.loc[audio_df['unique_id'] == u_id][f'prediction_probability_{audio_pred}'].values[0]
                video_pred = video_df.loc[video_df['unique_id'] == u_id]['predicted_class'].values[0]
                video_pred_prob = video_df.loc[video_df['unique_id'] == u_id][f'prediction_probability_{video_pred}'].values[0]
                audio_and_video_pred = audio_and_video_df.loc[audio_and_video_df['unique_id'] == u_id]['predicted_class'].values[0]
                audio_and_video_pred_prob = audio_and_video_df.loc[audio_and_video_df['unique_id'] == u_id][f'prediction_probability_{audio_and_video_pred}'].values[0]
                true_class = audio_df.loc[audio_df['unique_id'] == u_id]['true_class'].values[0]

                data['unique_id'].append(u_id)
                data['filename'].append(fileName)
                data['start time'].append(start_time)
                data['true class'].append(raga_labels[true_class])
                data['audio prediction'].append(raga_labels[audio_pred])
                data['audio prediction probability'].append(np.around(audio_pred_prob, 5))
                data['video prediction'].append(raga_labels[video_pred])
                data['video prediction probability'].append(np.around(video_pred_prob, 5))
                data['audio-video prediction'].append(raga_labels[audio_and_video_pred])
                data['audio-video prediction probability'].append(np.around(audio_and_video_pred_prob, 5))

        pd.DataFrame(data).to_csv(csvPath, index=False)

main()