import cv2
import wandb
import pandas as pd
import ffmpeg
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath

import pdb

summary_file = '../Seqs/mod-summary.csv'

video_preds_art =  'Predictions-SCh:v3'
audio_preds_art = 'Predictions-SCh:v0'
audio_and_video_preds_art = "Predictions-SCh:v4"

video_folder = '../Data/Alap/'
dest_folder = '../Final Video Predictions/VideoOverlay/A_V_AV/SCh/'
dest_final_folder =  '../Final Video Predictions/VideoOverlayWithAudio/A_V_AV/SCh/'

raga_labels = ['Bag', 'Bahar', 'Bilas', 'Jaun', 'Kedar', 'MM', 'Marwa', 'Nand', 'Shree']

def read_preds(art_name, run):
    '''
    Returns predictions from wandb as a dataframe
    '''
    table = run.use_artifact(f'snnithya/Gesture Analysis/{art_name}')
    preds = table.get('test_table')
    temp_dict = {}
    for col in preds.columns:
        temp_dict[col] = preds.get_column(col)
    return process_df(pd.DataFrame(temp_dict))

def process_df(df):
    filenames = []
    ids = []
    for ind, row in df.iterrows():
        filename, id = row['unique_id'].rsplit('_', 1)
        filenames.append(filename)
        ids.append(id)
    df['filename'] = filenames
    df['id_num'] = ids
    df['id_num'] = df['id_num'].astype(int)

    return df

def prepare_write_items(summary_df, audio_df, video_df, audio_and_video_df):
    summary_df = summary_df.sort_values('Mid Frame Number') # sort acc to frame number
    data = []
    sum_ind = 0
    current_mid_num = summary_df['Mid Frame Number'].values[0]
    for i in range(int(summary_df['Mid Frame Number'].values[-1])):
        if i == current_mid_num:
            sum_ind += 1
            try:
                current_mid_num = summary_df['Mid Frame Number'].values[sum_ind]
            except:
                pdb.set_trace()
        u_id = summary_df.iloc[sum_ind, 0]
        audio_pred = audio_df.loc[audio_df['unique_id'] == u_id]['predicted_class'].values[0]
        audio_pred_prob = audio_df.loc[audio_df['unique_id'] == u_id][f'prediction_probability_{audio_pred}'].values[0]
        video_pred = video_df.loc[video_df['unique_id'] == u_id]['predicted_class'].values[0]
        video_pred_prob = video_df.loc[video_df['unique_id'] == u_id][f'prediction_probability_{video_pred}'].values[0]
        audio_and_video_pred = audio_and_video_df.loc[audio_and_video_df['unique_id'] == u_id]['predicted_class'].values[0]
        audio_and_video_pred_prob = audio_and_video_df.loc[audio_and_video_df['unique_id'] == u_id][f'prediction_probability_{audio_and_video_pred}'].values[0]
        true_class = audio_df.loc[audio_df['unique_id'] == u_id]['true_class'].values[0]
        
        data.append([
            u_id,
            raga_labels[audio_pred],
            np.around(audio_pred_prob, 2),
            raga_labels[video_pred],
            np.around(video_pred_prob, 2),
            raga_labels[audio_and_video_pred],
            np.around(audio_and_video_pred_prob, 2),
            raga_labels[true_class]
        ])
        if i == current_mid_num:
            sum_ind += 1
            current_mid_num = summary_df['Mid Frame Number'].values[sum_ind]
        
    return data

def write_vid(src_path, dest_path, dest_final_path, data):
    cap = cv2.VideoCapture(src_path)
    out = cv2.VideoWriter(dest_path,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1920,1080))
    # extend the data to the length of the capture
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data.extend([data[-1]] * (vid_len - len(data)))
    ind = 0
    with tqdm(total=vid_len) as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                font = cv2.FONT_HERSHEY_SIMPLEX

                # id
                cv2.putText(
                    frame,
                    f'Sequence id: {data[ind][0]}',
                    (50, 50),
                    font, 1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4
                )

                # Audio predictions
                cv2.putText(
                    frame,
                    f'Audio Prediction: {data[ind][1]} ({data[ind][2]})',
                    (50, 150),
                    font, 1,
                    (0, 0, 255) if data[ind][1] != data[ind][7] else (0, 255, 0),
                    2,
                    cv2.LINE_4
                )

                # video predictions
                cv2.putText(
                    frame,
                    f'Video Prediction: {data[ind][3]} ({data[ind][4]})',
                    (50, 250),
                    font, 1,
                    (0, 0, 255) if data[ind][3] != data[ind][7] else (0, 255, 0),
                    2,
                    cv2.LINE_4
                )

                # A+V prediction
                cv2.putText(
                    frame,
                    f'A+V Prediction: {data[ind][5]} ({data[ind][6]})',
                    (50, 350),
                    font, 1,
                    (0, 0, 255) if data[ind][5] != data[ind][7] else (0, 255, 0),
                    2,
                    cv2.LINE_4
                )

                # True label
                cv2.putText(
                    frame,
                    f'True label: {data[ind][7]}',
                    (50, 450),
                    font, 1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4
                )
                
                out.write(frame)
            else:
                break
            ind += 1
            pbar.update()
        
    cap.release()
    cv2.destroyAllWindows()
    combineAudioVideo(dest_path, src_path, dest_final_path)

def combineAudioVideo(vid_path, audio_path, dest_path):
	'''Function to combine audio and video into a single file. 

	Parameters
	----------
		vid_path    : str
			File path to the video file with squares.
		
		audio_path    : str
			File path to the audio file with clicks.

		dest_path    : str
			File path to store the combined file at.

	Returns
	-------
		None

	'''
	
	vid_file = ffmpeg.input(vid_path)
	audio_file = ffmpeg.input(audio_path)
	(
		ffmpeg
		.concat(vid_file.video, audio_file.audio, v=1, a=1)
		.output(dest_path)
		.overwrite_output()
		.run()
	)
	print('Video saved at ' + dest_path)

def main():
    summary = pd.read_csv(summary_file)

    with wandb.init(
        job_type='evaluation',
        project='Gesture Analysis'
    ) as run:
        audio_preds = read_preds(audio_preds_art, run)
        video_preds = read_preds(video_preds_art, run)
        audio_and_video_preds = read_preds(audio_and_video_preds_art, run)
        for filename, filename_audio_df in audio_preds.groupby('filename'):
            print('Processing file: ' + filename)
            filename_video_df = video_preds.loc[video_preds['filename'] == filename]
            filename_audio_and_video_df = audio_and_video_preds.loc[audio_and_video_preds['filename'] == filename]
            summary_filename = summary.loc[np.isin(summary['unique_id'], filename_audio_df.unique_id)]
            data = prepare_write_items(summary_filename, filename_audio_df, filename_video_df, filename_audio_and_video_df)

            src_path = checkPath(os.path.join(video_folder, filename, f'{filename}.mp4'))
            dest_path = checkPath(os.path.join(dest_folder, f'{filename}.avi'))
            dest_final_path = checkPath(os.path.join(dest_final_folder, f'{filename}.mp4'))
            write_vid(src_path, dest_path, dest_final_path, data)

# def test():
#     combineAudioVideo('/home/nithya/Projects/Gesture Analysis/Final Video Predictions/VideoOverlay/AG_1b_Jaun.avi', '/home/nithya/Projects/Gesture Analysis/Data/Alap/AG_1b_Jaun/AG_1b_Jaun.mp4', '/home/nithya/Projects/Gesture Analysis/Final Video Predictions/VideoOverlayWithAudio/AG_1b_Jaun.mp4')
# test()

main()