'''
This script creates the dataset from the pitch/openpose csv files. 
'''

import argparse

from new_data_extraction_pipeline_audio import create_csv as create_csv_audio
from new_data_extraction_video_pipeline import replace_nan, filter_data, normalize_data
from new_data_extraction_video_pipeline import create_csv as create_csv_video
from new_data_extraction_combine_pipeline import train_test_split
import numpy as np
import os
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath
import pdb

arg_vals = sys.argv[1:]

parser = argparse.ArgumentParser(description='Generates dataset for multimodal analysis of song')
parser.add_argument('opt', default='BOTH', help='Can be "AUDIO", "VIDEO" or "BOTH" depending on the type of dataset required. "BOTH" indicates early fusion.')
parser.add_argument('summaryFile', help='path to the summary file')
parser.add_argument('ragaIndexFile', help='path to the raga index file')
parser.add_argument('split_folder', help='path to the folder with train-test splits')
parser.add_argument('-ai', '--audio_input', help='Folder with input pitch files')
parser.add_argument('-vi', '--video_input', help='Folder with input openpose files')
parser.add_argument('-o', '--output_folder', help='Folder where intermediate and output files will be stored')
parser.add_argument('-slen', '--subseq_len', default=600, help='path to the raga index file')

args = parser.parse_args(arg_vals)

# # generate audio csvs

# if args.opt == 'AUDIO' or args.opt == 'BOTH':
#     print('Creating audio csvs...')
#     create_csv_audio(
#         keywords = ['pitch', 'mask'], 
#         summaryFile = args.summaryFile, 
#         dataFolder = args.audio_input, 
#         destFolder = checkPath(os.path.join(args.output_folder, 'csvs')), 
#         subseq_len=args.subseq_len, 
#         class_index_file=args.ragaIndexFile
#         )

# generate video csvs
if args.opt == 'VIDEO' or args.opt == 'BOTH':
    print('Creating video csvs...')
    # create csv files
    create_csv_video(keywords=[
            ('RWrist', 'x'),
            ('RWrist', 'y'),
            ('LWrist', 'x'),
            ('LWrist', 'y')
        ], 
        summaryFile=args.summaryFile,
        dataFolder=args.video_input, 
        destFolder=checkPath(os.path.join(args.output_folder, 'csvs')),
        subseq_len=args.subseq_len, 
        class_index_file=args.ragaIndexFile, 
        replace=True,
        fps=np.around(args.subseq_len/12, 0)
        )

# create train test splits
print('Creating final npz files...')
train_test_split(
    splitFolder = args.split_folder,
    csvFolder = os.path.join(args.output_folder, 'csvs'), 
    destFolder = checkPath(os.path.join(args.output_folder, 'final_npzs')), 
    subseq_len=args.subseq_len, 
    option=args.opt
    )

