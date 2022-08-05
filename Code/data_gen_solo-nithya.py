#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pdb
import random
import pandas as pd
import numpy as np
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath


##################################################################
##                                                              ##
## generate clips available for the MS-G3D from the CSV files   ##
## the csv contains one person's 2d skeleton                    ##
##                                                              ##
##################################################################

# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8, "REye"},
# {9, "LEye"},
# {10, "LEye"},
RAGA = [
    "Bag",
    "Bahar",
    "Bilas",
    "Jaun",
    "Kedar",
    "MM",
    "Marwa",
    "Nand",
    "Shree"
]
MUSICIAN = ["AG", "CC", "SCh"]

# path for the CSV source
csv_dir = "../Video Data/"
# path for the output dir
output_data_dir = checkPath("../Seqs/JSON-Video/Data Raw/")
output_label_dir = checkPath("../Seqs/JSON-Video/Label Raw/")
summary_file = '../Seqs/summary.csv'
rewrite = False     # if true will rewrite the entire directory, else will append to label file
length = 300  # number of frames of each clip
stride = 40  # stride for the start time for each clip

def read_csv(filename):
    """

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    df_new : dataframe
        Normalised coordinates of 3D pose.

    """
    dataframe = pd.read_csv(filename, index_col="Body Part")
    # find the bbox of the player for crop
    xmax = 0
    ymax = 0
    xmin = 10000
    ymin = 10000

    for key in dataframe.keys(): 
        # example so keys for LEar would be [LEar, LEar.1, LEar.2]
        data = list(dataframe[key][1:])
        data = list(map(float, data))
        data_num = np.array(data)
        data_num = data_num[np.where(~np.isnan(data_num))]

        keys = key.split(".")
        if len(keys) == 1:
            key_new = (keys[0], "x")
            xmax = max(xmax, np.max(data_num))
            xmin = min(xmin, np.min(data_num))

        elif len(keys) == 2 and keys[1] == "1":
            key_new = (keys[0], "y")
            ymax = max(ymax, np.max(data_num))
            ymin = min(ymin, np.min(data_num))

        if key == "MidHip":
            data_midhip = data_num
        if key == "Neck":
            data_neck = data_num

    xc = (np.mean(data_neck) + np.mean(data_midhip)) / 2
    width = 2 * max(xc - xmin, xmax - xc)
    height = ymax - ymin

    df = dict()
    for key in dataframe.keys():
        data = list(dataframe[key][1:])
        data = list(map(float, data))   # converts the data to float
        nan_idx = np.where(np.isnan(data))[0]
        if len(nan_idx) == len(data):
            # if all values are nan
            data[:] = 0
        elif len(nan_idx) > 0:
            # replace nan values in csv with previous (non-nan) value/for the first index, the next closest non-nan value
            for jj in nan_idx:
                if jj == 0:
                    data[jj] = np.where(~np.isnan(data))[0][0]
                else:
                    data[jj] = data[jj - 1]

        keys = key.split(".")
        if len(keys) == 1:
            key_new = (keys[0], "x")
            data = np.round(list((np.array(data) - xmin) / (width)), 5)

        elif len(keys) == 2 and keys[1] == "1":
            key_new = (keys[0], "y")
            data = np.round(list((np.array(data) - ymin) / height), 5)
        else:
            key_new = (keys[0], "c")
            data = np.array(data)

        df[key_new] = data
    df_new = pd.DataFrame(df)   # this dataframe has removed nans
    return df_new


def generate_json(df, file):
    """

    Parameters
    ----------
    df : dataframe
        The skeleton data.
    file: str
        The file name that provides the label information

    Returns
    -------
    json_data : dict
        The data written to the json file

    """
    file = file.split("_")
    json_data = dict()
    json_data["data"] = []
    json_data["label"] = file[2]
    json_data["label_index"] = RAGA.index(file[2])

    json_data["musician"] = file[0]
    json_data["musician_index"] = MUSICIAN.index(file[0])

    for ii in range(len(df)):
        # creating data for each row
        row = df.iloc[ii]
        data = dict()
        data["frame_index"] = ii
        data["skeleton"] = []
        skeleton = dict()
        skeleton["pose"] = [
            row["Nose"]["x"],
            row["Nose"]["y"],
            row["Neck"]["x"],
            row["Neck"]["y"],
            row["RShoulder"]["x"],
            row["RShoulder"]["y"],
            row["RElbow"]["x"],
            row["RElbow"]["y"],
            row["RWrist"]["x"],
            row["RWrist"]["y"],
            row["LShoulder"]["x"],
            row["LShoulder"]["y"],
            row["LElbow"]["x"],
            row["LElbow"]["y"],
            row["LWrist"]["x"],
            row["LWrist"]["y"],
            row["REye"]["x"],
            row["REye"]["y"],
            row["LEye"]["x"],
            row["LEye"]["y"],
            row["MidHip"]["x"],
            row["MidHip"]["y"],
        ]

        skeleton["score"] = [
            row["Nose"]["c"],
            row["Neck"]["c"],
            row["RShoulder"]["c"],
            row["RElbow"]["c"],
            row["RWrist"]["c"],
            row["LShoulder"]["c"],
            row["LElbow"]["c"],
            row["LWrist"]["c"],
            row["REye"]["c"],
            row["LEye"]["c"],
            row["MidHip"]["c"],
        ]
        data["skeleton"].append(skeleton)
        json_data["data"].append(data)
    return json_data

label_data = dict()

summary = pd.read_csv(summary_file)
for filename, filename_df in summary.groupby('filename'):
    # for each filename in summary
    try:
        dataframe = read_csv(filename.replace('../Data/', '../Video Data/').rsplit('/', 1)[0] + '.csv')
    except:
        print(f'Corresponding {filename} is not available. Skipping...')
    for ind_row, row in filename_df.iterrows():
        print(row['unique_id'])
        json_file = os.path.join(
            output_data_dir, row['unique_id'] + ".json"
        )
        if os.path.exists(json_file) and not rewrite:
            print('Skipping rewrite of ' + json_file)
            continue
        
        start_frame = np.around(row['start_times']*25).astype(int)
        df = dataframe.iloc[start_frame : start_frame + length]
        json_data = generate_json(df, row['unique_id'])
        # write label json
        key = row['unique_id']
        # raga and musician
        label_raga = dict()
        label_raga["has_skeleton"] = True
        label_raga["label"] = json_data["label"]
        label_raga["label_index"] = json_data["label_index"]
        label_raga["musician"] = json_data["musician"]
        label_raga["musician_index"] = json_data["musician_index"]
        label_data[key] = label_raga
        
        with open(checkPath(json_file), "w") as f:
            json.dump(json_data, f)
if not rewrite:
    if os.path.exists(output_label_dir + "music_solo_label.json"):
        with open(output_label_dir + "music_solo_label.json", 'r') as f:
            json_data = json.load(f)
    for key in list(label_data.keys()):
        json_data[key] = label_data[key]
    with open(output_label_dir + "music_solo_label.json", "w") as f:
        json.dump(json_data, f, indent=4)
else:
    with open(output_label_dir + "music_solo_label.json", "w") as f:
        json.dump(label_data, f, indent=4)